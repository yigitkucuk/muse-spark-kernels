# okc: problem=muse_bf16_proj
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# ---- Third-party notice: Nathen Fernandes routes are derived from
# https://github.com/NathenFernandes/muse-bf16-proj, used under the MIT License below:
# MIT License
#
# Copyright (c) 2026 Nathen Fernandes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---- end of third-party notice
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
# This file has been modified from the original.
_PRUNE = True   # prepare only the cells the final _TABLE routes to each kernel (benchmark prepare limit 300 s)


def _make_streamk():
    """Stream-K skinny GEMM: out[m, n] = x @ W[:n].T for m <= 32 (and the router at any m).
W is repacked into mma.sync A-fragment order (2 KB contiguous per warp per 64-wide K chunk); every warp streams an
equal contiguous range of (tile, chunk) units; tiles split across warps meet through fp32 atomics and the last
arriver writes the output. The FP32 router runs as a BF16 hi + lo pair with FP32 accumulation.
v13/v14: up to 32 tokens, the BF16 projections and the router (12-bit hi + lo) run a register-direct GEMV on losslessly 12-bit-packed weights (see GVCU),
launch geometry picked per shape in prepare() by timing against the v12 path.
the same gv kernel with W streamed through a private per-warp cp.async.bulk smem ring (V bit 3, paths gb-*);
decode, mma, escapes, reduction and epilogue unchanged. PROBE gvbulk lines compare it with r6 on pooler m=1.
r6 unchanged (its 12-bit paths, bake and dispatch are SASS-identical) plus codec E paths
(gv FMT 4, "ge-*"). The code length is fixed by a code's first 3 bits (LOP3 + PRMT on an 8-byte table), so the per-lane
bit-position chain has no memory access. The exponent pair value comes from a 256-entry smem LUT (single or lane-replicated)
off the chain. There are 1, 2 or 4 independent substreams per lane, and refill words come through a conflict-free
[substream][slot][lane] 4 B cp.async ring and are prefetched. 11.07 / 11.17 / 11.32 bits/weight (K 1 / 2 / 4) against 12
(proj_rsi/tools/CODEC_E_LAYOUT.md v3). Each pack is proven bit-exact by the reference unpacker before any E kernel runs. Each
path is validated like every second-bake path and timed against r6's pick (BAKE / PICK). PROBE ecod lines time the best E path
against r6 on pooler m=1 and m=16, alternated."""
    from typing import TypedDict
    import math
    import statistics
    import time
    import torch
    _T0 = time.time()
    from torch.utils.cpp_extension import load_inline

    def _NC(layer, m):
        return (not _PRUNE) or _TABLE.get('%s:%d' % (layer, m), _DEFAULT) == 'streamk'

    def _NCR(layer, ms):
        return any(_NC(layer, m_) for m_ in ms)


    class Weights(TypedDict):
        qkv_gate: torch.Tensor
        o: torch.Tensor
        pooler: torch.Tensor
        unpooler: torch.Tensor
        router: torch.Tensor


    class Config(TypedDict):
        m: int
        n: int
        k: int
        layer: str


    class Inputs(TypedDict):
        x: torch.Tensor
        config: Config


    CUDA = __cuda__("streamk.cu")

    # ---------------------------------------------------------------- v13/v14: register-direct swap-AB GEMV on 12-bit-packed BF16
    # A unit is one 16-row tile x one 64-wide K chunk in mma.sync A-fragment order, per lane 3 x 16 B (c12) or 4 x 16 B (bf16):
    # c12 keeps each value's low byte (exponent LSB + mantissa) and a nibble (sign + 3-bit offset of the upper 7 exponent bits
    # from a per-matrix window); out-of-window values are fixed from a per-warp escape list (exact). FMT 2/3: the router's
    # BF16 hi + lo pair (bf16 / c12), fp32 out. Warp (wr, wk) of CTA (bx, by) streams tiles (bx * WR + wr) * RT + j, chunks
    # [(by * WK + wk) * KU, + KU) straight into registers; x comes from registers (XS 0) or a CTA-shared smem slab (XS 1);
    # partials meet in smem, then across the S CTAs of a cluster through DSMEM (sm_90) or a last-arriver workspace (older GPUs).
    GVCU = __cuda__("streamk_gvcu.cu")
    # candidate launch geometries (WR, WK, KU, S, RT, XS) per (layer key, MT); FMT is c12 (1/3) with a bf16 (0/2) twin of the
    # first geometry as the fallback when a 12-bit pack fails
    # candidates per (layer key, MT): (WR, WK, KU, S, RT, XS) -> kernel variants V (bit 0: 9-op add decode, else 11-op table
    # decode; bit 1: rolled K loop); H200 v15 benchmark: the v14 kernel (V 0) was faster at deep K, the rolled/9-op one (V 3) at m32
    _0, _1, _01 = (0,), (1,), (0, 1)
    GVC = {
        ("qkv_gate", 1): [((2, 16, 8, 1, 1, 2), _1), ((1, 8, 16, 1, 1, 0), _1), ((1, 16, 8, 1, 1, 2, 1), (1 | 2 << 12,)),
                          ((1, 4, 16, 2, 1, 0), _01), ((1, 4, 8, 4, 1, 0), _1)],
        ("qkv_gate", 2): [((2, 8, 16, 1, 1, 0, 2), _1), ((1, 4, 16, 2, 1, 0), _1), ((1, 4, 8, 4, 1, 2), (1, 3)),
                          ((1, 4, 16, 2, 1, 2), _01), ((1, 4, 16, 2, 1, 2, 1), _1), ((1, 4, 8, 4, 1, 2, 1), _1),
                          ((2, 16, 8, 1, 1, 2), (3,)), ((1, 16, 8, 1, 1, 2, 1), (1 | 2 << 12,))],
        ("qkv_gate", 4): [((1, 8, 16, 1, 1, 2), _01), ((1, 4, 16, 2, 1, 2), _01), ((2, 4, 16, 2, 1, 2), _1),
                          ((1, 4, 8, 4, 1, 2, 1), _01), ((1, 4, 8, 4, 1, 2), _1), ((1, 4, 16, 2, 1, 2, 1), _1)],
        ("o", 1): [((1, 8, 2, 1, 1, 0), _1), ((1, 8, 2, 1, 1, 0), _0), ((2, 4, 4, 1, 1, 0), _1), ((1, 4, 4, 1, 1, 0), _1)],
        ("o", 2): [((1, 8, 2, 1, 1, 0), _1), ((1, 8, 2, 1, 1, 0), _0), ((1, 8, 2, 1, 1, 2), _1), ((2, 4, 4, 1, 1, 0), _1),
                   ((2, 4, 4, 1, 1, 2), _1), ((1, 4, 4, 1, 1, 2, 1), _1)],
        ("o", 4): [((1, 4, 4, 1, 1, 2), _01), ((2, 4, 4, 1, 1, 2), _01), ((1, 8, 2, 1, 1, 0), _1),
                   ((1, 4, 4, 1, 1, 2, 1), _01), ((1, 8, 2, 1, 1, 2), _1), ((1, 8, 2, 1, 1, 2, 1), _1)],
        ("pooler", 1): [((2, 16, 8, 1, 1, 0), (3,)), ((2, 8, 16, 1, 1, 0), _1), ((2, 8, 16, 1, 1, 0), _0),
                        ((1, 8, 16, 1, 1, 0), _1)],
        ("pooler", 2): [((2, 8, 16, 1, 1, 0), _1), ((2, 8, 16, 1, 1, 0), _0), ((1, 8, 16, 1, 1, 2), _1),
                        ((1, 8, 16, 1, 1, 0), _1)],
        ("pooler", 4): [((1, 8, 16, 1, 1, 2), _0), ((1, 8, 16, 1, 1, 2), _1), ((1, 4, 16, 2, 1, 2), _1),
                        ((2, 4, 16, 2, 1, 2), _1)],
        # unpooler_s serves m=1 AND m=8: proven = v0 at m=1, v1 at m=8 (one list, so only v0 can be literally first;
        # the pick rule matches the proven entry by NAME, so both hold)
        ("unpooler_s", 1): [((1, 16, 4, 1, 1, 0), _0), ((1, 16, 4, 1, 1, 0), _1), ((1, 8, 2, 4, 1, 0), _1),
                            ((1, 8, 4, 2, 1, 0), _1), ((1, 16, 4, 1, 1, 2), _1)],
        ("unpooler", 2): [((1, 4, 16, 1, 1, 2), _1), ((1, 4, 16, 1, 1, 2), _0), ((1, 4, 16, 1, 1, 0), _1),
                          ((1, 8, 8, 1, 1, 0), _1)],
        ("unpooler", 4): [((1, 4, 16, 1, 1, 2), _1), ((1, 4, 16, 1, 1, 2), _0), ((2, 4, 16, 1, 1, 2), _1),
                          ((1, 8, 8, 1, 1, 2), _1)],
        # router: plain BF16 hi + lo fragments (FMT 2), every warp's units in flight at once, split-K over a 16-CTA cluster
        ("router", 1): [((1, 8, 2, 8, 1, 0), _0), ((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), _0)],
        ("router", 2): [((1, 8, 2, 8, 1, 0), _0), ((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), _0),
                        ((1, 8, 2, 8, 1, 0), (1 << 26,)), ((1, 8, 2, 8, 1, 0), (1 << 13,)),
                        ((1, 8, 4, 4, 1, 0), _0), ((2, 8, 2, 8, 1, 0), _0)],
        ("router", 4): [((1, 4, 2, 16, 1, 0), _0), ((1, 8, 2, 8, 1, 0), _0), ((1, 8, 1, 16, 1, 0), _0),
                        ((1, 4, 2, 16, 1, 0), (1 << 26,)), ((1, 4, 4, 8, 1, 0), _0)],
    }
    PROVEN_GV = {
        ("qkv_gate", 1): "1.1.1.2.16.8.1.1.2.1", ("qkv_gate", 8): "1.1.1.2.16.8.1.1.2.1",
        ("qkv_gate", 16): "1.2.2.2.8.16.1.1.0.1", ("qkv_gate", 32): "v12",
        ("o", 1): "1.1.2.1.8.2.1.1.0.1", ("o", 8): "1.1.2.1.8.2.1.1.0.1", ("o", 16): "1.2.2.1.8.2.1.1.0.1",
        ("o", 32): "v12",
        ("pooler", 1): "1.1.1.2.16.8.1.1.0.3", ("pooler", 8): "1.1.1.2.16.8.1.1.0.3",
        ("pooler", 16): "1.2.2.2.8.16.1.1.0.1", ("pooler", 32): "1.4.2.1.8.16.1.1.2.0",
        ("unpooler", 1): "1.1.2.1.16.4.1.1.0.0", ("unpooler", 8): "1.1.2.1.16.4.1.1.0.1",
        ("unpooler", 16): "1.2.2.1.4.16.1.1.2.1", ("unpooler", 32): "1.4.2.1.4.16.1.1.2.1",
        ("router", 1): "rr6", ("router", 8): "2.1.2.1.8.2.8.1.0.0", ("router", 16): "2.2.2.1.8.2.8.1.0.0",
        ("router", 32): "2.4.2.1.4.2.16.1.0.0",
    }
    # the proven pick per gx group (unpooler m1/m8 = uk-s4x); every other gx group's proven pick is r6 (no gx pick).
    FORCE_GX = {("unpooler", 8)}
    PROVEN_GX = {("unpooler", 1): "uk-s4x", ("unpooler", 8): "uk-s4x",
                 ("router", 8): "rk-s16w8x", ("router", 16): "rk-s16w8z", ("router", 32): "rk-s8z"}
    # the proven pick per rq32 group; a group missing here was picked as r6 (no rq path).
    PROVEN_RQ32 = {("qkv_gate", 32): "t32s6hze-k7", ("qkv_gate", 16): "t16s4hze-k7", ("o", 32): "t32s6hze-k2",
                   ("pooler", 32): "t32s7hze-k4", ("unpooler", 32): "t32s6hze-k2"}
    GVRB = []  # [(128, 4), (256, 4)]
    GVB = {1: 1, 8: 1, 16: 2, 32: 4}  # bucket m -> MT (the kernel also runs m > 8 MT as token blocks in grid.x; slower than cuBLAS)


    def _gvst(geo):  # a 7th geometry field forces the register-ring depth
        if len(geo) == 7:
            return geo[6]
        wr, wk, ku, s, rt, xs = geo
        return 1 if wr * wk >= 32 else 3 if xs == 1 and rt == 1 else 2


    def _gvinst():
        out = []
        for (key, mt), geos in GVC.items():
            for k, (geo, vs) in enumerate(geos):
                if key == "router":
                    out.append((2, mt, _gvst(geo)) + geo[:6] + (0,))
                    continue
                out += [(1, mt, _gvst(geo)) + geo[:6] + (v,) for v in vs] + ([(0, mt, _gvst(geo)) + geo[:6] + (0,)] if k == 0 else [])
        return list(dict.fromkeys(out))


    GX = True
    GXP = []
    GX11 = False
    for _ms, _xs in (((1, 8), 0), ((16, 32), 2)):
        for _wr in (9, 8):  # 144 rows: 16 x 8 = 128 CTAs; 128 rows: 2304 = 18 x 128 -> 144 CTAs
            for _ab, _v in (("a", 1), ("b", 5)):
                if _ab == "a":
                    continue
                _ms2 = tuple(m_ for m_ in _ms if not (_wr == 8 and m_ == 32))
                if _ms2:
                    GXP.append(("gx-q%d%s" % (16 * _wr, _ab), "qkv_gate", _ms2, (1, _wr, 1, 16, 8, 1, _xs), _v))
    for _ab, _v in (("a", 1), ("b", 5)):
        if _ab == "b":
            GXP.append(("gx-q144%s-s" % _ab, "qkv_gate", (32,), (1, 9, 1, 16, 8, 1, 1), _v))
        GXP.append(("gx-p128%s" % _ab, "pooler", (16,), (1, 8, 2, 16, 4, 1, 2), _v))
        GXP.append(("gx-p128%s-s" % _ab, "pooler", (32,), (1, 8, 2, 16, 4, 1, 1), _v))
        GXP.append(("gx-u128%s" % _ab, "unpooler", (16,), (1, 8, 2, 16, 2, 1, 2), _v))
        GXP.append(("gx-u128%s-s" % _ab, "unpooler", (32,), (1, 8, 2, 16, 2, 1, 1), _v))
    GXP.append(("gx-o64-s", "o", (16, 32), (1, 4, 4, 4, 1, 1, 1), 1))
    GXP.append(("gx-r12g8", "router", (1, 8, 16, 32), (3, 1, 8, 2, 8, 1, 0), 5))
    if not GX11:
        GXP = []


    def _bv(v, ns, cu=1, fence=False):
        return v | 8 | ({2: 0, 4: 1, 8: 2, 16: 3}[ns] << 4) | ({1: 0, 2: 1, 4: 2}[cu] << 6) | (0 if fence else 256)


    GB = True
    for _nm, _ly, _ms, _g, _v, _vars in (
            ("q", "qkv_gate", (1, 8), (1, 1, 8, 16, 1, 1, 0), 1, ((4, 1), (8, 1))),
            ("q144b", "qkv_gate", (1, 8), (1, 9, 1, 16, 8, 1, 0), 5, ((8, 1),)),
            ("q", "qkv_gate", (16,), (1, 1, 4, 8, 4, 1, 2, 1), 1, ((4, 1), (8, 1))),
            ("q144b", "qkv_gate", (16, 32), (1, 9, 1, 16, 8, 1, 2), 5, ((8, 1),)),
            ("q", "qkv_gate", (32,), (1, 1, 4, 16, 2, 1, 2, 1), 1, ((4, 1), (8, 1))),
            ("o", "o", (1, 8), (1, 1, 8, 2, 1, 1, 0), 0, ((4, 1),)),
            ("o", "o", (16,), (1, 1, 8, 2, 1, 1, 0), 1, ((4, 1),)),
            ("o64-s", "o", (16, 32), (1, 4, 4, 4, 1, 1, 1), 1, ((4, 1),)),
            ("p", "pooler", (1, 8), (1, 2, 16, 8, 1, 1, 0), 3, ((4, 1),)),
            ("p16w", "pooler", (1, 8, 16), (1, 2, 8, 16, 1, 1, 0), 1, ((4, 1), (8, 1), (4, 2))),
            ("p", "pooler", (32,), (1, 1, 8, 16, 1, 1, 2), 1, ((4, 1), (8, 1))),
            ("p128b", "pooler", (16, 32), (1, 8, 2, 16, 4, 1, 2), 5, ((4, 1),)),
            ("u", "unpooler", (1, 8), (1, 1, 16, 4, 1, 1, 0), 1, ((4, 1),)),
            ("u", "unpooler", (16,), (1, 1, 4, 16, 1, 1, 2), 1, ((4, 1), (8, 1))),
            ("u", "unpooler", (32,), (1, 1, 4, 16, 1, 1, 2), 0, ((4, 1), (8, 1))),
            ("u128b", "unpooler", (16, 32), (1, 8, 2, 16, 2, 1, 2), 5, ((4, 1),)),
            ("r", "router", (1, 8, 16), (2, 1, 8, 2, 8, 1, 0), 0, ((4, 1),)),
            ("r", "router", (32,), (2, 1, 4, 2, 16, 1, 0), 0, ((4, 1),)),
            ("r12g8", "router", (1, 8, 16, 32), (3, 1, 8, 2, 8, 1, 0), 5, ((4, 1),))):
        for _ns, _cu in _vars if GB else ():
            if True:
                continue
            GXP.append(("gb-%s-n%d%s" % (_nm, _ns, "c%d" % _cu if _cu > 1 else ""), _ly, _ms, _g, _bv(_v, _ns, _cu)))


    GE = False  # codec E closed; r6 + router m=128/256 paths only
    _ER, _EK2, _EK4 = 1 << 20, 1 << 22, 2 << 22
    _GEG = {("qkv_gate", 1): (4, 1, 8, 16, 1, 1, 0), ("qkv_gate", 8): (4, 1, 8, 16, 1, 1, 0), ("qkv_gate", 16): (4, 1, 8, 16, 1, 1, 0),
            ("pooler", 1): (4, 2, 8, 16, 1, 1, 0), ("pooler", 8): (4, 2, 8, 16, 1, 1, 0), ("pooler", 16): (4, 2, 8, 16, 1, 1, 0),
            ("unpooler", 1): (4, 1, 4, 16, 1, 1, 0), ("unpooler", 8): (4, 1, 4, 16, 1, 1, 0), ("unpooler", 16): (4, 1, 4, 16, 1, 1, 2)}
    _GEV = (("", 0), ("r", _ER), ("r-k2", _ER | _EK2), ("r-k4", _ER | _EK4))
    _GEN = {"qkv_gate": "q", "pooler": "p512", "unpooler": "u"}
    # qkv (256 threads) carries the __launch_bounds__ minBlocks 2 override (V bits 12-14): r6's 144 CTAs need 2 per SM to stay in
    # one wave on 132 SMs, which caps registers at 128. qkv and pooler m=16 (MT 2) K2 / K4 spill at 128 and are not built.
    GEP = [("ge-%s%s" % (_GEN[ly] if not (ly == "unpooler" and m <= 8) else "u16", sfx), ly, (m,), g, 1 | b | (2 << 12 if ly == "qkv_gate" else 0))
           for (ly, m), g in _GEG.items() for sfx, b in _GEV if not (ly != "unpooler" and m == 16 and b & (3 << 22))] if GE else []
    _EL = 1 << 24
    GEPL = [("ge-l%s" % nm, ly, (m,), g, 1 | 2 | ((1024 // nt) << 12) | _EL)
            for nm, ly, g, nt in (("p1k", "pooler", (4, 2, 16, 8, 1, 1, 0, 1), 1024), ("p16", "pooler", (4, 1, 16, 8, 1, 1, 0, 1), 512),
                                  ("q16", "qkv_gate", (4, 1, 16, 8, 1, 1, 0, 1), 512), ("q8", "qkv_gate", (4, 1, 8, 16, 1, 1, 0, 1), 256),
                                  ("u8", "unpooler", (4, 1, 8, 8, 1, 1, 0, 1), 256))
            for m in (1, 8)] if GE else []
    GEP += GEPL
    GXP += GEP

    # split-K without a cluster, reduced by red.add into an output zeroed by a memset node in
    # the same graph (gv V bit 26), every W chunk of a warp requested at launch (ring depth ST = KU, 8th geometry field).
    # Router: plain BF16 hi + lo (FMT 2), as r6 at m <= 32 (r6: cluster 8 / 16 DSMEM, 4.35-6.29 benchmark-us). Unpooler (rows
    # 0..1023, K 4096) and o at m <= 8: r6's 12-bit pack (FMT 1, dec1).
    _RD = 1 << 26
    _ZF, _PZ, _XZ = 1 << 27, 1 << 28, 1 << 29
    GRD = []
    for _nm, _g, _ms, _vs in (("rk-s8", (2, 1, 8, 2, 8, 1, 0, 2), (8, 16, 32), "z"),           # 16 x 8 = 128 CTAs x 256 thr
                              ("rk-s16w8", (2, 1, 8, 1, 16, 1, 0, 1), (8, 16, 32), "z"),      # 256 CTAs x 256 thr, one chunk per warp
                              ("rk-s16w8", (2, 1, 8, 1, 16, 1, 0, 1), (8, 16), "x"),          # the x variant at m=32 is 272 CTAs > 264 resident slots
                              ("rk-s32", (2, 1, 4, 1, 32, 1, 0, 1), (8, 16), "x"),          # 512 CTAs x 128 thr
                              ("rk-s16w16", (2, 1, 16, 1, 8, 1, 0, 1), (8,), "z")):
        for _c in _vs:
            GRD.append((_nm + _c, "router", _ms, _g, {"z": _ZF, "x": _XZ, "p": _PZ}[_c]))  # P1 self-zeroing variants
    for _nm, _ly, _g, _vs in (("uk-s4", "unpooler", (1, 1, 8, 2, 4, 1, 0, 2), "zx"),    # n=1024: 64 x 4 = 256 CTAs x 256 thr
                              ("uk-s4w4", "unpooler", (1, 1, 4, 4, 4, 1, 0, 4), "x")):    # 256 CTAs x 128 thr
        for _c in _vs:
            GRD.append((_nm + _c, _ly, (1, 8), _g, 1 | {"z": _ZF, "x": _XZ}[_c]))
    GXP += GRD

    GXP += [("q144k16", "qkv_gate", (1, 8), (1, 1, 16, 8, 1, 1, 2), 1 | 2 << 12)]


    GL_P = (1, 2, 16, 8, 1, 1, 0)
    GL_P4 = (1, 1, 8, 16, 1, 1, 2)
    GL_Q = (1, 1, 8, 16, 1, 1, 0)
    GL_Q4 = (1, 1, 4, 16, 2, 1, 2, 1)
    GLAD = [("pooler", 1, GL_P, 3 | 1 << 9, "L0"), ("pooler", 1, GL_P, 3 | 4 << 9, "L3"),
            ("qkv_gate", 1, GL_Q, 1 | 1 << 9, "L0"), ("qkv_gate", 1, GL_Q, 1 | 4 << 9, "L3")]
    GLAD += [(ly, 4, g, 1 | k << 9, "L%d" % (k - 1)) for ly, g in (("pooler", GL_P4), ("qkv_gate", GL_Q4)) for k in (1, 2, 3, 4)]
    GLAD += [("pooler", 4, GL_P4, 1 | 4 << 12, "cap64"), ("pooler", 4, GL_P4, 1 | 1 << 12, "cap255"),
             ("qkv_gate", 4, GL_Q4, 1 | 1 << 12, "cap255"),
             ("pooler", 4, GL_P4, 1 | 8 | 2 << 4 | 256 | 1 << 9, "L0-bulk8")]
    GLAD = []

    # x cache-policy experiment on r6's own m=16/32 geometries (benchmark picks). V bits 15-16 = x load policy XP
    # (0 r6: ld.global.nc, L1-allocating; 1: ld.global.cg, L1 bypass = r6's L1 reuse removed; 2: ld.global.nc.L1::evict_last);
    # V bit 17 = PreferredSharedMemoryCarveout 0 (largest L1 the kernel's smem allows). W loads stay .nc.L1::no_allocate.
    GCX = (("-xcg", 1 << 15), ("-xel", 2 << 15), ("-L1", 1 << 17), ("-xelL1", 2 << 15 | 1 << 17))
    for _nm, _ly, _m, _g, _v in (("p", "pooler", 16, (1, 2, 8, 16, 1, 1, 0), 1), ("p", "pooler", 32, (1, 1, 8, 16, 1, 1, 2), 1),
                                 ("u", "unpooler", 16, (1, 1, 4, 16, 1, 1, 2), 1), ("u", "unpooler", 32, (1, 1, 4, 16, 1, 1, 2), 0),
                                 ("q", "qkv_gate", 16, (1, 1, 4, 8, 4, 1, 2, 1), 1), ("q", "qkv_gate", 32, (1, 1, 4, 16, 2, 1, 2, 1), 1)):
        for _sfx, _b in ():
            GXP.append(("gc-%s%s" % (_nm, _sfx), _ly, (_m,), _g, _v | _b))


    def _gxinst(geo7, v, bm):
        fmt, wr, wk, ku, s, rt, xs = geo7[:7]
        return (fmt, GVB.get(bm, 16), geo7[7] if len(geo7) > 7 else _gvst((wr, wk, ku, s, rt, xs)), wr, wk, ku, s, rt, xs, v)


    GVI = list(dict.fromkeys(_gvinst() + [_gxinst(g, v, bm) for _, _, ms, g, v in GXP for bm in ms]
                            + [_gxinst(g, v, {1: 1, 2: 16, 4: 32}[mt]) for _, mt, g, v, _ in GLAD]))
    _gvmod = None


    def _gvext():
        nonlocal _gvmod
        if _gvmod is None:
            cases = "\n".join("    case %d: return gvl<%s>(p, st, occ);" % (i, ",".join(map(str, c))) for i, c in enumerate(GVI))
            _gvmod = load_inline("gvp45_predxor_v2_%x" % (hash(tuple(GVI)) & 0xffffffff), cpp_sources="int64_t gvrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvocc(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvregs(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t rrrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor); int64_t rsrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor); int64_t prrun(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor);",
                                 cuda_sources=GVCU.replace("@@CASES@@", cases), functions=["gvrun", "gvocc", "gvregs", "rrrun", "rsrun", "prrun"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
        return _gvmod


    # code-word bit of each value's 3-bit offset and of its sign, per decode variant (see dec0 / dec1)
    _OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
    _SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))


    def _c12(fr, dv=1):
        """bf16 fragments [T, KC, 4, 32, 8] -> (uint8 [T, KC, 3, 32, 16], table, escapes [E, 6] (tile, ch, j, lane, v, xor)) or None"""
        T, KC = fr.shape[:2]
        dev = fr.device
        b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 0xffff
        h7, sg = (b >> 8) & 127, b >> 15
        win = torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1)
        B = int(win.argmax())
        d = h7 - B
        inw = (d >= 0) & (d < 8)
        d = torch.where(inw, d, 0).long()
        C = ((d << torch.tensor(_OS[dv], device=dev)) | (sg.long() << torch.tensor(_SS[dv], device=dev))).sum(-1)
        C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
        lo = (b & 255).to(torch.uint8)
        p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
        data = torch.stack((p0, p1, p2), 2).contiguous()
        ei = (~inw).nonzero()
        esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
        tab = B * 0x01010101 if dv else sum((B + i) << (8 * i) for i in range(8))
        if not torch.equal(_unc12(data, tab, esc, dv), b):
            return None
        return data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc


    def _unc12(data, tab, esc, dv):
        """independent decoder (mirrors the kernel's byte permutes) for the exactness proof"""
        T, KC = data.shape[:2]
        dev = data.device
        w = data.view(torch.int32).view(T, KC, 3, 32, 4).long() & 0xffffffff
        L = torch.stack((w[:, :, 0, :, 0], w[:, :, 0, :, 1], w[:, :, 0, :, 2], w[:, :, 0, :, 3], w[:, :, 1, :, 0], w[:, :, 1, :, 1], w[:, :, 1, :, 2], w[:, :, 1, :, 3]), 2).view(T, KC, 4, 2, 32)
        C = w[:, :, 2].permute(0, 1, 3, 2)
        off = (C[..., None] >> torch.tensor(_OS[dv], device=dev)) & 7
        sgn = (C[..., None] >> torch.tensor(_SS[dv], device=dev)) & 1
        tb = torch.tensor([(tab >> (8 * i)) & 255 for i in range(8)], device=dev)
        hi = ((tab & 255) + off if dv else tb[off]) | (sgn << 7)
        lob = (L.permute(0, 1, 2, 4, 3)[..., None] >> (8 * torch.arange(4, device=dev))) & 255
        val = (hi << 8) | lob.reshape(T, KC, 4, 32, 8)
        if len(esc):
            t, c, j, l, vv, x = esc.t()
            val[t, c, j, l, vv] ^= x << 8
        return val.int()


    def _gvesc(esc, T, geo):
        """per-warp escape lists for one launch geometry (32 entries per warp, sorted by (unit, tile)); None if a list overflows.
    esc: [E, 7] (tile, chunk, step, lane, value, xor, matrix)"""
        wr, wk, ku, s, rt, xs = geo
        gx, nw = T // (wr * rt), wr * wk
        out = torch.full((gx * s * nw * 32,), -1, dtype=torch.int64, device=esc.device)
        e = esc[esc[:, 0] < T]
        if len(e):
            t, c, j, l, v, x, h = e.t()
            gw = ((c // (wk * ku)) * gx + t // (wr * rt)) * nw + ((c // ku) % wk) * wr + (t // rt) % wr
            seq = (c % ku) * rt + t % rt
            o = torch.argsort(gw * 4096 + seq)
            gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | (h << 9) | x)[o]
            rank = torch.arange(len(gw), device=gw.device) - torch.searchsorted(gw, gw)
            if int(rank.max()) >= 32:
                return None
            out[gw * 32 + rank] = ent
        return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()


    def _bk(m):
        return 1 if m == 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m == 128 else 256 if m == 256 else 0


    def _tm(fn, flush, it=15):
        """median replay time (ms) of a CUDA graph of fn, each replay after the validated cold-L2 READ flush"""
        g = _capture(fn)
        ts = _replay({"a": g}, it)["a"]
        ts.sort(); return ts[len(ts) // 2]


    def _tmp(fn, base, flush, it=15):
        """fn timed ALTERNATING with `base` (order-swapped pairs, median paired ratio, fixed cost subtracted);
    returns fn's time in base units (ms), i.e. base_median / ratio -- the benchmark's own form."""
        ratio, bmed, cmed, ov = _paired(fn, base, max(2, it // 2))
        _LAST[0] = (ratio, bmed, cmed, ov)
        return bmed / ratio


    def _gvok(st, plan, exact, w, mr):
        """exact on every one-hot column, close on random rows"""
        ext, n, K = _gvext(), exact.shape[0], exact.shape[1]
        mm = (8 if plan[0] in (1002, 1003) else 1) if plan[0] >= 1000 else 8 * GVI[plan[0]][1]
        dt = exact.dtype
        eye = torch.eye(K, dtype=torch.bfloat16, device=w.device).view(K // mm, mm, K)
        got = torch.empty((K // mm, mm, n), dtype=dt, device=w.device)
        for i in range(K // mm):
            if _gvcall(st, plan, eye[i], got[i]):
                return False
        if not torch.equal(got.view(K, n), exact.t()):
            return False
        x = torch.randn((max(mm, mr), K), device=w.device).to(torch.bfloat16)
        out = torch.empty((max(mm, mr), n), dtype=dt, device=w.device)
        _gvcall(st, plan, x, out)
        ref = x.double() @ w.double().T
        return bool(torch.isfinite(out).all()) and float((out.double() - ref).norm() / ref.norm()) < (2e-5 if dt == torch.float32 else 4e-3)


    RRB = {1: (6, 7)}  # proven rr6 first (benchmark m1 2.56); rr7 kept (2.59, inside 3%); rr0 (2.94, +15%) and rr1 (3.01, +18%) dropped  # P1: rr2/rr3 (no cross-CTA reduction) re-measured at m=8 on the H100; fp32 router kernels per bucket (m8 rr: 7.7-14.9 us on the H200, v12 4.7)


    def _name(i):
        return "rr%d" % (i - 1000) if i >= 1000 else ".".join(map(str, GVI[i]))


    def _gvcall(st, p, x, out):
        if 0 <= p[0] < 1000 and (GVI[p[0]][9] >> 26) & 1:
            out.zero_()
        if 0 <= p[0] < 1000 and (GVI[p[0]][9] >> 27) & 5:  # per-instance ticket / zero-flag counters (self-zeroing)
            c = st.setdefault("p1cnt", {}).get(p[0])
            if c is None:
                c = st["p1cnt"][p[0]] = torch.zeros((2048,), dtype=torch.int32, device=x.device)
            return _gvext().gvrun(p[0], p[1], p[5], x, out, p[2], st["gvws"], c, p[3], p[6])
        if p[0] == -7:
            return _rscall(st, p, x, out)
        if p[0] >= 1000:
            return _gvext().rrrun(p[0] - 1000, p[1], x, out)
        return _gvext().gvrun(p[0], p[1], p[5], x, out, p[2], st["gvws"], st["gvcnt"], p[3], p[6])


    def _gvprep(st, flush):
        dev = flush.device
        st["gvws"] = torch.zeros((1 << 24,), dtype=torch.float32, device=dev)
        st["gvcnt"] = torch.zeros((1 << 16,), dtype=torch.int32, device=dev)
        noesc = torch.full((1 << 16,), -1, dtype=torch.int32, device=dev)
        st["gvlog"] = []
        ok = {}
        z = torch.zeros(1, device=dev)
        off = _tm(lambda: z.zero_(), flush, 31)
        for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
            hl = layer == "router"
            if not _NCR(layer, [bm_ for bm_, _mt in list(GVB.items()) + (GVRB if hl else [])]):
                continue
            frs = list(st[layer]) if hl else [st[layer][0]]
            css = [[_c12(f, dv) for f in frs] if GVC12 and not hl else [None] for dv in (0, 1)]
            c12 = all(c is not None for cs in css for c in cs)
            if c12:
                esc = torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(css[1])])
                st.setdefault("gvc12", {})[layer] = (css, esc)  # reused by the second bake
            Tf, KC = frs[0].shape[:2]
            w = st[layer + "_raw"]
            for bm, mt in list(GVB.items()) + (GVRB if hl else []):  # router m128/256 as token blocks: 1.6x slower than v12 on the 3090
                if not _NC(layer, bm):
                    continue
                key = "unpooler_s" if layer == "unpooler" and bm <= 8 else layer
                n = 1024 if key == "unpooler_s" else Tf * 16
                T, wn = n // 16, w[:n]
                exact = (wn.to(torch.bfloat16).float() + (wn - wn.to(torch.bfloat16).float()).to(torch.bfloat16).float()) if hl else wn
                cands = []
                for k, (geo, vs) in enumerate(GVC.get((key, mt), [])):
                    if not c12 and k and not hl:
                        break
                    if geo[1] * geo[2] * geo[3] != KC or T % (geo[0] * geo[4]):
                        continue
                    e = _gvesc(esc, T, geo[:6]) if c12 else noesc
                    for v in vs if c12 else (0,):
                        fmt = 1 if c12 else 2 if hl else 0
                        i = GVI.index((fmt, mt, _gvst(geo)) + geo[:6] + (v,))
                        if e is None:
                            continue
                        cs = css[v & 1]
                        src = [c[0] for c in cs] if fmt & 1 else frs
                        plan = (i, src[0], e, cs[0][1] if fmt & 1 else 0, n, src[-1], cs[-1][1] if fmt & 1 else 0)
                        if (i, n, bm > 8 * mt) not in ok:
                            ok[(i, n, bm > 8 * mt)] = _gvok(st, plan, exact, wn, bm)
                        if ok[(i, n, bm > 8 * mt)]:
                            cands.append(plan)
                for i in RRB.get(bm, ()) if hl else ():
                    plan = (1000 + i, wn.contiguous(), noesc, 0, n, wn, 0)
                    if (plan[0], n, 0) not in ok:
                        ok[(plan[0], n, 0)] = _gvok(st, plan, wn, wn, bm)
                    if ok[(plan[0], n, 0)]:
                        cands.append(plan)
                x = torch.randn((bm, KC * 64), device=dev).to(torch.bfloat16)
                dt = torch.float32 if hl else torch.bfloat16
                out = torch.empty((bm, n), dtype=dt, device=dev)
                v12 = lambda: _v12(x, layer, n, st)
                t12 = _tm(v12, flush)  # absolute (ms); the floor `off` is subtracted for display and comparison
                res = [(layer, bm, "v12", t12 - off)]
                # one reference per group (see _pickrule): the proven pick when the box does not read it
                # slower than v12 (window = half GVMARGIN), else v12; every candidate is timed ALTERNATING with that same
                # reference (order-swapped pairs, median paired ratio, floor `off` subtracted) and the pick is the best
                # ratio if it beats GVMARGIN, else the reference.
                pv = PROVEN_GV.get((layer, bm))
                prov = next((p for p in cands if _name(p[0]) == pv), None)
                refn, refp, refmed, reffn = "v12", None, t12, v12
                cands2, done = [], set()
                if prov is not None:
                    done.add(id(prov))
                    tp = _tmp(lambda: _gvcall(st, prov, x, out), v12, flush)
                    res.append((layer, bm, _name(prov[0]), tp - off))
                    rn, rm, isp = _refsel(_name(prov[0]), tp, "v12", t12, GWIN)
                    if isp:
                        refn, refp, refmed, reffn = rn, prov, rm, (lambda: _gvcall(st, prov, x, out))
                    else:
                        cands2.append((_name(prov[0]), prov, t12 / tp))
                for p in cands:
                    if id(p) in done:
                        continue
                    us = _tmp(lambda: _gvcall(st, p, x, out), reffn, flush)
                    res.append((layer, bm, _name(p[0]), us - off))
                    cands2.append((_name(p[0]), p, refmed / us))
                bp = _pickrule((refn, refp, refmed), cands2, GVMARGIN)
                best = (bp[2], bp[1])
                st["gvlog"] += res
                print("gv %s m%d v12 %.2f best %s %.2f (ref %s %.2f)" % (layer, bm, (t12 - off) * 1e3, _name(best[1][0]) if best[1] is not None else "v12", (best[0] - off) * 1e3, refn, (refmed - off) * 1e3), flush=True)
                print("  " + " ".join("%s:%.2f" % (r[2], r[3] * 1e3) for r in res), flush=True)
                if best[1] is not None:
                    st["gv"][(layer, bm)] = best[1]
                if P1ALT and hl and bm == 1 and best[1] is not None and best[1][0] >= 1001:  # the rr pick vs r6's rr0, alternated
                    p0 = next((p_ for p_ in cands if p_[0] == 1000), None)
                    if p0 is not None:
                        ta, tb = [], []
                        for _ in range(3):
                            ta.append(_tm(lambda: _gvcall(st, best[1], x, out), flush, 21)); tb.append(_tm(lambda: _gvcall(st, p0, x, out), flush, 21))
                        print("PROBE p1alt router m=1 %s %.2f rr0=%.2f delta=%+.2f" % (_name(best[1][0]), min(ta) * 1e3, min(tb) * 1e3, (min(ta) - min(tb)) * 1e3), flush=True)

    _mod = None
    # host-only (g++): the benchmark's nvcc crashes on these STL containers under -std=c++20
    LTCPP = __cuda__("streamk_ltcpp.cpp")
    _DBG = []
    RTR = True
    GV = True
    GVC12 = True
    GVMARGIN = 0.97  # a variant must beat the PROVEN pick by 3% at m <= 32 (FACTS 14)
    TUNE = True  # prepare(): time torch vs cuBLASLt algorithms per cuBLAS-served shape
    V6 = False  # deep per-CTA ring GEMV: slower than v5 on H200


    def _big():
        pr = torch.cuda.get_device_properties(0)
        sm = getattr(pr, "shared_memory_per_block_optin", None)
        return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


    def _ext():
        nonlocal _mod
        if _mod is None:
            wpc = next((int(d[6:]) for d in _DBG if d.startswith("-DWPC=")), 16 if _big() else 8)
            _mod = load_inline("skinny_v20_%d" % wpc + "".join(_DBG).replace("-", "_").replace("=", ""),
                               cpp_sources=LTCPP + "void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run3(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
                               cuda_sources=CUDA, functions=["run", "run2", "run3", "lt_algos", "lt_mm", "lt_caps", "lt_add", "lt_cfg"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc] + ([] if _big() else ["-DNSG=4"]) + [d for d in _DBG if not d.startswith("-DWPC=")], extra_ldflags=_ltflags(),
                               verbose=False)
        return _mod


    def _ltflags():
        """link the process's cuBLASLt (torch's nvidia-cublas wheel ships only the versioned soname)"""
        import glob, os
        try:
            import nvidia.cublas as nc
            d = os.path.join(list(nc.__path__)[0], "lib")
        except Exception:
            d = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64")
        so = sorted(glob.glob(os.path.join(d, "libcublasLt.so*")))
        return ["-L" + d, "-l:" + os.path.basename(so[0]) if so else "-lcublasLt", "-Wl,-rpath," + d]


    def _tune(st, flush):
        """pick, per cuBLAS-served shape, the fastest of torch's default and every cuBLASLt heuristic algorithm (timed here,
    on this GPU, as the benchmark times: CUDA graph replays from a cold L2); m 128/256 get the pinned-config search below"""
        ext = _ext()
        tm = lambda fn, it=12: _tm(fn, flush, it)
        dev = flush.device
        jobs = []
        for layer, (N, K) in (("qkv_gate", (2304, 8192)), ("o", (8192, 1024)), ("pooler", (4096, 8192)), ("unpooler", (8192, 4096))):
            for m in (1, 8, 16, 32, 128, 256):
                if _cublas(layer, m) and _NC(layer, m):
                    jobs.append((layer, m, 1024 if layer == "unpooler" and m <= 8 else N, K, st[layer + "_raw"], st.get(layer + "_t"), torch.bfloat16))
        for m in (128, 256):
            if _NC("router", m):
                jobs.append(("router", m, 512, 8192, st["router_hl"], st["router_hl_t"], torch.float32))
        t0 = time.time()
        for layer, m, n, K, W, Wt, dt in jobs:
            x = torch.randn((m, K), device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), dtype=dt, device=dev)
            ref = torch.mm(x.float(), W[:n].float().T)
            f32 = int(dt == torch.float32)
            torch_fn = (lambda: torch.mm(x, W[:n].T, out_dtype=dt)) if f32 else (lambda: torch.mm(x, W[:n].T))
            ok = lambda c: ext.lt_mm(x, W, Wt if Wt is not None else W, out, n, c) == 0 and _close(out, ref)
            run = lambda c: ext.lt_mm(x, W, Wt if Wt is not None else W, out, n, c)
            if m < 128 or not LTSEARCH or Wt is None:
                best = (tm(torch_fn), -1)
                for i in range(ext.lt_algos(m, n, K, f32, 0)):
                    if ok(i):
                        best = min(best, (tm(lambda: run(i)), i))
                if best[1] >= 0:
                    st["lt"][(layer, m)] = best[1]
                continue
            st["lt"][(layer, m)] = _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0)


    LTSEARCH = True
    LTBUDGET = 8.0 if _PRUNE else 22.0  # seconds for the whole m128/256 search (10 shapes)
    LTW = {"qkv_gate": 1.6, "o": 0.3, "pooler": 1.0, "unpooler": 0.7, "router": 1.6}
    _LTSEED = [(2, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 18, 1, 0, 0, 3, 35, 0, 5]), (3, [66, 20, 1, 0, 0, 4, 35, 0, 2]),
               (0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (0, [66, 124, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 4, 35, 0, 2]),
               (3, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 1, 35, 0, 2]),
               (2, [66, 15, 1, 0, 0, 0, 35, 0, 4]), (2, [66, 18, 1, 0, 0, 4, 35, 0, 2]), (3, [66, 18, 1, 0, 0, 2, 35, 0, 5]),
               (3, [66, 18, 1, 0, 0, 5, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 2, 35, 0, 5]), (2, [66, 18, 1, 0, 0, 0, 35, 0, 2]),
               (2, [66, 18, 1, 0, 0, 6, 35, 0, 5])]


    def _close(out, ref):
        torch.cuda.synchronize()
        return bool(torch.isfinite(out).all()) and float((out.float() - ref).norm() / ref.norm()) < 1e-2


    def _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0):
        """heuristic algorithms of all four layouts + seeds + an algo-66 (sm90 nvjet) sweep over tile / stages / cluster, then
    split-K / custom option / cluster / stages refinement of the leaders; all checked numerically and timed (benchmark timer)"""
        if not _LTDONE and not _LT0:
            _LT0.append(time.time())
        t0 = _LT0[0]
        order = [(l, mm) for l in ("qkv_gate", "o", "pooler", "unpooler", "router") for mm in (128, 256) if _NC(l, mm)]
        wsum = sum(LTW[l] for l, _ in order)
        dl = t0 + LTBUDGET * sum(LTW[l] for l, mm in order[:order.index((layer, m)) + 1]) / wsum  # cumulative, weighted
        njob = wsum / LTW[layer]
        tried, res = set(), []
        def tryc(code):
            if code < 0 or code in tried:
                return
            tried.add(code)
            if ok(code):
                res.append((_tm(lambda: run(code), flush, 5), code))
        def addc(lay, cfg):
            c = ext.lt_add(m, n, K, f32, lay, cfg)
            if c >= 0:
                cfg2 = tuple(ext.lt_cfg(m, n, K, f32, c))
                if (lay, cfg2) in seen:
                    return
                seen.add((lay, cfg2))
                tryc(c)
        seen = set()
        t_torch = _tm(torch_fn, flush, 9)
        for lay in range(4):
            for i in range(min(ext.lt_algos(m, n, K, f32, lay), 8)):
                seen.add((lay, tuple(ext.lt_cfg(m, n, K, f32, lay * 100000 + i))))
                tryc(lay * 100000 + i)
        for lay, cfg in _LTSEED:
            addc(lay, cfg)
        caps = ext.lt_caps(f32, 66)
        if caps:
            nt = caps[0]; tiles = caps[1:1 + nt]; ns = caps[1 + nt]; stages = caps[2 + nt:2 + nt + ns]
            cmax, ksup, swz = caps[2 + nt + ns], caps[3 + nt + ns], caps[4 + nt + ns]
            stg = [35] if 35 in stages else stages[:1] or [0]
            for lay in (2, 0, 3, 1):
                for tile in tiles:
                    if time.time() > dl - 0.4 * LTBUDGET / njob:
                        break
                    for cl in (2, 5):
                        addc(lay, [66, tile, 1, 0, 0, 0, stg[0], 0, cl])
            res.sort()
            for _, code in res[:4]:
                if time.time() > dl:
                    break
                lay, cfg = code // 100000, list(ext.lt_cfg(m, n, K, f32, code))
                if cfg[0] != 66:
                    continue
                for cu in range(min(cmax, 7) + 1):
                    for cl in (0, 2, 3, 4, 5, 6):
                        if time.time() > dl:
                            break
                        addc(lay, cfg[:5] + [cu] + cfg[6:8] + [cl])
                if ksup:
                    for sk in (2, 3, 4):
                        addc(lay, cfg[:2] + [sk, 2] + cfg[4:])
                for sg in stages[::max(1, len(stages) // 6)]:
                    if time.time() > dl:
                        break
                    addc(lay, cfg[:6] + [sg] + cfg[7:])
        res.sort()
        best = (t_torch, -1)
        for _, code in res[:3]:  # re-time the leaders against torch's default
            a = _tm(lambda: run(code), flush, 15)
            t_torch = min(t_torch, _tm(torch_fn, flush, 15))
            if a < best[0]:
                best = (a, code)
        if best[1] >= 0 and best[0] > t_torch * 0.99:
            best = (t_torch, -1)
        _LTDONE.append((layer, m))
        cfg = "torch" if best[1] < 0 else "%d:%s" % (best[1] // 100000, ".".join(map(str, ext.lt_cfg(m, n, K, f32, best[1]))))
        print("lt %s m%d heur %.2f best %s %.2f (%d tried, %.1fs)" % (layer, m, t_torch * 1e3, cfg, best[0] * 1e3, len(tried), time.time() - t0), flush=True)
        return best[1] if best[1] >= 0 else None


    _LTDONE, _LT0 = [], []


    def _cublas(layer, m):
        return layer != "router" and (m > 1 or layer == "o") and not (layer == "unpooler" and m == 1)


    def _pack(w):
        """bf16 [N, K] -> A fragments [N/16, K/64, 32 lanes, 4 steps, 8] (K permuted so each lane's 16 k are contiguous)."""
        N, K = w.shape
        W = w.view(N // 16, 16, K // 64, 4, 4, 4)  # mt, i, ch, t, j, e
        lo, hi = W[:, :8], W[:, 8:]
        fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
        return fr.permute(0, 2, 4, 1, 3, 5).contiguous()  # mt, ch, j, g, t, 8: per step j the 32 lanes are contiguous


    def prepare(weights):
        st = {}
        if weights["qkv_gate"].device.type != "cuda":  # CPU (proj_rsi cpu-check) runs the plain torch math
            return {"cpu": True, "raw": dict(weights)}
        rqt = _rqstart() if RQ else None  # rs128: compile overlapped with r6's phase (CPU only)
        rqt2 = _rqstart(RQPROD2, _rqmod2, _rqthr2, "rq32") if RQ and RQS else None  # rs32: m = 16 / 32 instances
        _thr = {}
        if _PRUNE:  # compile the gv and flush extensions beside _ext() and the rq builds (nvcc subprocesses; the benchmark's prepare is compile-bound)
            import threading
            for nm_, fn_ in (("gv", _gvext), ("flush", _flushext)):
                def _bg(fn_=fn_):
                    try:
                        fn_()
                    except Exception:  # the main thread retries the build and raises as before
                        pass
                _thr[nm_] = threading.Thread(target=_bg, daemon=True)
                _thr[nm_].start()
        for k, w in weights.items():
            if k == "router":
                hi = w.to(torch.bfloat16)
                lo = (w - hi.float()).to(torch.bfloat16)
                st[k] = (_pack(hi), _pack(lo))
                st["router_hl_t"] = torch.cat((hi, lo)).t().contiguous()  # [K, 2N]: one cuBLAS mm for m > 32
                st["router_hl"] = torch.cat((hi, lo)).contiguous()  # [2N, K] for cuBLASLt
            else:
                st[k] = (_pack(w), None)
                st[k + "_t"] = w.t().contiguous()  # [K, N] for the packed cuBLASLt layouts
            st[k + "_raw"] = w
        dev = weights["qkv_gate"].device
        st["ws"] = torch.zeros((256 * 8192,), dtype=torch.float32, device=dev)
        st["cnt"] = torch.zeros((8 * 512,), dtype=torch.int32, device=dev)
        st["none"] = torch.empty(0, dtype=torch.bfloat16, device=dev)
        _ext()
        # the benchmark-faithful bake measurement -- build + validate the cold-L2 read flush and measure the graph
        # fixed cost once, before any bake (FACTS 14: the old torch.sum flush left x and small weights L2-warm).
        if "flush" in _thr:
            _thr["flush"].join()
        _flushext()
        _flush_validate()
        _fixed_cost()
        print("PREP flush+fixed done at %.0fs" % (time.time() - _T0), flush=True)
        st["lt"] = {}
        if TUNE:
            _tune(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))
        st["gv"] = {}
        if "gv" in _thr:
            _thr["gv"].join()
        if GV:
            _gvprep(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))
        print("PREP r6 done in %.0fs" % (time.time() - _T0), flush=True)
        st["sel"] = {}
        st["rq"] = {}  # rs128 before the gx / router phases: its gain is the larger
        if RQ and rqt is not None:
            try:
                _rqbake(st, rqt)
            except Exception as error:  # the rs128 bake can only add m = 128 / 256 paths; r6's choices stand on any failure
                st["rq"] = {}
                print("PREP rq bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
            finally:
                torch.cuda.synchronize()
        print("PREP phase rq done at %.0fs" % (time.time() - _T0), flush=True)  # phase timestamps
        if GV and GX:
            try:
                _gxbake(st)
            except Exception as error:  # the second bake can only add paths; r6's choices stand on any failure
                st["sel"] = {}
                print("PREP gx bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
        print("PREP phase gx done at %.0fs" % (time.time() - _T0), flush=True)  # phase timestamps
        if RQ and RQS and rqt2 is not None:
            try:
                _rqsbake(st, rqt2)
            except Exception as error:  # can only add (layer, 16 / 32) rq paths; the incumbent stands on any failure
                for k_ in [k_ for k_ in st["rq"] if k_[1] in (16, 32)]:
                    del st["rq"][k_]
                print("PREP rq32 bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
            finally:
                torch.cuda.synchronize()
        print("PREP phase rq32 done at %.0fs" % (time.time() - _T0), flush=True)  # phase timestamps
        st.pop("rq_pk", None)
        if PR27 and GV and GX and time.time() - _T0 < PREPARE_BUDGET_S - 30:
            try:
                _pr27(st)
            except Exception as error:
                print("PROBE rd8 failed: %s" % str(error).splitlines()[-1][:200], flush=True)
        if GV and GX and _NCR("router", (128, 256)) and time.time() - _T0 < PREPARE_BUDGET_S - 20:
            try:
                _rsbake(st)
            except Exception as error:  # can only add router paths; r6's choices stand on any failure
                for b_ in (128, 256):
                    st["sel"].pop(("router", b_), None)
                print("PREP rs bake failed: %s" % str(error).splitlines()[-1][:200], flush=True)
            finally:  # never leave a capture open or a side stream busy for the benchmark's own captures
                if torch.cuda.is_current_stream_capturing():
                    print("PROBE l2scratch capture_left_open", flush=True)
                torch.cuda.synchronize()
        print("PREP phase router done at %.0fs" % (time.time() - _T0), flush=True)  # phase timestamps
        st.pop("gvc12", None)
        st.pop("gepk", None)  # unpicked codec E packs (a picked plan keeps its own pack alive)
        print("PREP done in %.0fs" % (time.time() - _T0), flush=True)
        return st


    PREPARE_BUDGET_S = 240.0
    MARGIN = 0.99
    MARGIN_RQ128 = 0.98
    # m=128 layers whose pick is forced to the per-thread escape cursor variant (see the bake).
    RQPICK128 = {("pooler", 128): "d2s5hzwP", ("qkv_gate", 128): "d2s5hzwP", ("unpooler", 128): "d2s5hzwP"}
    RQPICK32 = {("pooler", 32): "t32s6hzw-k4", ("qkv_gate", 32): "t32s6hzw-k7", ("unpooler", 32): "t32s6hzw-k2"}
    RQPICK16 = {("qkv_gate", 16): "t16s5hzw-k7"}
    RQPICK32B = {("o", 32): "h32s5hze2-k2"}
    MARGIN_RQ32 = 0.97  # ... and the proven m<=32 pick by 3%
    MARGIN_RS = 0.98  # ... and the proven router m=128/256 pick by 2%
    MARGIN_GX = 0.97  # ... and the proven gx (unpooler m1/m8) pick by 3%
    TOL = {"bf16": 4e-3, "f32": 2e-5}
    _EYE = {}


    FLUSH_MB = 128  # >= 2x L2 (50 MB on the H100/H200)
    FLUSH_BLOCKS, FLUSH_THREADS = 1056, 256
    HBM_RATE_TBPS = 3.35  # a cold probe must not beat this
    PROBE_BYTES = 6_300_032
    DISCRIM_BYTES = 16 << 20  # the probe the flush is validated on (6.3 MB is latency-bound and cannot discriminate)
    PAIRS = 8  # timed pairs per (candidate, baseline) comparison

    FLUSH_CUDA = __cuda__("streamk_flush.cu")

    FLUSH_CPP = r"""
#include <torch/extension.h>
void flush_read(torch::Tensor buf, torch::Tensor sink, int64_t blocks, int64_t threads);
void probe_read(torch::Tensor buf, torch::Tensor out, int64_t blocks, int64_t threads);
void nop(torch::Tensor out, int64_t blocks, int64_t threads);
"""

    _flext = [None]


    def _flushext():
        if _flext[0] is None:
            t0 = time.time()
            _flext[0] = load_inline(name="l2_flush", cpp_sources=FLUSH_CPP, cuda_sources=FLUSH_CUDA,
                                    functions=["flush_read", "probe_read", "nop"],
                                    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
            print("BUILD l2_flush %.1fs" % (time.time() - t0), flush=True)
        return _flext[0]


    class _Flusher:
        """A validated cold-L2 flush: a READ of a buffer larger than L2, before every replay."""

        def __init__(self, mb=FLUSH_MB):
            self.ext = _flushext()
            self.buf = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
            self.sink = torch.zeros((), dtype=torch.float32, device="cuda")

        def __call__(self):
            self.ext.flush_read(self.buf, self.sink, FLUSH_BLOCKS, FLUSH_THREADS)


    _FL = [None]  # the flusher
    _OV = [0.0]   # graph fixed cost (us): the benchmark's 2*one_node - two_node
    _LAST = [None]  # (ratio, base_med, cand_med, ov) of the last paired comparison, for the BAKE lines


    def _flush():
        if _FL[0] is None:
            _FL[0] = _Flusher()
        return _FL[0]


    def _ev_median(fn, reps=20, fl=True):
        ts = []
        for _ in range(reps):
            if fl:
                _flush()()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]


    def _flush_validate():
        """Flush check: ld.global probes after the flush must run at <= HBM rate (the 6.3 MB probe is
    latency-bound and cannot discriminate; the pick is made on the 16 MB one)."""
        ext = _flushext()
        _flush()
        out = torch.zeros((), dtype=torch.float32, device="cuda")
        print("FLUSH_VALIDATE probes %.2f MB and %.2f MB; cold must be <= %.2f TB/s (pick on the larger probe)"
              % (PROBE_BYTES / 1e6, DISCRIM_BYTES / 1e6, HBM_RATE_TBPS), flush=True)
        ok_all = True
        for nb in (PROBE_BYTES, DISCRIM_BYTES):
            buf = torch.empty(nb, dtype=torch.uint8, device="cuda")
            buf.view(torch.float32).normal_()
            fn = lambda: ext.probe_read(buf, out, 528, 256)
            t = _ev_median(fn, fl=False)
            print("FLUSH_VALIDATE %-16s %6.2f MB %7.2f us %6.2f TB/s" % ("warm(no flush)", nb / 1e6, t,
                                                                         nb / 1e9 / (t * 1e-6) / 1e3), flush=True)
            t = _ev_median(fn)
            tb = nb / 1e9 / (t * 1e-6) / 1e3
            ok = tb <= HBM_RATE_TBPS
            if nb == DISCRIM_BYTES:
                ok_all = ok
            print("FLUSH_VALIDATE %-16s %6.2f MB %7.2f us %6.2f TB/s  %s"
                  % ("read/%dMB" % FLUSH_MB, nb / 1e6, t, tb, "COLD ok" if ok else "WARM reject"), flush=True)
        print("FLUSH_PICK read/%d %s" % (FLUSH_MB, "ok" if ok_all else "WARNING no cold flush"), flush=True)
        return ok_all


    def _capture(fn):
        """capture a CUDA graph of the zero-arg callable fn (two warm-up calls on a side stream).
    capture_begin/capture_end directly instead of the `torch.cuda.graph` context manager, which calls
    torch.cuda.empty_cache() + _host_emptyCache() on entry -- measured 28 ms per capture on the box (vs 1.2 ms
    here), i.e. ~150 s over the bake's ~1000 captures. The tensors a graph touches are held by the bake's scope."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin()
            fn()
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        return g


    def _replay(graphs, pairs):
        """pairs replays per graph, order swapped every pair, flush before every replay.
    Returns {name: [us]}."""
        names = list(graphs)
        ev = {n: [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(pairs)] for n in names}
        for p in range(pairs):
            for n in (names if p % 2 == 0 else names[::-1]):
                _flush()()
                a, b = ev[n][p]
                a.record(); graphs[n].replay(); b.record()
        torch.cuda.synchronize()
        return {n: [a.elapsed_time(b) * 1e3 for a, b in ev[n]] for n in names}


    def _fixed_cost(pairs=PAIRS):
        """A graph replay's fixed cost, measured the way the benchmark does: 2*one_node - two_node."""
        ext = _flushext()
        out = torch.zeros((), dtype=torch.float32, device="cuda")
        one = _capture(lambda: ext.nop(out, 132, 128))
        two = _capture(lambda: (ext.nop(out, 132, 128), ext.nop(out, 132, 128)))
        t = _replay({"one": one, "two": two}, pairs)
        o, w = statistics.median(t["one"]), statistics.median(t["two"])
        _OV[0] = 2 * o - w
        print("FIXED_COST one_node %.2f us two_node %.2f us overhead %.2f us" % (o, w, _OV[0]), flush=True)
        return _OV[0]


    def _paired(fn, base, pairs=PAIRS):
        """fn and base captured as graphs and replayed alternately; returns (ratio, base_med, cand_med, ov) with
    ratio = median over pairs of sqrt(r_a*r_b), r = (base-fixed)/(cand-fixed)."""
        ga, gb = _capture(fn), _capture(base)
        t = _replay({"a": ga, "b": gb}, pairs)
        ta, tb = t["a"], t["b"]
        ov = _OV[0]
        lim = 0.5 * min(statistics.median(ta), statistics.median(tb))
        if ov > lim:  # never let the subtraction reach the times (a bad fixed-cost read must not blow up a ratio)
            print("PAIR ov %.2f clamped to %.2f (times %.2f/%.2f)" % (ov, lim, statistics.median(ta),
                                                                     statistics.median(tb)), flush=True)
            ov = max(0.0, lim)
        r = [(b - ov) / max(a - ov, 1e-3) for a, b in zip(ta, tb)]  # (base-ov)/(cand-ov), the benchmark's r
        pr = [math.sqrt(r[i] * r[i + 1]) for i in range(0, len(r) - 1, 2)]
        return statistics.median(pr), statistics.median(tb), statistics.median(ta), ov


    def _time_us(fn, x, out, flush, sink, it=15, base=None):
        """base=None -> absolute median replay time (us) after the validated cold-L2 read flush.
    base=fn2 -> the benchmark's paired form: fn timed ALTERNATING with fn2 (order swapped every pair, flush before every
    replay), ranked by the median of sqrt(r_a*r_b) with r = (base-fixed)/(cand-fixed); returns base_median/ratio (us),
    i.e. fn's time expressed in the baseline's units. `flush`/`sink` are kept for call-site compatibility."""
        if base is None:
            g = _capture(lambda: fn(x, out))
            ts = _replay({"a": g}, it)["a"]
            ts.sort()
            return ts[len(ts) // 2]
        ratio, bmed, cmed, ov = _paired(lambda: fn(x, out), lambda: base(x, out), max(2, it // 2))
        _LAST[0] = (ratio, bmed, cmed, ov)
        return bmed / ratio


    RWIN128 = 1.01   # a proven pick is a group's reference only if this GPU does not read it slower than
    RWIN32 = 1.015   # r6 by more than HALF the group's margin (2% -> 1%, 3% -> 1.5%); otherwise r6 is the reference and
    RWIN_RS = 1.01
    RWIN_GX = 1.015  # become the reference -- then no candidate could ever lose to r6 (the qkv_gate m=256 failure).
    GWIN = 1.015     # the gv bake's window (half of GVMARGIN)


    def _refsel(prov_name, prov_med, base_name, base_med, window):
        """The group's reference: the stored pick unless this GPU reads it slower than the base (r6) by more
    than `window`, else the base. Returns (name, med, is_prov)."""
        if prov_name is not None and prov_med is not None and prov_med <= base_med * window:
            return prov_name, prov_med, True
        return base_name, base_med, False


    def _pickrule(ref, cands, margin):
        """The group's pick from ONE reference. ref = (name, plan, med); cands = [(name, plan, ratio)] with
    ratio = (ref-fixed)/(cand-fixed) > 1 meaning the candidate is faster than the reference -- every ratio in a group
    is measured against the same reference. Returns (name, plan, med): the candidate with the best ratio if it beats
    the reference by `margin` (ratio > 1/margin), else the reference itself."""
        best = None
        for nm, plan, r in cands:
            if best is None or r > best[2]:
                best = (nm, plan, r)
        if best is not None and best[2] > 1.0 / margin:
            return best[0], best[1], ref[2] / best[2]
        return ref


    def _validate(fn, w, m, n, f32, gen, exact=None, m_odd=None):
        """template _validate: one-hot rows x8 bit-exact against the weight (for the router: against hi + lo, which is what
    the lossless 12-bit hi/lo pack carries), random rows within TOL of FP64 on the true weight; plus random rows at an
    odd m of the same bucket (kernel_fn serves the whole bucket with this plan)."""
        K = w.shape[1]
        dt = torch.float32 if f32 else torch.bfloat16
        out = torch.empty((m, n), dtype=dt, device=w.device)
        exact = (exact if exact is not None else w)[:n].t().to(dt)  # [K, n]
        rows = max(m, 1)
        if K not in _EYE:
            _EYE[K] = torch.eye(K, dtype=torch.bfloat16, device=w.device) * 8.0
        eye = _EYE[K]
        for start in range(0, K, rows):
            x = eye[start:start + rows]
            if x.shape[0] < rows:
                x = torch.cat((x, torch.zeros((rows - x.shape[0], K), dtype=x.dtype, device=x.device)))
            fn(x, out)
            got = out[: min(rows, K - start)]
            want = exact[start:start + got.shape[0]] * 8.0
            if not torch.isfinite(got).all() or not torch.equal(got.to(torch.float32), want.to(torch.float32)):
                return False, "one-hot mismatch at column %d" % start
        tol = TOL["f32" if f32 else "bf16"]
        for mm in (rows,) + ((m_odd,) if m_odd else ()):
            x = torch.randn((mm, K), generator=gen, device=w.device).to(torch.bfloat16)
            o = torch.empty((mm, n), dtype=dt, device=w.device)
            fn(x, o)
            ref = x.double() @ w[:n].double().t()
            err = float((o.double() - ref).norm() / ref.norm())
            row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            if not (err <= tol and row <= 5 * tol and torch.isfinite(o).all()):
                return False, "random m=%d rel L2 %.2e row %.2e > %.0e" % (mm, err, row, tol)
        return True, "ok"


    ESC_MAX = 4e-4          # target escape rate (weights); keeps the 32-entry per-warp list far from overflow
    LMAX = 8                # code length limit: 4 pairs per step <= 32 bits (one refill per step suffices)
    PK = 3                  # v3: the first PK bits of a code (LSB-first) determine its length (8-entry byte table, one PRMT)


    def _package_merge(cnt, L):
        """optimal prefix-code lengths (max length L) for positive counts (python list)."""
        n = len(cnt)
        if n == 1:
            return [1]
        leaves = sorted((float(q), (i,)) for i, q in enumerate(cnt))
        cur = list(leaves)
        for _ in range(L - 1):
            pk = [(cur[2 * i][0] + cur[2 * i + 1][0], cur[2 * i][1] + cur[2 * i + 1][1]) for i in range(len(cur) // 2)]
            cur = sorted(leaves + pk, key=lambda t: t[0])
        ln = [0] * n
        for _, syms in cur[:2 * n - 2]:
            for s in syms:
                ln[s] += 1
        return ln


    def _canonical_lsb(lengths):
        """canonical codes, bit-reversed: the code of length l is the low l bits of the LSB-first peek window."""
        order = sorted(range(len(lengths)), key=lambda s: (lengths[s], s))
        code, prev, out = 0, None, [0] * len(lengths)
        for s in order:
            l = lengths[s]
            if prev is not None:
                code = (code + 1) << (l - prev)
            prev = l
            out[s] = int(format(code, "0%db" % l)[::-1], 2)
        return out


    def _geo(geo):
        wr, wk, ku, s, rt = geo[:5]
        return wr, wk, ku, s, rt


    def _to_segments(x, T, KC, geo):
        """[T, KC, 4, 32, 4, ...] (tile, chunk, step j, lane, reg-in-step i) -> [NSEG, 32, KU*RT*16, ...] in segment
    (gw) order and lane-stream order (u, jt, r = 4 j + i)."""
        wr, wk, ku, s, rt = _geo(geo)
        gx = T // (wr * rt)
        rest = x.shape[5:]
        x = x.reshape(gx, wr, rt, s, wk, ku, 4, 32, 4, *rest)
        nd = len(rest)
        x = x.permute(3, 0, 4, 1, 7, 5, 2, 6, 8, *range(9, 9 + nd))  # by, bx, wk, wr, lane, u, jt, j, i
        return x.reshape(s * gx * wk * wr, 32, ku * rt * 16, *rest)


    def _gvesc_e(t, c, j, l, v, x, T, geo, dev):
        """r6's _gvesc structure with an 8-bit exponent XOR (bits 0-7; the kernel applies x << (7 + 16 * (v & 1)))."""
        wr, wk, ku, s, rt = _geo(geo)
        gx, nw = T // (wr * rt), wr * wk
        out = torch.full((gx * s * nw * 32,), -1, dtype=torch.int64, device=dev)
        if len(t):
            gw = ((c // (wk * ku)) * gx + t // (wr * rt)) * nw + ((c // ku) % wk) * wr + (t // rt) % wr
            seq = (c % ku) * rt + t % rt
            o = torch.argsort(gw * 4096 + seq)
            gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | x)[o]
            rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
            if int(rank.max()) >= 32:
                return None
            out[gw * 32 + rank] = ent
        return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()


    def _prefix_code(cnt, k=PK, L=LMAX):
        """optimal prefix code (max length L) in which the first k bits of every code determine its length: codes of length
    <= k, and buckets (one k-bit prefix each) of 2^(Lb - k) codes of length Lb. cnt: python list (any order).
    Returns (lengths, codes LSB-first, tab[2^k] = length per k-bit window)."""
        n, P = len(cnt), 2 ** k
        order = sorted(range(n), key=lambda s: (-cnt[s], s))
        p = [float(cnt[s]) for s in order]
        cum = [0.0]
        for x in p:
            cum.append(cum[-1] + x)
        INF = float("inf")
        D = [[INF] * (P + 1) for _ in range(n + 1)]
        B = [[None] * (P + 1) for _ in range(n + 1)]
        D[0][0] = 0.0
        for i in range(n):
            for u in range(P + 1):
                d = D[i][u]
                if d == INF:
                    continue
                for l in range(1, k + 1):
                    w = 2 ** (k - l)
                    if u + w <= P and d + p[i] * l < D[i + 1][u + w]:
                        D[i + 1][u + w], B[i + 1][u + w] = d + p[i] * l, (i, u, "s", l)
                if u + 1 <= P:
                    for lb in range(k + 1, L + 1):
                        j = min(n, i + 2 ** (lb - k))
                        v = d + (cum[j] - cum[i]) * lb
                        if v < D[j][u + 1]:
                            D[j][u + 1], B[j][u + 1] = v, (i, u, "b", lb)
        u = min(range(P + 1), key=lambda q: D[n][q])
        assert D[n][u] < INF, "no prefix code fits"
        items, i = [], n
        while i > 0:
            pi, pu, kind, l = B[i][u]
            items.append((kind, l, order[pi:i]))
            i, u = pi, pu
        items.reverse()
        # canonical assignment in the k-bit prefix space (virtual length: l for short codes, k for buckets), then bit-reversed
        items.sort(key=lambda it: it[1] if it[0] == "s" else k)
        lengths, codes, tab = [0] * n, [0] * n, [L] * P
        code, prev = 0, None
        for kind, l, syms in items:
            vl = l if kind == "s" else k
            if prev is not None:
                code = (code + 1) << (vl - prev)
            prev = vl
            pre = int(format(code, "0%db" % vl)[::-1], 2)
            if kind == "s":
                lengths[syms[0]], codes[syms[0]] = l, pre
                for w in range(P):
                    if w & (2 ** l - 1) == pre:
                        tab[w] = l
            else:
                tab[pre] = l
                for j, s in enumerate(syms):
                    lengths[s], codes[s] = l, pre | (j << k)
        return lengths, codes, tab


    def _wpp(ilp):
        """words per ring piece. v3: 4 B pieces at every ilp (padding to a word, not 16 B: ilp 1/2/4 pad 0.12/0.22/0.36
    bits/w against 0.16/0.55/0.36 with 16 B pieces); the code supports 16 B pieces with per-lane word rotation"""
        return 1


    def _substreams(x, ilp):
        """[NSEG, 32, KU*RT*16, ...] (unit, register) -> [NSEG, ilp, 32, KU*RT*16/ilp, ...]: substream s holds steps
    j in [s*4/ilp, (s+1)*4/ilp) of every unit, in unit order"""
        nseg, _, n = x.shape[:3]
        rest = x.shape[3:]
        x = x.reshape(nseg, 32, n // 16, ilp, 16 // ilp, *rest)
        x = x.permute(0, 3, 1, 2, 4, *range(5, 5 + len(rest)))
        return x.reshape(nseg, ilp, 32, n // ilp, *rest)


    def pack_e(fr, geo, layer="?", verbose=True, ilp=1):
        """v3. r6 fragments [T, KC, 4, 32, 8] bf16 + gv geometry (WR, WK, KU, S, RT, ...) -> dict or None (escape overflow).
    keys: plane uint8 [T, KC, 2, 32, 16]; lut int32 [258] (256 value entries + the 8-byte length table); stream int32;
    soff int32 [NSEG + 1] (in 4 B words; a segment = ilp substream regions of P * 32 pieces of 16 B, 4 B at ilp 4);
    esc int32 [NSEG * 32]; ilp; stats dict."""
        t0 = time.time()
        dev = fr.device
        T, KC = fr.shape[:2]
        wr, wk, ku, s, rt = _geo(geo)
        assert wk * ku * s == KC and T % (wr * rt) == 0 and ilp in (1, 2, 4), (geo, T, KC, ilp)
        b = fr.reshape(T, KC, 4, 32, 8).contiguous().view(torch.int16).int() & 0xffff
        e8 = (b >> 7) & 0xff
        plane = ((b >> 8) & 0x80) | (b & 0x7f)
        n = b.numel()

        cnt = torch.bincount(e8.flatten(), minlength=256)
        val = torch.argsort(cnt, descending=True, stable=True)
        rank = torch.empty(256, dtype=torch.long, device=dev)
        rank[val] = torch.arange(256, device=dev)
        rk = rank[e8].view(T, KC, 4, 32, 4, 2)
        pid = rk[..., 0] * 256 + rk[..., 1]
        pc = torch.bincount(pid.flatten(), minlength=65536)
        order = torch.argsort(pc, descending=True, stable=True)
        npos = int((pc > 0).sum())
        pcf = pc.view(256, 256)
        N = None
        for cand in list(range(16, 256, 8)) + [256]:
            cand = min(cand, npos)
            C = torch.zeros(65536, dtype=torch.bool, device=dev)
            C[order[:cand]] = True
            C = C.view(256, 256)
            one = C.any(1)[:, None] | C.any(0)[None, :]
            esc = int((pcf * (~C) * (2 - one.long())).sum())
            if esc <= ESC_MAX * n or cand == npos:
                N = cand
                break
        syms = order[:N]
        C = torch.zeros(65536, dtype=torch.bool, device=dev)
        C[syms] = True
        Cm = C.view(256, 256)
        a_i = torch.arange(256, device=dev)
        rowbest = torch.where(Cm, pcf, -1).argmax(1)
        colbest = torch.where(Cm, pcf, -1).argmax(0)
        A, B = torch.meshgrid(a_i, a_i, indexing="ij")
        ph = torch.where(Cm, A * 256 + B,
                         torch.where(Cm.any(1)[:, None], A * 256 + rowbest[:, None],
                                     torch.where(Cm.any(0)[None, :], colbest[None, :] * 256 + B, syms[0].expand(256, 256))))
        ph = ph.flatten()
        eff = torch.bincount(ph[pid.flatten()], minlength=65536)
        symlist = syms.tolist()
        ln, code, tab = _prefix_code([max(int(eff[x]), 1) for x in symlist])
        sym_len = torch.zeros(65536, dtype=torch.long, device=dev)
        sym_code = torch.zeros(65536, dtype=torch.long, device=dev)
        sym_len[syms] = torch.tensor(ln, device=dev)
        sym_code[syms] = torch.tensor(code, device=dev)
        x = torch.arange(256, device=dev)
        sv = torch.tensor(symlist, device=dev)
        ent = (val[sv // 256] << 7) | (val[sv % 256] << 23) | torch.tensor(ln, device=dev)
        hit = (x[None, :] & (2 ** torch.tensor(ln, device=dev)[:, None] - 1)) == torch.tensor(code, device=dev)[:, None]
        assert bool((hit.sum(0) <= 1).all())
        lut = (hit.long() * ent[:, None]).sum(0)
        tabv = sum(t << (8 * i) for i, t in enumerate(tab))
        lut = torch.cat((lut, torch.tensor([tabv & 0xffffffff, tabv >> 32], device=dev)))

        php = ph[pid]
        pr = torch.stack((php // 256, php % 256), -1)
        mis = pr != rk
        ei = mis.nonzero()
        t_, c_, j_, l_, i_, h_ = ei.unbind(1)
        xr = (val[rk[mis]] ^ val[pr[mis]]).long()
        escl = _gvesc_e(t_, c_, j_, l_, 2 * i_ + h_, xr, T, geo, dev)
        if escl is None:
            if verbose:
                print("STATS_E %s escape list overflow (esc=%.5f)" % (layer, len(ei) / n))
            return None

        L = _substreams(_to_segments(sym_len[php], T, KC, geo), ilp)                 # [NSEG, ilp, 32, NP]
        V = _substreams(_to_segments(sym_code[php], T, KC, geo), ilp)
        nseg = L.shape[0]
        wpp = _wpp(ilp)                                                              # words per piece: 16 B, or 4 B at ilp 4
        off = torch.cumsum(L, 3) - L
        tot = L.sum(3)                                                               # [NSEG, ilp, 32]
        P = (tot.flatten(1).max(1).values + 32 * wpp - 1) // (32 * wpp)              # pieces per lane per substream
        soff = torch.zeros(nseg + 1, dtype=torch.long, device=dev)                   # in words (4 B)
        soff[1:] = torch.cumsum(P * 32 * ilp * wpp, 0)
        nwords = int(soff[-1])
        wd, sh = off >> 5, off & 31
        lane = torch.arange(32, device=dev)[None, None, :, None]
        sub = torch.arange(ilp, device=dev)[None, :, None, None]
        rot = ((lane >> 3) & 3) if wpp == 4 else 0
        gword = lambda w: soff[:-1, None, None, None] + ((sub * P[:, None, None, None] + w // wpp) * 32 + lane) * wpp + ((w % wpp) ^ rot)
        words = torch.zeros(nwords + 4, dtype=torch.long, device=dev)
        words.scatter_add_(0, gword(wd).flatten(), ((V << sh) & 0xffffffff).flatten())
        spill = (sh + L) > 32
        words.scatter_add_(0, gword(wd + 1)[spill], (V >> (32 - sh))[spill])
        words = words[:nwords]
        stream = torch.where(words >= 2 ** 31, words - 2 ** 32, words).int()

        pl = plane.to(torch.uint8)
        p0 = pl[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        p1 = pl[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
        planeb = torch.stack((p0, p1), 2).contiguous()
        lut32 = torch.where(lut >= 2 ** 31, lut - 2 ** 32, lut).int()

        stream_bits = nwords * 32
        pad_bits = stream_bits - int(tot.sum())
        total_bits = 8 * n + stream_bits + 32 * (nseg + 1) + 32 * 258 + 32 * escl.numel()
        st = dict(bits=total_bits / n, esc=len(ei) / n, maxpad=pad_bits / stream_bits, maxpad_bits=pad_bits / n,
                  N=N, code_bits=float(tot.sum()) / n, secs=time.time() - t0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            st["secs"] = time.time() - t0
        if verbose:
            print("STATS_E %s ilp%d bits=%.4f esc=%.5f maxpad=%.4f (code %.4f, pad %.4f bits/w, alphabet %d, len tab %s, %.2f s)"
                  % (layer, ilp, st["bits"], st["esc"], st["maxpad"], 8 + st["code_bits"], st["maxpad_bits"], N,
                     "".join(map(str, tab)), st["secs"]))
        return dict(plane=planeb, lut=lut32, stream=stream, soff=soff.int(), esc=escl, ilp=ilp, stats=st)


    def unpack_e(pk, T, KC, geo, ED=16):
        """v3 reference decoder: the device routine's logic per lane and substream (64-bit buffer, refill at step start when
    (V & 0x60) == 0, length = tab[lo & 7] (no memory on the chain), value = lut[lo & 0xff], V -= length), all lanes of
    all segments at once. Replays the per-substream smem ring protocol (ED words per lane in total, ED / K slots per
    substream; prologue: 3 rounds of NI = 4 / K predicated issues per substream + commit; per unit: one round + commit +
    wait_group(2)) and asserts every refill word (incl. the kernel's prefetched ones) is read from a landed, not yet
    overwritten piece. Returns bf16 bit patterns [T, KC, 4, 32, 8]."""
        dev = pk["stream"].device
        wr, wk, ku, s, rt = _geo(geo)
        K = pk["ilp"]
        NS = 4 // K
        M32 = 0xffffffff
        words = pk["stream"].long() & M32
        lut = pk["lut"].long() & M32
        tab = torch.tensor([((int(lut[256 + i // 4]) >> (8 * (i % 4))) & 255) for i in range(8)], device=dev)
        lut = lut[:256]
        soff = pk["soff"].long()
        nseg = soff.numel() - 1
        wpp = _wpp(K)
        P = (soff[1:] - soff[:-1]) // (32 * K * wpp)
        lane = torch.arange(32, device=dev)[None, None, :]
        sub = torch.arange(K, device=dev)[None, :, None]
        rot = ((lane >> 3) & 3) if wpp == 4 else 0
        z = torch.zeros(nseg, K, 32, dtype=torch.long, device=dev)
        lo, hi, Vr, ptr, iss = z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
        NU = ku * rt
        E = torch.zeros(nseg, K, 32, NU, NS, 4, dtype=torch.long, device=dev)
        Pl = P[:, None, None].expand_as(z)
        groups = []
        ES, NI = ED // K, NS                                                         # slots per substream, issues per unit

        def issue():
            nonlocal iss
            for _ in range(NI):
                go = (iss < Pl) & (iss < ptr // wpp + ES)
                iss = iss + go.long()
            groups.append(iss.clone())

        for _ in range(3):
            issue()
        for un in range(NU):
            issue()
            landed = groups[len(groups) - 1 - 2]
            for t in range(NS):
                need = (Vr & 0x60) == 0
                pc = ptr // wpp                                                    # ptr counts words
                inb = pc < Pl
                rd = need & inb                                                    # words the decode uses (other reads are masked)
                assert bool((pc < landed)[rd].all()), "ring: piece not landed"
                assert bool((pc + ES > iss - 1)[rd].all()), "ring: slot overwritten"
                gi = soff[:-1, None, None] + ((sub * P[:, None, None] + pc) * 32 + lane) * wpp + ((ptr % wpp) ^ rot)
                w = torch.where(need & inb, words[gi.clamp(max=words.numel() - 1)], 0)
                at = Vr & 31
                lo = torch.where(need, lo | ((w << at) & M32), lo)
                hi = torch.where(need, hi | ((w >> (32 - at)) * (at > 0)), hi)
                Vr = torch.where(need, (Vr + 32) & M32, Vr)
                ptr = ptr + need.long()
                for q in range(4):
                    n_ = tab[lo & 7]
                    E[..., un, t, q] = lut[lo & 0xff]
                    lo, hi = ((lo >> n_) | (hi << (32 - n_))) & M32, hi >> n_
                    Vr = (Vr - n_) & M32
        # [nseg, K, 32, NU, NS, 4] -> [nseg, 32, NU, 4 j, 4 i] -> fragment order
        E = E.permute(0, 2, 3, 1, 4, 5).reshape(nseg, 32, NU, 4, 4)
        gx = T // (wr * rt)
        E = E.view(s, gx, wk, wr, 32, ku, rt, 4, 4).permute(1, 3, 6, 0, 2, 5, 7, 4, 8).reshape(T, KC, 4, 32, 4)
        pl = pk["plane"].long()
        pb = pl.view(T, KC, 2, 32, 2, 8).permute(0, 1, 2, 4, 3, 5).reshape(T, KC, 4, 32, 8)
        lowh, hih = pb[..., 0::2], pb[..., 1::2]
        dup0, dup1 = lowh | (lowh << 8), hih | (hih << 8)
        m = 0x807f
        v0 = (E & ~m & 0xffff) | (dup0 & m)
        v1 = ((E >> 16) & ~m & 0xffff) | (dup1 & m)
        vals = torch.stack((v0, v1), -1).reshape(T, KC, 4, 32, 8)
        esc = pk["esc"].long() & M32
        ent = esc[esc != M32]
        if len(ent):
            gwi = (esc != M32).nonzero().flatten() // 32
            seq, ln_, r, half, x = ent >> 20, (ent >> 15) & 31, (ent >> 11) & 15, (ent >> 10) & 1, ent & 0xff
            nw = wr * wk
            wwr, wwk, g2 = gwi % nw % wr, gwi % nw // wr, gwi // nw
            bx, by = g2 % gx, g2 // gx
            t = (bx * wr + wwr) * rt + seq % rt
            c = (by * wk + wwk) * ku + seq // rt
            v = (r % 4) * 2 + half
            vals[t, c, r // 4, ln_, v] ^= x << 7
        return vals.int()


    def _frag_bits(fr):
        return fr.reshape(*fr.shape[:2], 4, 32, 8).contiguous().view(torch.int16).int() & 0xffff


    def _gepack(st, layer, inst, n):
        """codec E v3 pack for one geometry and substream count, proven lossless by the reference unpacker (incl. the
    per-substream ring protocol at the instance's ring size) before any kernel uses it. pack_e runs once per (layer,
    rows, geometry, substreams). Returns the pack dict or None (escape overflow, >= 12 bits, or a failed proof)."""
        fmt, mt, stg, wr, wk, ku, s, rt, xs, v = inst
        ed = 32 if (v >> 19) & 1 else 16
        ilp = 1 << ((v >> 22) & 3)
        geo = (wr, wk, ku, s, rt)
        cache = st.setdefault("gepk", {})
        key = (layer, n, geo, ilp, ed)
        if key in cache:
            return cache[key]
        t0 = time.time()
        fr = st[layer][0][: n // 16]  # unpooler at m <= 8 serves rows 0..1023 only
        T, KC = fr.shape[:2]
        tag = "%s n=%d g%d.%d.%d.%d.%d" % (layer, n, wr, wk, ku, s, rt)
        base = (layer, n, geo, ilp)
        if base not in cache:
            cache[base] = pack_e(fr, geo, tag, ilp=ilp)
            if fr.is_cuda:
                torch.cuda.synchronize()
            print("PREP gepack %s ilp%d pack_e %.2fs" % (tag, ilp, time.time() - t0), flush=True)
        pk = cache[base]
        if pk is not None:
            if pk["stats"]["bits"] >= 12.0:
                print("GE %s ilp%d: %.3f bits >= 12, not served" % (tag, ilp, pk["stats"]["bits"]), flush=True)
                pk = None
            else:
                try:
                    good = torch.equal(unpack_e(pk, T, KC, geo, ed), _frag_bits(fr))
                except AssertionError as err:  # ring protocol violated for this data at this ring size
                    print("GE %s ilp%d ed%d: %s, not served" % (tag, ilp, ed, err), flush=True)
                    good = False
                if not good:
                    print("GE %s ilp%d ed%d: reference unpack failed, not served" % (tag, ilp, ed), flush=True)
                    pk = None
        cache[key] = pk
        if fr.is_cuda:
            torch.cuda.synchronize()
        print("PREP gepack %s ilp%d ed%d %s in %.2fs" % (tag, ilp, ed, "proven bit-exact" if pk is not None else "declined", time.time() - t0), flush=True)
        return pk


    def _gxplan(st, layer, inst, n, packs):
        fmt, mt, stg, wr, wk, ku, s, rt, xs, v = inst
        if fmt == 4:  # codec E v3: W = plane, W2 = stream, esc = 8-bit XOR list, T = LUT (+ length table) pointer, U = soff pointer
            pk = _gepack(st, layer, inst, n)
            if pk is None:
                return None
            return (GVI.index(inst), pk["plane"], pk["esc"], pk["lut"].data_ptr(), n, pk["stream"], pk["soff"].data_ptr(), pk)
        if not fmt & 1:  # plain bf16 fragments (r6's FMT 0 / router hi+lo FMT 2): no escapes
            frs = list(st[layer]) if fmt == 2 else [st[layer][0]]
            return (GVI.index(inst), frs[0], st["gvnoesc"], 0, n, frs[-1], 0)
        if layer not in packs:
            return None
        css, esc = packs[layer]
        e = _gvesc(esc, n // 16, (wr, wk, ku, s, rt, xs))
        if e is None:
            return None  # an escape list overflows (32 per warp): the path is not registered
        cs = css[v & 1]
        src = [c[0] for c in cs]
        return (GVI.index(inst), src[0], e, cs[0][1], n, src[-1], cs[-1][1])


    def _gxbake(st):
        ext = _gvext()
        dev = st["qkv_gate_raw"].device
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(1234)
        packs = dict(st.get("gvc12", {}))
        frs = list(st["router"])  # 12-bit hi and lo: the lo half must fit the escape lists too (checked per geometry)
        rc = [[_c12(f, dv) for f in frs] for dv in (0, 1)] if _NCR("router", (1, 8, 16, 32, 128, 256)) else [[None]]
        if all(c is not None for cs in rc for c in cs):
            packs["router"] = (rc, torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(rc[1])]))
            print("GX router c12 escapes hi %d lo %d" % (len(rc[1][0][2]), len(rc[1][1][2])), flush=True)
        st["gvnoesc"] = torch.full((1 << 16,), -1, dtype=torch.int32, device=dev)
        res_all = {}
        rw = st["router_raw"]
        rexact = rw.to(torch.bfloat16).float() + (rw - rw.to(torch.bfloat16).float()).to(torch.bfloat16).float()
        todo = []
        for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
            for m in (1, 8, 16, 32, 128, 256):
                ps = [(nm, g, v) for nm, ly, ms, g, v in GXP if ly == layer and m in ms]
                if ps and _NC(layer, m):
                    todo.append((layer, m, ps))
        for layer, m, ps in todo:
            f32 = layer == "router"
            w = st[layer + "_raw"]
            K = w.shape[1]
            n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
            x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
            r6 = lambda xx, oo, layer=layer, n=n: _r6(xx, layer, n, st)
            best = ("r6", None, _time_us(r6, x, out, flush, sink))
            cands = []
            print("BAKE %s m=%d r6 %.2f (ref)" % (layer, m, best[2]), flush=True)
            for name, geo, v in ps:
                if time.time() - _T0 > PREPARE_BUDGET_S:
                    print("BAKE %s m=%d %s skipped-budget" % (layer, m, name), flush=True)
                    continue
                try:
                    inst = _gxinst(geo, v, m)
                    plan = _gxplan(st, layer, inst, n, packs)
                    if plan is None:
                        print("BAKE %s m=%d %s declined-%s" % (layer, m, name, "pack" if inst[0] == 4 else "escapes"), flush=True)
                        continue
                    a = (plan[1], plan[5], x, out, plan[2], st["gvws"], st["gvcnt"], plan[3], plan[6])
                    grid = ((m + 8 * inst[1] - 1) // (8 * inst[1])) * (n // (16 * inst[3] * inst[7])) * inst[6]
                    res = int(ext.gvocc(plan[0], *a))
                    if inst[6] > 1 and not (v & 4) and not ((v >> 26) & 15) and res < grid:  # cluster variant: the whole grid must be co-resident
                        print("BAKE %s m=%d %s declined-occ %d<%d" % (layer, m, name, res, grid), flush=True)
                        continue
                    if _gvcall(st, plan, x, out) != 0:
                        print("BAKE %s m=%d %s ERROR launch" % (layer, m, name), flush=True)
                        continue
                    fn = lambda xx, oo, plan=plan: _gvcall(st, plan, xx, oo)
                    ok, msg = _validate(fn, w, m, n, f32, gen, rexact if f32 else None, {8: 5, 16: 9, 32: 17, 128: 121, 256: 250}.get(m))
                    if not ok:
                        print("BAKE %s m=%d %s INVALID %s" % (layer, m, name, msg), flush=True)
                        continue
                    us = _time_us(fn, x, out, flush, sink)
                except Exception as error:
                    print("BAKE %s m=%d %s ERROR %s" % (layer, m, name, str(error).splitlines()[-1][:160]), flush=True)
                    continue
                print("BAKE %s m=%d %s %.2f (ref r6 %.2f)%s"
                      % (layer, m, name, us, best[2], "" if res >= grid else " (res %d<%d)" % (res, grid)), flush=True)
                res_all[(layer, m, name)] = (us, fn)
                cands.append((us, name, plan))
            # one reference per group, same rule as the rq bakes (the gx times are absolute, all against
            # the same r6 measurement): the proven pick is
            # the reference when this GPU does not read it slower than r6 (window = half the margin), else r6; the pick is
            # the best candidate if it beats the reference by MARGIN_GX, else the reference.
            prov = next((c for c in cands if c[1] == PROVEN_GX.get((layer, m))), None)
            refn, refp, refmed = "r6", None, best[2]
            if prov is not None:
                rn, rm, isp = _refsel(prov[1], prov[0], "r6", best[2], RWIN_GX)
                if isp:
                    refn, refp, refmed = rn, prov[2], rm
            cl = [(c[1], c[2], refmed / c[0]) for c in cands if not (refp is not None and c is prov)]
            best = _pickrule((refn, refp, refmed), cl, MARGIN_GX)
            if _PRUNE and (layer, m) in FORCE_GX and prov is not None:
                best = (prov[1], prov[2], prov[0])
            if best[1] is not None:
                st["sel"][(layer, m)] = best[1]
            print("PICK %s m=%d %s %.2f (ref %s %.2f)" % (layer, m, best[0], best[2], refn, refmed), flush=True)
            if P1ALT and best[1] is not None and layer in ("router", "unpooler") and m <= 32 and time.time() - _T0 < PREPARE_BUDGET_S - 30:
                fb = lambda xx, oo, plan=best[1]: _gvcall(st, plan, xx, oo)
                tb, tr = [], []
                for _ in range(3):  # the pick vs r6, alternated (the bake's 15-replay medians drift ~0.3 us between runs)
                    tb.append(_time_us(fb, x, out, flush, sink, 21)); tr.append(_time_us(r6, x, out, flush, sink, 21))
                print("PROBE p1alt %s m=%d %s %.2f r6=%.2f delta=%+.2f" % (layer, m, best[0], min(tb), min(tr), min(tb) - min(tr)), flush=True)
            if layer == "pooler" and m in (1, 16):  # PROBE ecod: the best codec E path vs r6, alternated
                cand = sorted((v_[0], k[2], v_[1]) for k, v_ in res_all.items() if k[:2] == (layer, m) and k[2].startswith("ge-"))
                if not cand:
                    print("PROBE ecod pooler m=%d none" % m, flush=True)
                elif time.time() - _T0 > PREPARE_BUDGET_S:
                    print("PROBE ecod pooler m=%d skipped-budget" % m, flush=True)
                else:
                    tb, tr = [], []
                    for _ in range(3):
                        tb.append(_time_us(cand[0][2], x, out, flush, sink, 21))
                        tr.append(_time_us(r6, x, out, flush, sink, 21))
                    print("PROBE ecod pooler m=%d %s %.2f r6=%.2f delta=%+.2f" % (m, cand[0][1], min(tb), min(tr), min(tb) - min(tr)), flush=True)


    RWCU = __cuda__("streamk_rwcu.cu")
    RWFLAGS = ["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-DNDEBUG"]
    # variant -> rwk<KR, NSTG, NBUF, TOKS, WN, MINB>
    RWV = {100: (256, 4, 4, 128, 128, 1), 101: (256, 8, 8, 128, 128, 1), 102: (128, 4, 4, 128, 128, 2), 103: (512, 8, 6, 128, 128, 1),
           110: (256, 4, 4, 256, 256, 1), 111: (256, 8, 8, 256, 256, 1), 112: (128, 4, 4, 256, 256, 2), 113: (256, 8, 8, 256, 128, 1),
           114: (512, 8, 4, 256, 256, 1)}
    RWV = {k: v + (0, 1) for k, v in RWV.items()}  # + (MODE, CL)
    RWV.update({104: (256, 4, 3, 128, 128, 2, 0, 1), 105: (256, 4, 2, 128, 128, 2, 0, 1), 106: (256, 4, 2, 128, 128, 2, 4, 1),
                120: RWV[100][:6] + (4, 1), 121: RWV[100][:6] + (0, 2), 123: RWV[100][:6] + (4, 2), 126: RWV[101][:6] + (4, 1),
                127: RWV[102][:6] + (4, 1), 107: (256, 4, 3, 128, 128, 2, 4, 1), 140: (256, 4, 3, 128, 128, 2, 4, 2),
                141: (256, 4, 3, 128, 128, 2, 0, 2), 142: RWV[101][:6] + (4, 2),
                150: RWV[101][:6] + (12, 1), 151: (256, 4, 3, 128, 128, 2, 12, 1), 152: (256, 4, 4, 128, 128, 1, 12, 1),
                130: RWV[113][:6] + (4, 1), 131: RWV[113][:6] + (0, 2), 136: RWV[112][:6] + (4, 1)})
    RWV.update({108: (512, 8, 3, 128, 128, 2, 0, 1)})
    RWST = (101, 104, 107, 103, 108, 113, 130)  # MODE 16 (globaltimer stamps) instances: variant + 20000
    RWV.update({v + 20000: RWV[v][:6] + (RWV[v][6] | 16, RWV[v][7]) for v in RWST})
    RTV = {200: (512, 6, 128, 128, 1, 4), 201: (256, 3, 128, 128, 2, 4), 202: (256, 4, 256, 128, 1, 4), 203: (512, 4, 256, 128, 1, 4),
           204: (1024, 6, 128, 128, 1, 4), 206: (512, 6, 128, 128, 1, 0), 210: (256, 4, 128, 128, 1, 4), 211: (256, 3, 128, 128, 2, 4),
           230: (512, 6, 128, 128, 1, 32), 231: (256, 3, 128, 128, 2, 32), 232: (256, 4, 128, 128, 1, 32),
           240: (512, 6, 128, 128, 1, 68), 241: (256, 3, 128, 128, 2, 68), 242: (256, 4, 128, 128, 1, 68),
           250: (128, 3, 128, 128, 2, 4), 251: (256, 3, 128, 128, 2, 132), 252: (512, 3, 128, 128, 2, 4),
           253: (256, 3, 128, 128, 2, 196), 254: (128, 3, 128, 128, 2, 132), 255: (512, 3, 128, 128, 2, 132),
           256: (256, 3, 128, 128, 1, 548), 257: (512, 3, 128, 128, 1, 548), 258: (512, 3, 128, 128, 1, 32),
           261: (256, 3, 128, 128, 2, 36), 262: (256, 3, 128, 128, 2, 548),
           263: (256, 4, 128, 128, 2, 36), 264: (512, 3, 128, 128, 2, 36),
           265: (256, 3, 128, 128, 2, 1028), 266: (256, 3, 128, 128, 2, 1156), 267: (256, 3, 128, 128, 2, 1220),
           268: (256, 4, 128, 128, 1, 132), 269: (256, 4, 128, 128, 1, 1156),
           270: (256, 3, 128, 128, 2, 2180), 271: (256, 3, 128, 128, 2, 6276), 272: (256, 3, 128, 128, 2, 2052),
           273: (256, 3, 128, 128, 2, 6148), 274: (512, 3, 128, 128, 2, 6276),
           275: (256, 3, 128, 128, 2, 3076), 276: (256, 3, 128, 128, 2, 2116), 277: (256, 4, 128, 128, 1, 2052),
           278: (256, 3, 64, 64, 1, 4),  # TOKS 64 / WN 64: MINB 2 needs >168 regs (ptxas C7600), MINB 1 = 2 waves at m=128
           282: (512, 6, 128, 128, 1, 2052), 283: (512, 6, 128, 128, 1, 3076), 284: (512, 6, 128, 128, 1, 6148),
           285: (256, 2, 128, 128, 2, 132), 286: (256, 2, 128, 128, 2, 4),
           288: (256, 3, 128, 128, 2, 3204), 289: (256, 3, 128, 128, 2, 3140), 290: (256, 4, 128, 128, 1, 1156)}
    # 220-223 (multicast + self-zeroing) deadlock when not every cluster is co-resident (the zeroing wait spans clusters): not built
    RTV.update({v + 20000: c[:5] + (c[5] | 16,) for v, c in list(RTV.items())})
    RWLAD = True
    if RWLAD:
        RWV.update({k + 1000 * md: RWV[k][:6] + (md, 1) for k in (101, 104) for md in (1, 2, 3)})


    def _rwwg(n):
        """the SS wgmma m64nNk16 (bf16 in, fp32 accumulate in place) as an inline-asm device function"""
        r = n // 2
        regs = ", ".join("%%%d" % i for i in range(r))
        outs = ", ".join('"+f"(d[%d])' % i for i in range(r))
        return ('__device__ __forceinline__ void rw_mma%d(float* d, uint64_t a, uint64_t b) {\n'
                '#if defined(__CUDA_ARCH_FEAT_SM90_ALL)\n'
                '  asm volatile("wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, 1, 1, 1, 0, 0;\\n"\n'
                '    : %s : "l"(a), "l"(b));\n'
                '#endif\n}\n') % (n, n, regs, r, r + 1, outs)


    def _rwsrc():
        use = {v: c for v, c in RWV.items() if RWALL or v in {p[1] for p in RSP}}
        ruse = {v: c for v, c in RTV.items() if RWALL or v in {p[1] for p in RSP}}
        rtc = "\n".join("    case %d: return rt_go<%s>(xp, h, l, o, m, N, K, c, %d, st);" % (v, ", ".join(map(str, c)), 64 + i) for i, (v, c) in enumerate(sorted(RTV.items())) if v in ruse)
        cases = "\n".join("    case %d: return rw_go<%s>(xp, h, l, o, m, N, K, c, %d, st);" % (v, ", ".join(map(str, c)), i) for i, (v, c) in enumerate(sorted(RWV.items())) if v in use)
        occ = "\n".join("    case %d: return rw_occ<%s>(m, N, K);" % (v, ", ".join(map(str, c))) for v, c in sorted(use.items()))
        return RWCU.replace("@@WG@@", _rwwg(128) + _rwwg(256)).replace("@@RWCASES@@", cases).replace("@@RWOCC@@", occ).replace("@@RTCASES@@", rtc).replace("@@RTOCC@@", "\n".join("    case %d: return rt_occ<%s>(m, N, K);" % (v, ", ".join(map(str, c))) for v, c in sorted(ruse.items())))


    _rwmod = []


    def _rwext():
        if not _rwmod:
            import zlib
            t = time.time()
            src = _rwsrc()
            import os
            arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"  # torch otherwise adds compute_90 and ptxas rejects wgmma there
            try:
                _rwmod.append(load_inline("rw29_%x" % zlib.crc32(src.encode()), cpp_sources="int64_t rwrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor); int64_t rwocc(int64_t, int64_t, int64_t, int64_t);",
                                          cuda_sources=src, functions=["rwrun", "rwocc"], extra_cuda_cflags=RWFLAGS, verbose=False))
            finally:
                if arch is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            print("PREP rw compiled in %.0fs (at %.0fs)" % (time.time() - t, time.time() - _T0), flush=True)
        return _rwmod[0]


    RSPALL = [("rs64k32", 0, True, (128, 256)), ("rp64k32s4", 5, True, (128,)), ("rp64k32s2", 6, True, (128,)),
           ("rp64k64x2", 7, True, (128,)), ("rp64k32sc", 8, False, (128,)),
           ("rv64k32", 9, True, (256,)), ("rv64k64x2", 10, True, (256,)), ("rv64k32sc", 11, False, (256,)),
           # rw64k32s4 / s8 = 128 CTAs, 4 / 8 stages; rw64k64s4x2 = KR 128, 256
           # CTAs at 2/SM (also m=256 as two 128-token blocks, 512 CTAs); rw64k16r6 = KR 512 (64 CTAs), ring of 6 slots;
           # rwv* = both 128-token blocks in one CTA (n256, or h2 = two n128 sets), rwv64k16r4 = 64 CTAs, ring of 4
           ("rw64k32s4", 100, True, (128,)), ("rw64k32s8", 101, True, (128, 256)), ("rw64k64s4x2", 102, True, (128, 256)),
           ("rw64k16r6", 103, True, (128,)), ("rwv64k32s4", 110, True, (256,)), ("rwv64k32s8", 111, True, (256,)),
           ("rwv64k64s4x2", 112, True, (256,)), ("rwv64k32s8h2", 113, True, (256,)), ("rwv64k16r4", 114, True, (256,)),
           # z = self-zeroing (no memset node), cN = cluster of N K-splits pre-reducing through DSMEM
           ("rw64k32s4z", 120, False, (128,)), ("rw64k32s4c2", 121, True, (128,)), ("rw64k32s4zc2", 123, False, (128,)),
           ("rw64k32s8z", 126, False, (128, 256)), ("rw64k64s4x2z", 127, False, (128, 256)),
           ("rw64k32r3x2", 104, True, (128, 256)), ("rw64k32r2x2", 105, True, (128, 256)), ("rw64k32r2x2z", 106, False, (128, 256)),
           ("rw64k32r3x2z", 107, False, (128, 256)), ("rw64k16r3x2", 108, True, (128, 256)), ("rw64k32r3x2zc2", 140, False, (128, 256)), ("rw64k32r3x2c2", 141, True, (128, 256)), ("rw64k32s8zc2", 142, False, (128,)),
           ("rw64k32s8zr", 150, False, (128, 256)), ("rw64k32r3x2zr", 151, False, (128, 256)), ("rw64k32s4zr", 152, False, (128,)),
           ("rt64k16b6z", 200, False, (128, 256)), ("rt64k32b3x2z", 201, False, (128, 256)), ("rtv64k32b4z", 202, False, (256,)),
           ("rtv64k16b4z", 203, False, (256,)), ("rt64k8b6z", 204, False, (128, 256)), ("rt64k16b6", 206, True, (128, 256)),
           ("rt64k32b4z", 210, False, (128, 256)), ("rt64k32b3x2bz", 211, False, (128,)),
           ("rt64k16b6m", 230, True, (128, 256)), ("rt64k32b3x2m", 231, True, (128, 256)), ("rt64k32b4m", 232, True, (128,)),
           ("rt64k16b6zp", 240, False, (128, 256)), ("rt64k32b3x2zp", 241, False, (128, 256)), ("rt64k32b4zp", 242, False, (128,)),
           ("rwv64k32s8h2z", 130, False, (256,)), ("rwv64k32s8h2c2", 131, True, (256,)), ("rwv64k64s4x2z", 136, False, (256,))]

    RSP = [p for p in RSPALL if p[0] in ("rw64k32s8z", "rw64k32r3x2z", "rt64k32b4z", "rt64k32b3x2bz", "rt64k32b3x2z", "rt64k16b6z")]
    # m = 128: rw64k32s8z (the benchmark's current pick) vs the rtk paths; m = 256: rw64k32r3x2z vs rtk, all against r6 (F1).
    # One co-resident wave per self-zeroing grid (rt_go also declines otherwise).
    RSP = [(nm, v, zf, {"rw64k32s8z": (128,), "rt64k32b4z": (128,), "rt64k32b3x2bz": (128,), "rw64k32r3x2z": (256,),
                        "rt64k16b6z": (256,)}.get(nm, ms)) for nm, v, zf, ms in RSP]
    RSP = RSP + [
        ("rt64k32b4x2ze", 268, False, (128,)),         # NBUF 4 (whole tile in flight), 1 CTA/SM + early TMA
        ("rt64k32b3x2zk2pr", 289, False, (128, 256))]  # KC CLK 2 + prefetch + rotation
    # the proven pick is baked first and is the baseline.
    RSP = sorted(RSP, key=lambda p: 0 if p[0] == "rt64k32b3x2z" else 1)
    RWALL = False  # rbench sets True: build every RWV instance, not only the shipped ones


    P1ALT = True
    PR27 = False  # the rd8 ladder was read on the H100


    def _pr27(st):
        """PROBE rd8: the floor of an 8 MB read with every load issued at launch, in the bake's timing
    (cold-L2 graph replay, in-prepare): ld.global 16 B x NL per thread at G = 128 / 132 / 264 / 528 CTAs x 256 threads;
    + 64 fp32 red.adds per CTA into a memset-zeroed 8 KB output (memset in the graph); + a cluster-8 DSMEM reduce;
    cp.async.bulk 8 / 16 KB chunks per CTA. 'PROBE rd8 <name> <us> <tb/s> tbs'."""
        dev = st["pooler_raw"].device
        ext = _gvext()
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink32 = torch.empty((), dtype=torch.float32, device=dev)
        nb = 8 << 20
        buf = torch.randint(0, 255, (nb,), dtype=torch.uint8, device=dev)
        out = torch.zeros((2048,), dtype=torch.float32, device=dev)
        sink = torch.zeros((1024,), dtype=torch.int32, device=dev)
        x0 = torch.empty((1, 1), device=dev)
        zero = lambda: out.zero_()
        for name, kind, G, z in (("ldg-g128-nl16", 0, 128, False), ("ldg-g132-nl16", 0, 132, False), ("ldg-g264-nl8", 1, 264, False),
                                 ("ldg-g528-nl4", 2, 528, False), ("ldg-g1056-nl2", 3, 1056, False), ("ldg-g264-red", 4, 264, True),
                                 ("ldg-g264-c8", 5, 264, False), ("blk8k-g128", 6, 128, False), ("blk16k-g128", 7, 128, False),
                                 ("blk8k-g256", 6, 256, False)):
            try:
                def fn(xx, oo, kind=kind, G=G, z=z):
                    if z:
                        zero()
                    return ext.prrun(kind, G, buf, out, sink)
                code = fn(x0, out)
                if code:
                    print("PROBE rd8 %s launch-error %d" % (name, code), flush=True)
                    continue
                us = _time_us(fn, x0, out, flush, sink32)
                print("PROBE rd8 %s %.2f %.2f tbs" % (name, us, nb / us / 1e6), flush=True)
            except Exception as error:
                print("PROBE rd8 %s ERROR %s" % (name, str(error).splitlines()[-1][:120]), flush=True)
        print("PROBE rd8 empty-memset %.2f" % _time_us(lambda xx, oo: zero(), x0, out, flush, sink32), flush=True)


    def _rscall(st, p, x, out):
        if p[2]:
            out.zero_()  # a memset node in the captured graph (red.add accumulates into it)
        if p[1] >= 100:
            return _rwext().rwrun(p[1], x if x.data_ptr() % 16 == 0 else x.clone(), st["rs_hi"], st["rs_lo"], out, st["rw_cnt"])
        return _gvext().rsrun(p[1], x, st["rs_hi"], st["rs_lo"], out, st["rs_ws"], st["rs_cnt"])  # ws / cnt: rsk2 RED 2 only


    def _rsbake(st):
        dev = st["router_raw"].device
        st["rs_hi"], st["rs_lo"] = st["router_hl"][:256].contiguous(), st["router_hl"][256:].contiguous()
        st["rs_ws"] = torch.zeros((256 * 256,), dtype=torch.float32, device=dev)  # RED 2: [<= 256 tokens][256] fp32, zero between calls
        st["rs_cnt"] = torch.zeros((64,), dtype=torch.int32, device=dev)
        st["rw_cnt"] = torch.zeros((16384 + 2 * 8 * 1024,), dtype=torch.int32, device=dev)
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(4242)
        w = st["router_raw"]
        rexact = st["rs_hi"].float() + st["rs_lo"].float()
        n, K = w.shape
        rw = False
        # F1 baseline rule: r6 timed before the heavy work (the rw build) and again after the variants; the lower wins
        r6pre = {}
        for m in (128, 256):
            xx = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            oo = torch.empty((m, n), dtype=torch.float32, device=dev)
            r6pre[m] = _time_us(lambda a, b: _r6(a, "router", n, st), xx, oo, flush, sink)
        if time.time() - _T0 < PREPARE_BUDGET_S - 60:  # build the wgmma router extension; a failure drops only its paths
            try:
                _rwext()
                rw = True
            except Exception as error:
                print("PREP rw build failed: %s" % str(error).splitlines()[-1][:200], flush=True)
        else:
            print("PREP rw skipped-budget", flush=True)
        for m in (128, 256):
            x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            out = torch.empty((m, n), dtype=torch.float32, device=dev)
            r6 = lambda xx, oo: _r6(xx, "router", n, st)
            r6med = _time_us(r6, x, out, flush, sink)
            print("BAKE router m=%d r6 %.2f (pre %.2f) at %.0fs" % (m, r6med, r6pre[m], time.time() - _T0), flush=True)

            def _try(name, v, zf):
                """validate one router path; returns (fn, plan) or None"""
                plan = (-7, v, zf, None, n, None, None)
                fn = lambda xx, oo, plan=plan: _rscall(st, plan, xx, oo)
                try:
                    if fn(x, out) != 0:
                        print("BAKE router m=%d %s ERROR launch" % (m, name), flush=True)
                        return None
                    ok, msg = _validate(fn, w, m, n, True, gen, rexact, {128: 121, 256: 250}[m])
                    if not ok:
                        print("BAKE router m=%d %s INVALID %s" % (m, name, msg), flush=True)
                        return None
                except Exception as error:
                    print("BAKE router m=%d %s ERROR %s" % (m, name, str(error).splitlines()[-1][:160]), flush=True)
                    return None
                return fn, plan

            prov = next((p for p in RSP if p[0] == "rt64k32b3x2z" and m in p[3] and not (p[1] >= 100 and not rw)), None)
            refn, refp, refmed, reffn = "r6", None, r6med, r6
            cands, done = [], set()
            if prov is not None:
                done.add(prov[0])
                got = _try(prov[0], prov[1], prov[2])
                if got is not None:
                    fn, plan = got
                    us = _time_us(fn, x, out, flush, sink, base=r6)
                    print("BAKE router m=%d %s %.2f (ratio %.4f vs r6 %.2f) at %.0fs"
                          % (m, prov[0], us, _LAST[0][0], r6med, time.time() - _T0), flush=True)
                    rn, rm, isp = _refsel(prov[0], us, "r6", r6med, RWIN_RS)
                    if isp:
                        refn, refp, refmed, reffn = rn, plan, rm, fn
                    else:
                        cands.append((prov[0], plan, _LAST[0][0]))
            for name, v, zf, ms in RSP:
                if m not in ms or (v >= 100 and not rw) or name in done:
                    continue
                if time.time() - _T0 > PREPARE_BUDGET_S:
                    print("BAKE router m=%d %s skipped-budget" % (m, name), flush=True)
                    continue
                got = _try(name, v, zf)
                if got is None:
                    continue
                fn, plan = got
                us = _time_us(fn, x, out, flush, sink, base=reffn)
                print("BAKE router m=%d %s %.2f (ratio %.4f vs %s %.2f) at %.0fs"
                      % (m, name, us, _LAST[0][0], refn, refmed, time.time() - _T0), flush=True)
                cands.append((name, plan, _LAST[0][0]))
            best = _pickrule((refn, refp, refmed), cands, MARGIN_RS)
            if m == 128:  # PROBE rszero: the cost of the zeroing memset node alone (on the critical path of the red variants)
                print("PROBE rszero router m=128 %.2f" % _time_us(lambda xx, oo: oo.zero_(), x, out, flush, sink), flush=True)
            if best[1] is not None:
                st["sel"][("router", m)] = best[1]
            print("PICK router m=%d %s %.2f (ref %s %.2f)" % (m, best[0], best[2], refn, refmed), flush=True)
        # the self-cleaning workspace/counters are shared by every RED-2 path; a declined or failed path
        # may have left them dirty, so restore the all-zero invariant before any picked path serves the benchmark.
        st["rs_ws"].zero_(); st["rs_cnt"].zero_(); torch.cuda.synchronize()


    def _gxladder(st):
        """PROBE gvlad: where r6's gv time goes as m grows. Each step is timed with the bake's _time_us."""
        ext = _gvext()
        dev = st["qkv_gate_raw"].device
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(4321)
        packs = dict(st.get("gvc12", {}))
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        for layer in ("pooler", "qkv_gate"):
            for m in (1, 8, 16, 32):
                if time.time() - _T0 > PREPARE_BUDGET_S:
                    print("PROBE gvlad %s m=%d skipped-budget" % (layer, m), flush=True)
                    continue
                w = st[layer + "_raw"]
                n, K = w.shape
                x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                for ly, mt, geo, v, step in GLAD:
                    if ly != layer or GVB[m] != mt:
                        continue
                    try:
                        inst = _gxinst(geo, v, m)
                        plan = _gxplan(st, layer, inst, n, packs)
                        if plan is None:
                            print("PROBE gvlad %s m=%d %s declined" % (layer, m, step), flush=True)
                            continue
                        a = (plan[1], plan[5], x, out, plan[2], st["gvws"], st["gvcnt"], plan[3], plan[6])
                        if _gvcall(st, plan, x, out) != 0:
                            print("PROBE gvlad %s m=%d %s launch-error" % (layer, m, step), flush=True)
                            continue
                        us = _time_us(lambda xx, oo, plan=plan: _gvcall(st, plan, xx, oo), x, out, flush, sink)
                        print("PROBE gvlad %s m=%d %s %.2f regs=%d ctas/sm=%.2f" % (layer, m, step, us, int(ext.gvregs(plan[0], *a)), int(ext.gvocc(plan[0], *a)) / sms), flush=True)
                    except Exception as error:
                        print("PROBE gvlad %s m=%d %s ERROR %s" % (layer, m, step, str(error).splitlines()[-1][:120]), flush=True)
                us = _time_us(lambda xx, oo: _r6(xx, layer, n, st), x, out, flush, sink)
                p = st["gv"].get((layer, _bk(m)))
                info = ""
                if p is not None and p[0] < 1000:
                    a = (p[1], p[5], x, out, p[2], st["gvws"], st["gvcnt"], p[3], p[6])
                    info = " %s regs=%d ctas/sm=%.2f" % (_name(p[0]), int(ext.gvregs(p[0], *a)), int(ext.gvocc(p[0], *a)) / sms)
                print("PROBE gvlad %s m=%d L4 %.2f%s" % (layer, m, us, info or " v12"), flush=True)


    def kernel_fn(data, state):
        cfg = data["config"]
        layer, n = cfg["layer"], cfg["n"]
        x = data["x"]
        if "cpu" in state:
            w = state["raw"][layer]
            return x.float() @ w[:n].float().t() if layer == "router" else x @ w[:n].t()
        rp = state.get("rq", {}).get((layer, x.shape[0]))  # rs128: m exactly 128 / 256, picked in the bake
        if rp is not None and rp["n"] == n:
            out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
            if _rqcall(state, rp, x.contiguous(), out) == 0:
                return out
        p = state["sel"].get((layer, _bk(x.shape[0])))
        if p is not None and p[4] == n:
            out = torch.empty((x.shape[0], n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
            _gvcall(state, p, x.contiguous(), out)
            return out
        return _r6(x, layer, n, state)


    def _r6(x, layer, n, state):
        """r6-c12's kernel_fn, unchanged"""
        p = state["gv"].get((layer, _bk(x.shape[0])))
        if p is not None and p[4] == n:
            out = torch.empty((x.shape[0], n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
            _gvcall(state, p, x.contiguous(), out)
            return out
        return _v12(x, layer, n, state)


    def _v12(x, layer, n, state):
        m = x.shape[0]
        if _cublas(layer, m):  # cuBLAS beats the skinny kernel here (benchmark v7: 0.45-0.9x); algorithm tuned in prepare()
            i = state["lt"].get((layer, m))
            if i is None:
                return x @ state[layer + "_raw"][:n].T
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _ext().lt_mm(x.contiguous(), state[layer + "_raw"], state[layer + "_t"], out, n, i)
            return out
        if layer == "router" and m > 32:
            i = state["lt"].get(("router", m))
            if i is None:
                a = torch.mm(x, state["router_hl_t"], out_dtype=torch.float32)
            else:
                a = torch.empty((m, 2 * n), dtype=torch.float32, device=x.device)
                _ext().lt_mm(x.contiguous(), state["router_hl"], state["router_hl_t"], a, 2 * n, i)
            return a[:, :n] + a[:, n:]
        A, A2 = state[layer]
        out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        if layer == "router" and m <= 32 and RTR:
            _ext().run3(A, A2, x.contiguous(), out, state["ws"], state["cnt"], n)
            return out
        (_ext().run2 if V6 and m <= 32 else _ext().run)(A, A2 if A2 is not None else state["none"], x.contiguous(), out, state["ws"], state["cnt"], n)
        return out


    RQV = [  # (name, NT, ST, GS, PRB, WM, INS, FL): ST k64 slots (W 12 KB + x NT*128 B each), GS k64 stages per consumer
        # round, WM 0 wait_group 1 + lagged x release / 1 decode-next under the wgmma + wait_group 0, FL bit 0 L2 hints,
        # bit 1 in-kernel zeroing (no memset node)
        ("f12", 128, 8, 1, 5, 0, 0, 0),
        ("f12x", 128, 8, 1, 6, 0, 0, 0),
        ("s0", 128, 8, 1, 1, 0, 0, 0),
        ("s1", 128, 8, 1, 2, 0, 0, 0),
        ("s2", 128, 8, 1, 3, 0, 0, 0),
        ("s3", 128, 8, 1, 4, 0, 0, 0),
        ("s3g2", 128, 8, 2, 4, 0, 0, 0),
        ("s3m1", 128, 8, 1, 4, 1, 0, 0),
        ("s3m2", 128, 8, 1, 4, 2, 0, 0),
        ("s3m2s7", 128, 7, 1, 4, 2, 0, 0),
        ("s3s6", 128, 6, 1, 4, 0, 0, 0),
        ("s3s5", 128, 5, 1, 4, 0, 0, 0),
        ("s3s4", 128, 4, 1, 4, 0, 0, 0),
        ("s3s6h", 128, 6, 1, 4, 0, 0, 1),
        ("s3s5h", 128, 5, 1, 4, 0, 0, 1),
        ("s3s6g2", 128, 6, 2, 4, 0, 0, 0),
        ("s1s6", 128, 6, 1, 2, 0, 0, 0),
        ("s1s5", 128, 5, 1, 2, 0, 0, 0),
        ("s4s6z", 128, 6, 1, 0, 0, 0, 2),
        ("s4s5z", 128, 5, 1, 0, 0, 0, 2),
        ("s4s6hz", 128, 6, 1, 0, 0, 0, 3),
        ("s4s5hz", 128, 5, 1, 0, 0, 0, 3),
        ("s4s5hzi", 128, 5, 1, 0, 0, 1, 3),
        ("s4s5zi", 128, 5, 1, 0, 0, 1, 2),
        ("s4s5zni", 128, 5, 1, 0, 0, 1, 2097154),
        ("s4s5zwi", 128, 5, 1, 0, 0, 1, 9437186),
        ("s4hz", 128, 8, 1, 0, 0, 0, 3),
        ("n256s5hz", 256, 5, 1, 0, 0, 0, 3),
        ("n256s4hz", 256, 4, 1, 0, 0, 0, 3),
        ("n256s5z", 256, 5, 1, 0, 0, 0, 2),
        ("n256s5m1hzmEH", 256, 5, 1, 0, 1, 0, 65603),
        ("n256s5m1hzm", 256, 5, 1, 0, 1, 0, 67),
        ("s3s7", 128, 7, 1, 4, 0, 0, 0),
        ("s3h", 128, 8, 1, 4, 0, 0, 1),
        ("s3m2h", 128, 8, 1, 4, 2, 0, 1),
        ("s4m2z", 128, 8, 1, 0, 2, 0, 3),
        ("r8m2", 128, 8, 1, 8, 2, 0, 0),
        ("s3m2i", 128, 8, 1, 4, 2, 1, 0),
        ("s3g2m1", 128, 8, 2, 4, 1, 0, 0),
        ("s4", 128, 8, 1, 0, 0, 0, 1),
        ("s4m1", 128, 8, 1, 0, 1, 0, 1),
        ("s4z", 128, 8, 1, 0, 0, 0, 3),
        ("s4m1z", 128, 8, 1, 0, 1, 0, 3),
        ("s4g2m1z", 128, 8, 2, 0, 1, 0, 3),
        ("ep", 128, 8, 1, 7, 0, 0, 0),
        ("epst", 128, 8, 1, 7, 0, 0, 4),
        ("ep0", 128, 8, 1, 7, 0, 0, 8),
        ("s4st", 128, 8, 1, 0, 0, 0, 5),
        ("s4zi", 128, 8, 1, 0, 0, 1, 3),
        ("r8", 128, 8, 1, 8, 0, 0, 0),
        ("r10", 128, 8, 1, 10, 0, 0, 0),
        ("r8m1", 128, 8, 1, 8, 1, 0, 0),
        ("s3i", 128, 8, 1, 4, 0, 1, 0),
        ("s3m1i", 128, 8, 1, 4, 1, 1, 0),
        ("s4m1zi", 128, 8, 1, 0, 1, 1, 3),
        # rs32: NT 32 / 16 (m exactly 32 / 16) and HALF (FL bit 4: 64 W rows, one math warpgroup; bit 2 = plain store, KS 1)
        ("t32s5z", 32, 5, 1, 0, 0, 0, 2),
        ("t32s5hz", 32, 5, 1, 0, 0, 0, 3),
        ("t32s8z", 32, 8, 1, 0, 0, 0, 2),
        ("t32s8hz", 32, 8, 1, 0, 0, 0, 3),
        ("t32s12z", 32, 12, 1, 0, 0, 0, 2),
        ("t32s12hz", 32, 12, 1, 0, 0, 0, 3),
        ("t32g2s8z", 32, 8, 2, 0, 0, 0, 2),
        ("t32m2s8z", 32, 8, 1, 0, 2, 0, 2),
        ("t32s8zi", 32, 8, 1, 0, 0, 1, 2),
        ("h32s8", 32, 8, 1, 0, 0, 0, 20),
        ("h32s8z", 32, 8, 1, 0, 0, 0, 18),
        ("h32s12", 32, 12, 1, 0, 0, 0, 20),
        ("h32s12z", 32, 12, 1, 0, 0, 0, 18),
        ("t16s5z", 16, 5, 1, 0, 0, 0, 2),
        ("t16s5hz", 16, 5, 1, 0, 0, 0, 3),
        ("t16s8z", 16, 8, 1, 0, 0, 0, 2),
        ("t16s8hz", 16, 8, 1, 0, 0, 0, 3),
        ("t16s12z", 16, 12, 1, 0, 0, 0, 2),
        ("t16s12hz", 16, 12, 1, 0, 0, 0, 3),
        ("t16g2s8z", 16, 8, 2, 0, 0, 0, 2),
        ("t16m2s8z", 16, 8, 1, 0, 2, 0, 2),
        ("t16s8zi", 16, 8, 1, 0, 0, 1, 2),
        ("h16s8", 16, 8, 1, 0, 0, 0, 20),
        ("h16s8z", 16, 8, 1, 0, 0, 0, 18),
        ("h16s12", 16, 12, 1, 0, 0, 0, 20),
        ("h16s12z", 16, 12, 1, 0, 0, 0, 18),
        ("t32s3z", 32, 3, 1, 0, 0, 0, 2),
        ("t32s3hz", 32, 3, 1, 0, 0, 0, 3),
        ("t32s4z", 32, 4, 1, 0, 0, 0, 2),
        ("t32s4hz", 32, 4, 1, 0, 0, 0, 3),
        ("t32s5hzi", 32, 5, 1, 0, 0, 1, 3),
        ("t16s3z", 16, 3, 1, 0, 0, 0, 2),
        ("t16s3hz", 16, 3, 1, 0, 0, 0, 3),
        ("t16s4z", 16, 4, 1, 0, 0, 0, 2),
        ("t16s4hz", 16, 4, 1, 0, 0, 0, 3),
        ("t16s5hzi", 16, 5, 1, 0, 0, 1, 3),
        ("t32s5hze", 32, 5, 1, 0, 0, 0, 35),
        ("t32s5ze", 32, 5, 1, 0, 0, 0, 34),
        ("t32s6hze", 32, 6, 1, 0, 0, 0, 35),
        ("t32s6hz", 32, 6, 1, 0, 0, 0, 3),
        ("t32s5hzei", 32, 5, 1, 0, 0, 1, 35),
        ("t32s6hzei", 32, 6, 1, 0, 0, 1, 35),
        ("t16s5hze", 16, 5, 1, 0, 0, 0, 35),
        ("t16s5ze", 16, 5, 1, 0, 0, 0, 34),
        ("t16s6hze", 16, 6, 1, 0, 0, 0, 35),
        ("t16s6hz", 16, 6, 1, 0, 0, 0, 3),
        ("t16s5hzei", 16, 5, 1, 0, 0, 1, 35),
        ("t16s6hzei", 16, 6, 1, 0, 0, 1, 35),
        ("h128s8e", 128, 8, 1, 0, 0, 0, 52),
        ("h128s5e", 128, 5, 1, 0, 0, 0, 52),
        ("h32s4e", 32, 4, 1, 0, 0, 0, 52),
        ("h32s5e", 32, 5, 1, 0, 0, 0, 52),
        ("s4s5hze", 128, 5, 1, 0, 0, 0, 35),
        ("s4s5hzem", 128, 5, 1, 0, 0, 0, 99),
        ("s4s5hzer", 128, 5, 1, 0, 0, 0, 163),
        ("s4s5hzemr", 128, 5, 1, 0, 0, 0, 227),
        ("s4s5hzen", 128, 5, 1, 0, 0, 0, 547),
        ("s4s5hzem2", 128, 5, 1, 0, 2, 0, 35),
        ("s4s5hzemf", 128, 5, 1, 0, 0, 0, 16483),
        ("s4s5hzemi", 128, 5, 1, 0, 0, 1, 99),
        ("s4s5hzeri", 128, 5, 1, 0, 0, 1, 163),
        ("s4s5hzemfi", 128, 5, 1, 0, 0, 1, 16483),
        # bit 11 (2048) patch-table walker
        ("s4s5hzew", 128, 5, 1, 0, 0, 0, 32803),
        ("s4s5hzewi", 128, 5, 1, 0, 0, 1, 32803),
        ("s4s5hzemw", 128, 5, 1, 0, 0, 0, 32867),
        ("s4s5hzemwi", 128, 5, 1, 0, 0, 1, 32867),
        ("s4s5hzeni", 128, 5, 1, 0, 0, 1, 547),
        ("h128s5", 128, 5, 1, 0, 0, 0, 20),
        ("h128s5z", 128, 5, 1, 0, 0, 0, 18),
        ("h128s8", 128, 8, 1, 0, 0, 0, 20),
        ("h128s8z", 128, 8, 1, 0, 0, 0, 18),
        ("s4s5zw", 128, 5, 1, 0, 0, 0, 9437186),
        ("s4s5hzw", 128, 5, 1, 0, 0, 0, 9437187),
        ("s4s5zn", 128, 5, 1, 0, 0, 0, 2097154),
        ("s4s5hzn", 128, 5, 1, 0, 0, 0, 2097155),
        ("s4s5zEH", 128, 5, 1, 0, 0, 0, 65538),
        ("s4s5hzEH", 128, 5, 1, 0, 0, 0, 65539),
        # NT=32 cursor twins (bit 20 + bit 23, EARLY bit 5 dropped as s4s5hzw relates
        # to s4s5hze) and the no-escape floor twins (bit 21, NOT exact: attribution only)
        ("t32s5hzw", 32, 5, 1, 0, 0, 0, 9437187),
        ("t32s6hzw", 32, 6, 1, 0, 0, 0, 9437187),
        ("t32s5zw", 32, 5, 1, 0, 0, 0, 9437186),
        ("h32s5hzw", 32, 5, 1, 0, 0, 0, 9437204),
        ("h32s4hzw", 32, 4, 1, 0, 0, 0, 9437204),
        ("t32s5hzwE", 32, 5, 1, 0, 0, 0, 9437219),
        ("t32s6hzwE", 32, 6, 1, 0, 0, 0, 9437219),
        ("t32s5hzn", 32, 5, 1, 0, 0, 0, 2097155),
        ("t32s6hzn", 32, 6, 1, 0, 0, 0, 2097155),
        ("t32s5zn", 32, 5, 1, 0, 0, 0, 2097154),
        ("h32s5hzn", 32, 5, 1, 0, 0, 0, 2097156),
        ("h32s4hzn", 32, 4, 1, 0, 0, 0, 2097156),
        ("t32s5hzwi", 32, 5, 1, 0, 0, 1, 9437187),
        ("t32s5hzni", 32, 5, 1, 0, 0, 1, 2097155),
        # NT=16 cursor twins, same derivation as the NT=32 ones (FL = e-row FL - bit 5 | bit 20 | bit 23).
        # h16s5hzw2 is the 2-CTA/SM HALF twin of h16s5hze2 (FL 1075 -> 9438227). The n/i entries are lab-only (floor and INS
        # stage clocks): they are in no RQSBAKE list, so prepare never builds them.
        ("t16s4hzw", 16, 4, 1, 0, 0, 0, 9437187),
        ("t16s5hzw", 16, 5, 1, 0, 0, 0, 9437187),
        ("h16s5hzw2", 16, 5, 1, 0, 0, 0, 9438227),
        ("t16s4hzn", 16, 4, 1, 0, 0, 0, 2097155),
        ("t16s5hzn", 16, 5, 1, 0, 0, 0, 2097155),
        ("t16s5hzwi", 16, 5, 1, 0, 0, 1, 9437187),
        ("t16s5hzni", 16, 5, 1, 0, 0, 1, 2097155),
        ("s4s6hze", 128, 6, 1, 0, 0, 0, 35),
        ("s4s7hze", 128, 7, 1, 0, 0, 0, 35),
        ("s4s8hze", 128, 8, 1, 0, 0, 0, 35),
        ("s4s8hzeEH", 128, 8, 1, 0, 0, 0, 65571),
        ("s4s5hzeEH", 128, 5, 1, 0, 0, 0, 65571),
        ("s4s6hzeEH", 128, 6, 1, 0, 0, 0, 65571),
        ("s4s5hzem1", 128, 5, 1, 0, 1, 0, 35),
        ("s4s5hzem2", 128, 5, 1, 0, 2, 0, 35),
        ("h128s6e", 128, 6, 1, 0, 0, 0, 52),
        ("h128s10e", 128, 10, 1, 0, 0, 0, 52),
        ("h128s8eg2", 128, 8, 2, 0, 0, 0, 52),
        ("h128s10eg2", 128, 10, 2, 0, 0, 0, 52),
        ("h128s8em1", 128, 8, 1, 0, 1, 0, 52),
        ("s4s5hze2", 128, 5, 2, 0, 0, 0, 35),
        ("s4s6hze2", 128, 6, 2, 0, 0, 0, 35),
        ("s4s5hzei", 128, 5, 1, 0, 0, 1, 35),
        ("h128s8ei", 128, 8, 1, 0, 0, 1, 52),
        ("h32s5hze2", 32, 5, 1, 0, 0, 0, 1075),
        ("h32s5hze2i", 32, 5, 1, 0, 0, 1, 1075),
        ("h32s6hze2", 32, 6, 1, 0, 0, 0, 1075),
        ("h32s8hze2", 32, 8, 1, 0, 0, 0, 1075),
        ("h32s5ze2", 32, 5, 1, 0, 0, 0, 1074),
        ("h32s4hze2", 32, 4, 1, 0, 0, 0, 1075),
        ("h32s7hze2", 32, 7, 1, 0, 0, 0, 1075),
        ("f32s5hze2", 32, 5, 1, 0, 0, 0, 2083),
        ("f32s5hze2i", 32, 5, 1, 0, 0, 1, 2083),
        ("h16s5hze2", 16, 5, 1, 0, 0, 0, 1075),
        ("h16s5hze2i", 16, 5, 1, 0, 0, 1, 1075),
        ("h16s8hze2", 16, 8, 1, 0, 0, 0, 1075),
        ("h16s4hze2", 16, 4, 1, 0, 0, 0, 1075),
        ("h16s6hze2", 16, 6, 1, 0, 0, 0, 1075),
        ("h16s7hze2", 16, 7, 1, 0, 0, 0, 1075),
        ("f16s5hze2", 16, 5, 1, 0, 0, 0, 2083),
        ("d1s5hze", 128, 5, 1, 0, 0, 0, 4131),
        ("d2s5hze", 128, 5, 1, 0, 0, 0, 12323),
        ("d2s5hzw", 128, 5, 1, 0, 0, 0, 9449507),  # d2s5hze + FL bit 20|23 (cursor)
        # the same cursor + a SHORTER apply block (FL bit 25: patch the 8 hi-byte
        # registers, then build A with the 16 PRMTs) so the taken branch over the apply is cheap. W-resident m=128:
        # s4s5hzwP 15.58/10.88/15.62 vs s4s5hzw 15.84/11.01/15.94 (pooler/qkv/unpooler), exact, regs 168 / local 0.
        ("d2s5hzwP", 128, 5, 1, 0, 0, 0, 42999939),  # d2s5hzw + FL bit 25 (9449507 | 33554432)
        ("s4s5hzwP", 128, 5, 1, 0, 0, 0, 42991619),  # s4s5hzw + FL bit 25 (9437187 | 33554432)
        ("d1s5hzei", 128, 5, 1, 0, 0, 1, 4131),
        ("d2s5hzei", 128, 5, 1, 0, 0, 1, 12323),
        ("d0s5hzen", 128, 5, 1, 0, 0, 0, 4643),
        ("d0s5hzeni", 128, 5, 1, 0, 0, 1, 4643),
        ("t32d1s5hze", 32, 5, 1, 0, 0, 0, 4131),
        ("t32d2s5hze", 32, 5, 1, 0, 0, 0, 12323),
        ("t32d1s5hzei", 32, 5, 1, 0, 0, 1, 4131),
        ("m2s3m1hz", 128, 3, 2, 0, 1, 0, 3),
        ("m2s4m1hz", 128, 4, 2, 0, 1, 0, 3),
        ("m2s3m1h", 128, 3, 2, 0, 1, 0, 1),
        ("m2s4m1h", 128, 4, 2, 0, 1, 0, 1),
        ("m2s3m1hze", 128, 3, 2, 0, 1, 0, 35),
        ("m2s4m1hze", 128, 4, 2, 0, 1, 0, 35),
        ("m2s3m2hz", 128, 3, 2, 0, 2, 0, 3),
        ("m2s4m2hz", 128, 4, 2, 0, 2, 0, 3),
        ("m2s3m1z", 128, 3, 2, 0, 1, 0, 2),
        ("m2s4m1z", 128, 4, 2, 0, 1, 0, 2),
        ("m2s3m1zEH", 128, 3, 2, 0, 1, 0, 65538),
        ("m2s4m1zEH", 128, 4, 2, 0, 1, 0, 65538),
        ("m2s3m1hzEH", 128, 3, 2, 0, 1, 0, 65539),
        ("m2s4m1hzEH", 128, 4, 2, 0, 1, 0, 65539),
        ("g2s5z", 128, 5, 2, 0, 0, 0, 2),
        ("g2s5hz", 128, 5, 2, 0, 1, 0, 3),
        ("g2s5hze", 128, 5, 2, 0, 1, 0, 35),
        ("g2s6hz", 128, 6, 2, 0, 1, 0, 3),
        ("g2s4hz", 128, 4, 2, 0, 1, 0, 3),
        ("s4s6hzew1", 128, 6, 1, 0, 1, 0, 35),
        ("s4s7hzew1", 128, 7, 1, 0, 1, 0, 35),
        ("s4s8hzew1", 128, 8, 1, 0, 1, 0, 35),
        ("m2s3m1hzi", 128, 3, 2, 0, 1, 1, 3),
        ("m2s4m1hzi", 128, 4, 2, 0, 1, 1, 3),
        ("d2s5z", 128, 5, 1, 0, 0, 0, 12322),
        ("d2s4hze", 128, 4, 1, 0, 0, 0, 12323),
        ("d2s6hze", 128, 6, 1, 0, 0, 0, 12323),
        ("d2s5hzem2", 128, 5, 1, 0, 2, 0, 12323),
        ("d2s3hze", 128, 3, 1, 0, 0, 0, 12323),
        ("d2s4z", 128, 4, 1, 0, 0, 0, 12322),
        # rs32 (NT = m): ring depth 4 / 7 off the shipped t32/t16s5-6hze shape
        ("t32s4hze", 32, 4, 1, 0, 0, 0, 35),
        ("t32s7hze", 32, 7, 1, 0, 0, 0, 35),
        ("t16s4hze", 16, 4, 1, 0, 0, 0, 35),
        ("t16s7hze", 16, 7, 1, 0, 0, 0, 35),
    ]
    RQMS = {nm: 2 for nm in ("m2s3m1hz", "m2s4m1hz", "m2s3m1h", "m2s4m1h", "m2s3m1hze", "m2s4m1hze", "m2s3m2hz", "m2s4m2hz",
                            "m2s3m1z", "m2s4m1z", "m2s3m1zEH", "m2s4m1zEH", "m2s3m1hzEH", "m2s4m1hzEH",
                            "m2s3m1hzi", "m2s4m1hzi")}
    RQI = {v[0]: i for i, v in enumerate(RQV)}
    RQTOL = {1: 4e-3, 2: 5e-3, 3: 6e-3, 4: 6e-3, 5: 7e-3, 6: 7e-3, 7: 8e-3, 8: 8e-3, 9: 8e-3}  # benchmark 1e-2 (rows 5x)

    RQKS = {"pooler": 4, "unpooler": 2, "o": 2, "qkv_gate": 7}  # one-wave KS grid: N/128 x KS <= 132
    RQPROD = ["s4s5z", "s4s5hz", "s4s5hze", "d2s5hze", "n256s4hz", "n256s5hz", "n256s5z", "s4s5zEH", "s4s5hzEH"]
    RQI = {v[0]: i for i, v in enumerate(RQV)}
    RQCUR = {RQI[n] for n in ("s4s5zw", "s4s5hzw", "d2s5hzwP", "s4s5hzwP", "t32s5hzw", "t32s6hzw", "t32s5zw", "h32s5hzw", "h32s4hzw",
                              "t32s5hzwE", "t32s6hzwE", "t32s5hzwi",
                              "t16s4hzw", "t16s5hzw", "h16s5hzw2", "t16s5hzwi")}  # variants that need the FL bit 20 cursor array
    RQTOL = {1: 4e-3, 2: 5e-3, 3: 6e-3, 4: 6e-3, 5: 7e-3, 6: 7e-3, 7: 8e-3, 8: 8e-3, 9: 8e-3}  # benchmark 1e-2 (rows 5x)

    RQCU = __cuda__("streamk_rqcu.cu")


    def _rqwg(n):
        """rq_wgmma<n>: wgmma.mma_async m64n<n>k16 f32 += bf16 (A: 4 registers) x bf16 (B: smem descriptor), scale-d from a
    register (0 on the CTA's first k16: no accumulator zero-fill), scale-a = scale-b = 1, trans-b = 0"""
        nd = n // 2
        d = ", ".join("%%%d" % i for i in range(nd))
        a = ", ".join("%%%d" % (nd + i) for i in range(4))
        outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
        return (("template <> __device__ __forceinline__ void rq_wgmma<%d>(float (&D)[%d], const uint32_t* a, uint64_t desc, int scd) {\n"
                 "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
                 "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(scd));\n}\n")
                % (n, nd, nd + 5, n, d, a, nd + 4, outs))


    def _rqss(n):
        nd = n // 2
        d = ", ".join("%%%d" % i for i in range(nd))
        outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
        return (("template <> __device__ __forceinline__ void rq_wgss<%d>(float (&D)[%d], uint64_t da, uint64_t db, int scd) {\n"
                 "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n\"\n"
                 "    : %s\n    : \"l\"(da), \"l\"(db), \"r\"(scd));\n}\n")
                % (n, nd, nd + 2, n, d, nd, nd + 1, outs))


    def _rqsrc(vs):
        cases = "\n".join("    case %d: return rq_go<%s>(p, x, out, (int)KS, (int)info);" % (i, ", ".join(str(a) for a in v[1:8] + (RQMS.get(v[0], 1),)))
                          for i, v in enumerate(vs))
        return RQCU.replace("@@WGMMA@@", _rqwg(16) + _rqwg(32) + _rqwg(128) + _rqwg(256) + _rqss(128) + _rqss(256)).replace("@@CASES@@", cases)


    RQFLAGS = ["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr"]
    _rqmod = [None, None]
    _rqmod2 = [None, None]  # rs32: the m = 16 / 32 instances, a second extension built on its own thread


    def _rqext(vs=None, verbose=False, slot=None):
        vs = RQV if vs is None else vs
        src = _rqsrc(vs)
        slot = _rqmod if slot is None else slot
        if slot[0] is None or slot[1] != src:
            import os
            tag = ("rq128_%x" if slot is _rqmod else "rq32_%x") % (hash(src) & 0xffffffff)
            flags = RQFLAGS + (["-Xptxas=-v"] if verbose else [])
            slot[0] = load_inline(tag + ("v" if verbose else ""), cpp_sources="int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);",
                                    cuda_sources=src, functions=["rqrun"], extra_cuda_cflags=flags, verbose=verbose)
            slot[1] = src
        return slot[0]


    def _pack_nat(w):
        """bf16 [N, K] -> A fragments [N/16, K/64, 4 steps, 32 lanes, 8] in natural k order (wgmma B = x
    in natural k): step q, lane g*4+t, value v = (kh, half, e) holds W[16T + 8*half + g][64c + 16q + 8*kh + 2t + e]"""
        N, K = w.shape
        W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half, g, c, q, kh, t, e
        return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()


    def _rqpack(w):
        """12-bit pack re-blocked to 128-row blocks: [N/128][K/64][8 tiles][1536 B] (one 12,288 B bulk copy per stage),
    the decode table word, and the escape rows (tile, chunk, step, lane, value, xor); None if the pack is not exact"""
        N, K = w.shape
        c = _c12(_pack_nat(w), 1)
        if c is None or N % 128:
            return None
        data, tab, esc = c
        T, KC = data.shape[:2]
        lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
        return lay, tab, esc


    def _rqunpack(lay, tab, esc, N, K):
        """independent check of the re-blocking: unpack [NB][KC][8][1536] with _unc12 and undo _pack_nat -> bf16 [N, K]"""
        NB, KC = lay.shape[:2]
        data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
        v = _unc12(data, tab, esc, 1)  # [T, KC, 4, 32, 8] int32 bf16 bits
        W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)  # T, c, q, g, t, kh, half, e
        return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)  # int32 bf16 bits (compare with w.view(int16) & 0xffff)


    def _rqesc(esc6, NB, KC, KS, dev):
        """escape lists for grid (NB, KS): one fixed-stride slot of CAP entries per (ks, nb, warp) (CAP a multiple of 32,
    > the longest list, so every list ends in a sentinel), + 32 trailing sentinels; int32 [KS*NB*8*CAP + 32]"""
        G = KS * NB * 8
        if len(esc6):
            tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
            nb, wt = tile // 8, tile % 8
            s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
            ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
            u = ch - s0[ks]
            ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
            gw = (ks * NB + nb) * 8 + wt
            o = torch.argsort(gw * 65536 + u)
            gw, ent = gw[o], ent[o]
            rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
            cap = (int(torch.bincount(gw, minlength=G).max()) + 1 + 31) // 32 * 32
        else:
            cap = 32
        arr = torch.full((G * cap + 32,), 0xffffffff, dtype=torch.long, device=dev)
        if len(esc6):
            arr[gw * cap + rank] = ent
        arr = torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int()
        return arr.contiguous(), torch.zeros(1, dtype=torch.int32, device=dev)


    def _rqesc2(esc6, NB, KC, KS, dev, CAPT=4):
        """FL bit 20 lists: one fixed slot of CAPT entries per (ks, nb, warp, lane), sorted by stage, sentinel-terminated
    -> int32 [KS*NB*8*32*CAPT].  Returns (arr, max entries in any slot); arr is None if a slot overflows CAPT."""
        G = KS * NB * 8
        if not len(esc6):
            return torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.int32, device=dev), 0
        tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
        nb, wt = tile // 8, tile % 8
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
        ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
        u = ch - s0[ks]
        ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
        slot = ((ks * NB + nb) * 8 + wt) * 32 + ln
        o = torch.argsort(slot * 65536 + u)
        slot, ent = slot[o], ent[o]
        rank = torch.arange(len(slot), device=dev) - torch.searchsorted(slot, slot)
        mx = int(rank.max()) + 1
        if mx > CAPT:
            return None, mx
        arr = torch.full((G * 32 * CAPT,), 0xffffffff, dtype=torch.long, device=dev)
        arr[slot * CAPT + rank] = ent
        return torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int().contiguous(), mx


    RQKS = {"pooler": 4, "unpooler": 2, "o": 2, "qkv_gate": 7}  # one-wave KS grid: N/128 x KS <= 132
    RQPROD = ["s4s5hze", "d2s5hze", "d2s6hze", "d2s5hzem2", "s4s5hzeEH", "n256s5m1hzmEH", "n256s5m1hzm",
              "d2s5hzwP", "s4s5hzw"]
    RQKSV = {"h128s8e": 1, "h128s6e": 1, "h128s10e": 1, "h128s8em1": 1, "h128s8eg2": 1, "h128s10eg2": 1}  # variant -> KS
    RQBAKE_L = {  # per (layer, m) probe list (falls back to RQBAKE[m]); <= 6 timed paths per (layer, m)
        ("unpooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "s4s5hze", "d2s6hze"],
        ("pooler", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        ("qkv_gate", 128): ["d2s5hze", "d2s5hzwP", "s4s5hzw", "s4s5hzeEH", "d2s6hze"],
        # the n256 WM=1/MRG/EH entries at m=256 (r6 is the baseline; unpooler/o stay r6)
        ("qkv_gate", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
        ("pooler", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
    }
    RQBAKE = {128: [], 256: []}
    RQLAYERS = ("pooler", "qkv_gate", "unpooler", "o")
    RQ = True
    _rqthr = []


    def _rqstart(prod=None, slot=None, thr=None, tag="rq"):
        """compile the RS128 extension in a background thread while r6's phase runs (CPU-only work; never fatal)"""
        import threading
        prod = RQPROD if prod is None else prod
        thr = _rqthr if thr is None else thr

        def build():
            try:
                _rqext([RQV[RQI[nm]] for nm in prod], slot=slot)
                thr.append("ok")
                thr.append(time.time() - _T0)
            except Exception as error:
                thr.append("failed: %s" % (str(error).splitlines()[-1][:200] if str(error) else type(error).__name__))
        t = threading.Thread(target=build, daemon=True)
        t.start()
        print("PREP %s thread started at %.0fs" % (tag, time.time() - _T0), flush=True)
        return t


    def _rqcall(st, p, x, out):
        e2 = p.get("esc2")
        if e2 is None:
            e2 = st.get("rq_e2e")
            if e2 is None:
                e2 = st["rq_e2e"] = torch.empty(0, dtype=torch.int32, device=x.device)
        return p.get("mod", _rqmod[0]).rqrun(p["i"], p["lay"], x, out, p["esc"], e2, p["cnt"], p["KS"], p["tab"], 0, st["rq_e"], st["rq_sink"], 0)


    def _rqval(fn, w, m, n, gen, tol):
        """one-hot x8 rows bit-exact against W over every K column (the split-K bf16 reduce-add must keep them exact),
    random rows within the KS-dependent tolerance (benchmark: 1e-2, rows 5x)"""
        K = w.shape[1]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
        exact = w[:n].t()
        if K not in _EYE:
            _EYE[K] = torch.eye(K, dtype=torch.bfloat16, device=w.device) * 8.0
        eye = _EYE[K]
        for start in range(0, K, m):
            x = eye[start:start + m]
            if x.shape[0] < m:
                x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=x.device)))
            if fn(x.contiguous(), out) != 0:
                return False, "launch error"
            got = out[: min(m, K - start)]
            want = exact[start:start + got.shape[0]] * 8.0
            if not torch.isfinite(got).all() or not torch.equal(got.float(), want.float()):
                return False, "one-hot mismatch at column %d" % start
        for dist in (1.0, 30.0):
            x = torch.randn((m, K), generator=gen, device=w.device)
            if dist != 1.0:
                x[:, torch.randperm(K, generator=gen, device=w.device)[: K // 100]] *= dist  # the task's 'outliers'
            x = x.to(torch.bfloat16)
            o = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
            fn(x, o)
            ref = x.double() @ w[:n].double().t()
            err = float((o.double() - ref).norm() / ref.norm())
            row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            if not (err <= tol and row <= 5 * tol and torch.isfinite(o).all()):
                return False, "random rel L2 %.2e row %.2e > %.0e" % (err, row, tol)
        return True, "ok %.1e" % err


    def _rqbake(st, thr):
        """RS128 bake: per (layer, m in 128 / 256) time r6's own path and each RS128 variant (graph replays from a cold L2,
    the benchmark's method), pick a variant only if it validates and beats r6 by MARGIN; kernel_fn serves picks at m exactly"""
        st["rq"] = {}
        if time.time() - _T0 > PREPARE_BUDGET_S - 45:
            print("BAKE rq skipped-budget %.0fs" % (time.time() - _T0), flush=True)
            return
        thr.join(timeout=max(1.0, PREPARE_BUDGET_S - 45 - (time.time() - _T0)))
        if not _rqthr or _rqthr[0] != "ok" or _rqmod[0] is None:
            print("BAKE rq build %s" % (_rqthr[0] if _rqthr else "not finished"), flush=True)
            return
        print("PREP rq thread ready at %.0fs (bake starts at %.0fs)" % (_rqthr[1], time.time() - _T0), flush=True)
        names = [v[0] for v in [RQV[RQI[nm]] for nm in RQPROD]]
        dev = st["pooler_raw"].device
        st["rq_e"] = torch.empty(0, dtype=torch.int64, device=dev)
        st["rq_sink"] = torch.zeros(4, dtype=torch.int32, device=dev)
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(12828)
        for layer in RQLAYERS:
            if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                print("BAKE rq %s skipped-budget" % layer, flush=True)
                continue
            ms_ = tuple(m_ for m_ in (128, 256) if _NC(layer, m_))
            if not ms_:
                continue
            w = st[layer + "_raw"]
            n, K = w.shape
            xs, r6a = {}, {}
            for m in ms_:
                xs[m] = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
                o6 = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                r6a[m] = _time_us(lambda xx, oo: _r6(xx, layer, n, st), xs[m], o6, flush, sink)
            pk = _rqpack(w)
            if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
                print("BAKE rq %s pack not exact" % layer, flush=True)
                continue
            lay, tab, esc6 = pk
            st.setdefault("rq_pk", {})[layer] = pk  # rs32: the small-m phase (after gx) reuses the pack
            KS = RQKS[layer]
            _bl = RQBAKE_L.get((layer, 128), RQBAKE[128]) + RQBAKE_L.get((layer, 256), RQBAKE[256])
            ESC = {}
            for _kk in sorted(set([KS] + [RQKSV.get(nm, KS) for nm in _bl])):
                ESC[_kk] = _rqesc(esc6, lay.shape[0], lay.shape[1], _kk, dev)[0]
            esc = ESC[KS]
            esc2, mx2 = _rqesc2(esc6, lay.shape[0], lay.shape[1], KS, dev)
            print("BAKE rq %s esc2 slots max=%d %s" % (layer, mx2, "ok" if esc2 is not None else "OVERFLOW"), flush=True)
            cnt = torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev)  # 2 per CTA, GX up to 4*NB (HALF x KS)
            # discarded warm-up: the first path timed after the pack reads ~3 us high (r6 31.3 vs 27.4, and an rq path
            # 27.8 vs its 24-25), which would bias whichever path is timed first
            _time_us(lambda xx, oo: _r6(xx, layer, n, st), xs[128], torch.empty((128, n), dtype=torch.bfloat16, device=dev), flush, sink, it=7)
            keep = False
            for m in ms_:
                x = xs[m]
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                r6 = lambda xx, oo: _r6(xx, layer, n, st)
                r6med = _time_us(r6, x, out, flush, sink)
                print("BAKE %s m=%d r6 %.2f at %.0fs" % (layer, m, r6med, time.time() - _T0), flush=True)

                def _try(nm):
                    """validate one rq128 path; returns (fn, p) or None"""
                    ksv = RQKSV.get(nm, KS)
                    if esc2 is None and RQI[nm] in RQCUR:
                        return None
                    p = {"i": names.index(nm), "lay": lay, "esc": ESC[ksv], "esc2": esc2, "cnt": cnt, "KS": ksv, "tab": tab, "n": n}
                    fn = lambda xx, oo, p=p: _rqcall(st, p, xx, oo)
                    try:
                        r = fn(x, out)
                        torch.cuda.synchronize()
                        if r != 0:
                            print("BAKE %s m=%d rq:%s ERROR launch %d" % (layer, m, nm, r), flush=True)
                            return None
                        ok, msg = _rqval(fn, w, m, n, gen, RQTOL[KS])
                        torch.cuda.synchronize()
                        if int(cnt.abs().sum()):  # the zero handshake must leave its counters at 0 after every launch
                            cnt.zero_()
                            ok, msg = False, "counters not reset"
                        if not ok:
                            print("BAKE %s m=%d rq:%s INVALID %s" % (layer, m, nm, msg), flush=True)
                            return None
                    except Exception as error:
                        print("BAKE %s m=%d rq:%s ERROR %s" % (layer, m, nm, str(error).splitlines()[-1][:160]), flush=True)
                        return None
                    return fn, p

                # ONE reference per group. The proven pick (first entry of the list) is the reference
                # only if this GPU does not read it slower than r6 (window = half the group's margin); otherwise r6 is the
                # reference and the proven pick is an ordinary candidate. Every candidate's paired ratio is taken against
                # that same reference; the pick is the best ratio if it beats the margin, else the reference.
                names_m = RQBAKE_L.get((layer, m), RQBAKE[m])
                pv = names_m[0] if names_m else None
                refn, refp, refmed, reffn = "r6", None, r6med, r6
                cands, done = [], set()
                if pv is not None:
                    done.add(pv)
                    got = _try(pv)
                    if got is not None:
                        fn, p = got
                        us = _time_us(fn, x, out, flush, sink, base=r6)
                        print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs r6 %.2f) at %.0fs"
                              % (layer, m, pv, us, _LAST[0][0], r6med, time.time() - _T0), flush=True)
                        rn, rm, isp = _refsel("rq:" + pv, us, "r6", r6med, RWIN128)
                        if isp:
                            refn, refp, refmed, reffn = rn, p, rm, fn
                        else:
                            cands.append(("rq:" + pv, p, _LAST[0][0]))
                for nm in names_m:
                    if nm in done:
                        continue
                    if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                        print("BAKE %s m=%d rq:%s skipped-budget %.0fs" % (layer, m, nm, time.time() - _T0), flush=True)
                        continue
                    got = _try(nm)
                    if got is None:
                        continue
                    fn, p = got
                    us = _time_us(fn, x, out, flush, sink, base=reffn)
                    print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs %s %.2f) at %.0fs"
                          % (layer, m, nm, us, _LAST[0][0], refn, refmed, time.time() - _T0), flush=True)
                    cands.append(("rq:" + nm, p, _LAST[0][0]))
                best = _pickrule((refn, refp, refmed), cands, MARGIN_RQ128)
                if best[1] is not None:
                    st["rq"][(layer, m)] = best[1]
                    keep = True
                print("PICK %s m=%d %s %.2f (ref %s %.2f)" % (layer, m, best[0], best[2], refn, refmed), flush=True)
                _fo = RQPICK128.get((layer, m))
                if _fo is not None:
                    for _t, _p, _r in cands:
                        if _t == "rq:" + _fo:
                            st["rq"][(layer, m)] = _p
                            print("PICK %s m=%d rq:%s (forced: cursor + short apply, resident -0.26/-0.13/-0.32 us, bake ratio %.4f)"
                                  % (layer, m, _fo, _r), flush=True)
                            break
            if not keep:
                del lay, esc, cnt
        torch.cuda.synchronize()


    # ---------------------------------------------------------------- rs32: rq at m exactly 32 / 16 (NT = m)
    # <= 4 timed paths per (layer, m): (variant, KS). Baked after gx against the incumbent kernel_fn would serve without rq
    # (st["sel"] if gx picked the shape, else r6's gv / cuBLAS choice); F1 baseline rule: one discarded warm-up timing, the
    # incumbent timed before and after the variants, the lower kept, both printed.
    RQS = True
    RQSBAKE = {
        # + the 2-CTA HALF shape at the proven K-split (7); no new build (name already in RQPROD2)
        # the NT=32 cursor twins first (forced pick t32s6hzw-k7; the rest are the fallback set if the
        # forced variant fails validation in this bake).
        ("qkv_gate", 32): [("t32s6hzw", 7), ("t32s5hzw", 7), ("t32s6hzwE", 7), ("t32s5hzwE", 7),
                           ("t32s6hze", 7), ("t32s5hze", 7)],
        # the NT=16 cursor twins appended as EXTRA entries at the proven K-split 7 (the forced pick is
        # t16s5hzw-k7; these are its fallback set and the bake's own record of the twins).
        ("qkv_gate", 16): [("t16s4hze", 7), ("t16s5hze", 7), ("t16s6hze", 7), ("h16s5hze2", 7), ("h16s5hze2", 6), ("h16s5hze2", 8),
                           ("t16s4hzw", 7), ("t16s5hzw", 7), ("h16s5hzw2", 7)],
        ("o", 32): [("t32s6hzw", 2), ("t32s5hzw", 2), ("t32s6hze", 2), ("t32s5hze", 2), ("h32s5hze2", 1), ("h32s5hze2", 2)],
        ("o", 16): [("t16s5hze", 2), ("t16s6hze", 2), ("t16s5ze", 2), ("h16s5hze2", 2)],
        ("o", 128): [],
        # cursor twins first (forced pick t32s6hzw-k4) + the fallback set.
        ("pooler", 32): [("t32s6hzw", 4), ("t32s5hzw", 4), ("t32s6hzwE", 4), ("t32s5hzwE", 4),
                         ("t32s6hze", 4), ("t32s5hze", 4)],
        ("pooler", 16): [("t16s5hze", 4), ("t16s4hze", 4), ("t16s6hze", 4), ("t16s7hze", 4), ("h16s5hze2", 3), ("h16s5hze2", 5)],
        # cursor twins first (forced pick t32s6hzw-k2) + the fallback set.
        ("unpooler", 32): [("t32s6hzw", 2), ("t32s5hzw", 2), ("t32s6hzwE", 2), ("t32s5hzwE", 2),
                           ("t32s6hze", 2), ("t32s5hze", 2)],
        ("unpooler", 16): [("t16s5hze", 2), ("t16s4hze", 2), ("t16s6hze", 2), ("t16s7hze", 2), ("h16s5hze2", 1), ("h16s5hze2", 3)],
    }
    RQCAP = {("o", 32): 6, ("pooler", 32): 6, ("qkv_gate", 32): 6, ("unpooler", 32): 6}  # timed candidates per (layer, m)
    RQPROD2 = list(dict.fromkeys(nm for ky, ps in RQSBAKE.items() if _NC(*ky) for nm, _ in ps))
    RQPROD = [nm for nm in RQPROD if not nm.startswith("n256") or _NCR("qkv_gate", (256,)) or _NCR("pooler", (256,)) or _NCR("unpooler", (256,))]
    _rqthr2 = []


    def _rqsbake(st, thr):
        if time.time() - _T0 > PREPARE_BUDGET_S - 30:
            print("BAKE rq32 skipped-budget %.0fs" % (time.time() - _T0), flush=True)
            return
        thr.join(timeout=max(1.0, PREPARE_BUDGET_S - 30 - (time.time() - _T0)))
        if not _rqthr2 or _rqthr2[0] != "ok" or _rqmod2[0] is None:
            print("BAKE rq32 build %s" % (_rqthr2[0] if _rqthr2 else "not finished"), flush=True)
            return
        print("PREP rq32 thread ready at %.0fs" % _rqthr2[1], flush=True)
        names = RQPROD2
        dev = st["pooler_raw"].device
        st.setdefault("rq_e", torch.empty(0, dtype=torch.int64, device=dev))
        st.setdefault("rq_sink", torch.zeros(4, dtype=torch.int32, device=dev))
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(3216)
        print("PREP rq32 bake starts at %.0fs" % (time.time() - _T0), flush=True)
        for layer in RQLAYERS:
            if not _NCR(layer, (32, 16, 128)):
                continue
            if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                print("BAKE rq32 %s skipped-budget" % layer, flush=True)
                continue
            w = st[layer + "_raw"]
            n, K = w.shape
            pk = st.get("rq_pk", {}).get(layer)
            if pk is None:
                pk = _rqpack(w)
                if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
                    print("BAKE rq32 %s pack not exact" % layer, flush=True)
                    continue
            lay, tab, esc6 = pk
            escs = {}
            cnt = torch.zeros(4 * lay.shape[0], dtype=torch.int32, device=dev)
            for m in (32, 16, 128):
                ps = RQSBAKE.get((layer, m), [])
                if not ps or not _NC(layer, m):
                    continue
                x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                sp = st["sel"].get((layer, _bk(m)))
                rp0 = st["rq"].get((layer, m))  # m = 128: an rq pick of the m=128 phase is the incumbent
                if rp0 is not None:
                    inc, iname = (lambda xx, oo, rp0=rp0: _rqcall(st, rp0, xx, oo)), "rq128"
                elif sp is not None and sp[4] == n:
                    inc, iname = (lambda xx, oo, sp=sp: _gvcall(st, sp, xx, oo)), "sel"
                else:
                    inc, iname = (lambda xx, oo: _r6(xx, layer, n, st)), "r6"
                _time_us(inc, x, out, flush, sink, it=5)  # discarded warm-up
                imed = _time_us(inc, x, out, flush, sink)
                print("BAKE %s m=%d %s %.2f at %.0fs" % (layer, m, iname, imed, time.time() - _T0), flush=True)

                def _try(nm, KS):
                    """validate one rq32 path; returns (fn, p, tag) or None"""
                    if KS not in escs:
                        escs[KS] = _rqesc(esc6, lay.shape[0], lay.shape[1], KS, dev)[0]
                    p = {"i": names.index(nm), "lay": lay, "esc": escs[KS], "cnt": cnt, "KS": KS, "tab": tab, "n": n, "mod": _rqmod2[0]}
                    if RQI[nm] in RQCUR:  # the FL bit 20 cursor array, keyed on the k-split
                        if ("e2", KS) not in escs:
                            _a2, _mx = _rqesc2(esc6, lay.shape[0], lay.shape[1], KS, dev)
                            escs[("e2", KS)] = _a2
                            print("BAKE rq32 %s esc2 ks=%d slots max=%d %s" % (layer, KS, _mx, "ok" if _a2 is not None else "OVERFLOW"), flush=True)
                        if escs[("e2", KS)] is None:
                            print("BAKE %s m=%d rq:%s esc2 OVERFLOW" % (layer, m, nm, KS), flush=True)
                            return None
                        p["esc2"] = escs[("e2", KS)]
                    fn = lambda xx, oo, p=p: _rqcall(st, p, xx, oo)
                    tag = "%s-k%d" % (nm, KS)
                    if nm.endswith("2"):
                        e0 = st["rq_e"]
                        if _rqmod2[0].rqrun(p["i"], e0, e0, e0, e0, e0, e0, 0, 0, 0, e0, e0, 4) < 2:
                            print("BAKE %s m=%d rq:%s SKIP occ<2" % (layer, m, tag), flush=True)
                            return None
                    try:
                        r = fn(x, out)
                        torch.cuda.synchronize()
                        if r != 0:
                            print("BAKE %s m=%d rq:%s ERROR launch %d" % (layer, m, tag, r), flush=True)
                            return None
                        ok, msg = _rqval(fn, w, m, n, gen, RQTOL[KS])
                        torch.cuda.synchronize()
                        if int(cnt.abs().sum()):
                            cnt.zero_()
                            ok, msg = False, "counters not reset"
                        if not ok:
                            print("BAKE %s m=%d rq:%s INVALID %s" % (layer, m, tag, msg), flush=True)
                            return None
                    except Exception as error:
                        print("BAKE %s m=%d rq:%s ERROR %s" % (layer, m, tag, str(error).splitlines()[-1][:160]), flush=True)
                        return None
                    return fn, p, tag

                # one reference per group, same rule as _rqbake: the proven pick (PROVEN_RQ32) when
                # this GPU does not read it slower than the incumbent (window = half the margin), else the incumbent; every
                # candidate's paired ratio is taken against that same reference and the pick is the best ratio if it beats
                # the margin, else the reference.
                _fo = RQPICK32.get((layer, m))
                _fmsg = "NT=32 cursor, resident -0.70..-1.38 us"
                if _fo is None:
                    _fo = RQPICK16.get((layer, m))
                    _fmsg = "NT=16 cursor, resident -0.66 us"
                if _fo is None:  # o m=32 HALF twin
                    _fo = RQPICK32B.get((layer, m))
                    _fmsg = "resident -0.64 us"
                if _fo is not None:
                    _fnm, _fks = _fo.rsplit("-k", 1)
                    got = _try(_fnm, int(_fks))
                    if got is not None:
                        fn, p, tag = got
                        us = _time_us(fn, x, out, flush, sink, base=inc)
                        st["rq"][(layer, m)] = p
                        print("PICK %s m=%d rq:%s (forced: %s, bake %.2f vs %s %.2f)"
                              % (layer, m, tag, _fmsg, us, iname, imed), flush=True)
                        continue
                    print("BAKE %s m=%d rq:%s forced variant INVALID -> sweep" % (layer, m, _fo), flush=True)
                pv = PROVEN_RQ32.get((layer, m))
                refn, refp, refmed, reffn = iname, None, imed, inc
                cands, done = [], set()
                if pv is not None:
                    for nm, KS in ps[:RQCAP.get((layer, m), 6)]:
                        if "%s-k%d" % (nm, KS) != pv:
                            continue
                        done.add(pv)
                        got = _try(nm, KS)
                        if got is not None:
                            fn, p, tag = got
                            us = _time_us(fn, x, out, flush, sink, base=inc)
                            print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs %s %.2f) at %.0fs"
                                  % (layer, m, tag, us, _LAST[0][0], iname, imed, time.time() - _T0), flush=True)
                            rn, rm, isp = _refsel("rq:" + tag, us, iname, imed, RWIN32)
                            if isp:
                                refn, refp, refmed, reffn = rn, p, rm, fn
                            else:
                                cands.append(("rq:" + tag, p, _LAST[0][0]))
                        break
                for nm, KS in ps[:RQCAP.get((layer, m), 6)]:
                    tag = "%s-k%d" % (nm, KS)
                    if tag in done:
                        continue
                    if time.time() - _T0 > PREPARE_BUDGET_S - 30:
                        print("BAKE %s m=%d rq:%s skipped-budget %.0fs" % (layer, m, tag, time.time() - _T0), flush=True)
                        continue
                    got = _try(nm, KS)
                    if got is None:
                        continue
                    fn, p, tag = got
                    us = _time_us(fn, x, out, flush, sink, base=reffn)
                    print("BAKE %s m=%d rq:%s %.2f (ratio %.4f vs %s %.2f) at %.0fs"
                          % (layer, m, tag, us, _LAST[0][0], refn, refmed, time.time() - _T0), flush=True)
                    cands.append(("rq:" + tag, p, _LAST[0][0]))
                best = _pickrule((refn, refp, refmed), cands, MARGIN_RQ32)
                if best[1] is not None:
                    st["rq"][(layer, m)] = best[1]
                print("PICK %s m=%d %s %.2f (ref %s %.2f)" % (layer, m, best[0], best[2], refn, refmed), flush=True)

    return {'prepare': prepare, 'kernel_fn': kernel_fn, 'rqcall': _rqcall}

