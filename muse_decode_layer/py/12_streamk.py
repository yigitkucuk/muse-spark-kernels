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
    for _nm, _ly, _m, _g, _v in (("p", "pooler", 16, (1, 2, 8, 16, 1, 1, 0), 1), ("p", "pooler", 32, (1, 1, 8, 16, 1, 1, 2), 1),
                                 ("u", "unpooler", 16, (1, 1, 4, 16, 1, 1, 2), 1), ("u", "unpooler", 32, (1, 1, 4, 16, 1, 1, 2), 0),
                                 ("q", "qkv_gate", 16, (1, 1, 4, 8, 4, 1, 2, 1), 1), ("q", "qkv_gate", 32, (1, 1, 4, 16, 2, 1, 2, 1), 1)):
        for _sfx, _b in ():
            GXP.append(("gc-%s%s" % (_nm, _sfx), _ly, (_m,), _g, _v | _b))
    _OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
    _SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))
    def _c12(fr, dv=1):
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
    PREPARE_BUDGET_S = 280.0
    MARGIN_RQ128 = 0.98
    RQPICK128 = {("pooler", 128): "d2s5hzwP", ("qkv_gate", 128): "d2s5hzwP", ("unpooler", 128): "d2s5hzwP"}
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
        if base is None:
            g = _capture(lambda: fn(x, out))
            ts = _replay({"a": g}, it)["a"]
            ts.sort()
            return ts[len(ts) // 2]
        ratio, bmed, cmed, ov = _paired(lambda: fn(x, out), lambda: base(x, out), max(2, it // 2))
        _LAST[0] = (ratio, bmed, cmed, ov)
        return bmed / ratio
    RWIN128 = 1.01   # a proven pick is a group's reference only if this GPU does not read it slower than
    def _refsel(prov_name, prov_med, base_name, base_med, window):
        if prov_name is not None and prov_med is not None and prov_med <= base_med * window:
            return prov_name, prov_med, True
        return base_name, base_med, False
    def _pickrule(ref, cands, margin):
        best = None
        for nm, plan, r in cands:
            if best is None or r > best[2]:
                best = (nm, plan, r)
        if best is not None and best[2] > 1.0 / margin:
            return best[0], best[1], ref[2] / best[2]
        return ref
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
    RTV.update({v + 20000: c[:5] + (c[5] | 16,) for v, c in list(RTV.items())})
    RWLAD = True
    if RWLAD:
        RWV.update({k + 1000 * md: RWV[k][:6] + (md, 1) for k in (101, 104) for md in (1, 2, 3)})
    RQV = [  # (name, NT, ST, GS, PRB, WM, INS, FL): ST k64 slots (W 12 KB + x NT*128 B each), GS k64 stages per consumer
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
        N, K = w.shape
        W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half, g, c, q, kh, t, e
        return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()
    def _rqpack(w):
        N, K = w.shape
        c = _c12(_pack_nat(w), 1)
        if c is None or N % 128:
            return None
        data, tab, esc = c
        T, KC = data.shape[:2]
        lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
        return lay, tab, esc
    def _rqunpack(lay, tab, esc, N, K):
        NB, KC = lay.shape[:2]
        data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
        v = _unc12(data, tab, esc, 1)  # [T, KC, 4, 32, 8] int32 bf16 bits
        W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)  # T, c, q, g, t, kh, half, e
        return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)  # int32 bf16 bits (compare with w.view(int16) & 0xffff)
    def _rqesc(esc6, NB, KC, KS, dev):
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
        ("qkv_gate", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
        ("pooler", 256): ["n256s5m1hzmEH", "n256s5m1hzm"],
    }
    RQBAKE = {128: [], 256: []}
    RQLAYERS = ("qkv_gate", "unpooler")
    _rqthr = []
    def _rqstart(prod=None, slot=None, thr=None, tag="rq"):
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
            w = st[layer + "_raw"]
            n, K = w.shape
            xs, r6a = {}, {}
            for m in (128, 256):
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
            _time_us(lambda xx, oo: _r6(xx, layer, n, st), xs[128], torch.empty((128, n), dtype=torch.bfloat16, device=dev), flush, sink, it=7)
            keep = False
            for m in (128, 256):
                x = xs[m]
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                r6 = lambda xx, oo: _r6(xx, layer, n, st)
                r6med = _time_us(r6, x, out, flush, sink)
                print("BAKE %s m=%d r6 %.2f at %.0fs" % (layer, m, r6med, time.time() - _T0), flush=True)

                def _try(nm):
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
    def _r6(x, layer, n, state):
        return x @ state[layer + "_raw"][:n].T


    return dict(locals())


import time as _rq_time
_STREAMK = _make_streamk()
_RQ_NAMES = {"qkv_gate": "qkv", "unpooler": "un"}
_rq_prior_prepare = prepare


def prepare(weights):
    t0 = _rq_time.time()
    thr = None
    try:
        thr = _STREAMK["_rqstart"]()  # the rs128 extension compiles on a CPU thread while the layer builds
    except Exception as ex:
        print(f"[rq] start failed: {ex!r}"[:200], flush=True)
    st = _rq_prior_prepare(weights)
    print(f"[rq] layer prepare done at {_rq_time.time() - t0:.0f}s", flush=True)
    st["streamk_rq"] = {}
    if thr is None:
        return st
    try:
        cs = {"gv": {}, "lt": {}, "pooler_raw": weights["pooler"], "qkv_gate_raw": weights["qkv_gate"],
              "unpooler_raw": weights["unpooler"]}
        _STREAMK["_flushext"]()
        _STREAMK["_flush_validate"]()
        _STREAMK["_fixed_cost"]()
        _STREAMK["_rqbake"](cs, thr)
        for (lay, m), p in cs.get("rq", {}).items():
            if (_RQ_NAMES[lay], m) in RQ_USE:
                st["streamk_rq"][(_RQ_NAMES[lay], m)] = p
        st["streamk_st"] = {"rq_e": cs["rq_e"], "rq_sink": cs["rq_sink"]}
        print(f"[rq] picks {sorted(st['streamk_rq'])} (baked {sorted(cs.get('rq', {}))}) at {_rq_time.time() - t0:.0f}s", flush=True)
    except Exception as ex:  # the rs128 paths can only be added; the layer's own paths stand on any failure
        st["streamk_rq"] = {}
        print(f"[rq] bake failed: {str(ex).splitlines()[-1][:200] if str(ex) else repr(ex)}", flush=True)
    torch.cuda.synchronize()
    return st


RQ_USE = {('un', 128), ('qkv', 256), ('qkv', 128)}


def _rq_gemm(state, name, x, T):
    p = state.get("streamk_rq", {}).get((name, T)) if state is not None else None
    if p is None:
        return None
    out = torch.empty((T, p["n"]), dtype=torch.bfloat16, device=x.device)
    if _STREAMK["_rqcall"](state["streamk_st"], p, x.contiguous(), out) != 0:
        return None
    return out


# ---------------------------------------------------------------- MFIN: paired in-prepare timing of the MoE kernel
MFAB_LAYER = True
MFAB_LAYER_SEEDS = 4
MFAB_LAYER_REPS = 24


