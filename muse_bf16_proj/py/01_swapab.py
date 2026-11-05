def _make_swapab():
    """Dense projections for Muse Spark Open: skinny swap-AB tensor-core GEMV/GEMM with
losslessly compressed BF16 weights.

Math
----
``out[m, n] = x[m, k] @ W[n, k]^T`` for the five projection matrices of one
model (``qkv_gate``, ``o``, ``pooler``, ``unpooler``, ``router``). The router's
FP32 weights run as a BF16 hi + lo pair concatenated along K with FP32
accumulation, as SGLang does above 8 tokens.

Lossless 12-bit weights (BF16 projections)
-------------------------------------------
Each BF16 weight is stored as its low byte (exponent bit 0 + 7 mantissa bits,
raw) and a 4-bit code for its high byte (sign + exponent bits 7..1, as an
offset into an 8-value window chosen per matrix, the window covering the most
weights). The kernel rebuilds the exact BF16 bits in registers (a few integer
ops per weight). The rare weights whose exponent falls outside the window
(tiny magnitudes) are exact-corrected: the kernel adds ``x[:, k] * (w -
w_decoded)`` for each of them (both factors exact in FP32), so every output
equals the product with the original weights up to FP32 accumulation order.
``prepare()`` rejects a packing whose escape share exceeds
``1 / MAX_ESCAPE_DENOMINATOR`` or whose correction is not exact in FP32, and
falls back to uncompressed fragments.

Why the kernels look the way they do
------------------------------------
Weights dominate: for m <= 64 every weight byte is streamed once from HBM
straight into registers, in mma.m16n8k16 fragment order (prepared once), and
multiplied against x (staged in shared memory) on BF16 tensor cores with FP32
accumulation. Split-K across warps (shared-memory reduction) and across CTAs
(FP32 partials + a self-resetting semaphore; the last CTA of a row block
reduces) keeps ~128 CTAs streaming on every shape.

Routes (chosen per layer and token count m by ``_static_choice`` / ``_autotune``)
---------------------------------------------------------------------------------
* ``sk2`` -- m <= 32, swapped operands, no cross-CTA reduction (the production
  skinny path on sm_90). ``skinny`` (TMA-streamed tensor-core GEMV) is the
  fallback / pre-sk2 path, kept for non-sm_90 and as the autotuner's
  alternative.
* ``cgemv`` / ``sgemv`` / ``rgemv`` -- m == 1 scalar GEMV specialists.
* ``wide`` / ``gemm`` / ``gq`` / ``gqx`` -- m 33..256 TMA + wgmma GEMMs
  (``wide`` is the BF16 route; the router uses ``wide``/``gemm``/``rk``).
* ``rk`` -- router-only split-K GEMM (order-dependent FP32 adds, inside the
  router tolerance).
* ``cutlass`` -- CUTLASS pingpong GEMM, pooler m == 256 only, built in the
  background during ``prepare()``; a failed build just drops the route.
* ``torch`` / ``torchT`` -- cuBLAS fallbacks (``F.linear`` / ``mm`` on a
  K-major copy). Any custom path that fails ``_selftest`` is disabled in
  favour of cuBLAS.
"""

    import importlib.util
    import os
    import sys
    import threading
    from typing import TypedDict

    import torch
    import torch.nn.functional as F

    from torch.utils.cpp_extension import load_inline

    # Debug only, from the environment: one random escape warp per CTA stalls this
    # many ns at a random stage; ring waits trap after 2 s.
    _SK2_DELAY = int(os.environ.get("SK2_DELAY_ESC", "0"))


    # ---------------------------------------------------------------------------
    # Shared constants (tuning numbers used by more than one helper)
    # ---------------------------------------------------------------------------
    # Lossless 12-bit packing: an exponent window of EXPONENT_WINDOW consecutive
    # exponent values, picked per matrix to cover the most weights.
    EXPONENT_WINDOW = 8
    EXPONENT_VALUES = 128  # distinct values of BF16 exponent bits 7..1
    # Packing is rejected when more than this share of weights escapes the window.
    MAX_ESCAPE_DENOMINATOR = 200
    # base4 packs one window base byte into each byte of a uint32 (see decode8).
    BASE4_REPLICATE = 0x01010101
    # Stage-count clamps for the TMA skinny kernel's pipeline.
    TMA_MIN_STAGES = 2
    TMA_MAX_STAGES = 8
    # sm_90 / sm_80 shared-memory budgets (bytes) for that pipeline.
    SM90_SMEM_BUDGET_BYTES = 112 * 1024
    SM80_SMEM_BUDGET_BYTES = 40 * 1024
    # Stage-count clamps for the sk2 kernel's ring.
    SK2_MIN_STAGES = 2
    SK2_MAX_STAGES = 16
    # Failure seeds keep selftest / autotune reproducible across runs.
    SELFTEST_SEED = 1234
    AUTOTUNE_SEED = 4321
    AUTOTUNE_REPS = 24
    # Autotune replays from a cold L2: this read sweep is larger than L2.
    L2_FLUSH_ROWS = 65_536  # 65_536 x 1024 float32 = 256 MiB
    # sk2 ring sizing: a config name ending in _rNNN caps the ring at NNN KiB,
    # otherwise the default applies; the ring also never exceeds the smem budget
    # left after the rest of the kernel's shared-memory layout.
    SK2_DEFAULT_RING_CAP_KB = 200
    SK2_RING_SMEM_BUDGET_BYTES = 232_448


    class Weights(TypedDict):
        qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
        o: torch.Tensor         # bfloat16 [8192, 1024]
        pooler: torch.Tensor    # bfloat16 [4096, 8192]
        unpooler: torch.Tensor  # bfloat16 [8192, 4096]
        router: torch.Tensor    # float32 [256, 8192]


    class Config(TypedDict):
        m: int
        n: int
        k: int
        layer: str


    class Inputs(TypedDict):
        x: torch.Tensor  # bfloat16 [m, k]
        config: Config


    CUDA_SRC = __cuda__("swapab.cu")

    MBAR_HELPERS = __cuda__("swapab_mbar_helpers.cu")

    WG_HELPERS = r"""
// ---------------------------------------------------------------- wgmma helpers (sm_90a)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }


// smem matrix descriptor: K-major, 128B swizzle, SBO = 1024 B (8 rows x 128 B)
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;                 // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;       // SBO
  d |= (uint64_t)1 << 62;                 // 128B swizzle
  return d;
}


"""


    def wgmma_rs_defs(bns):
        """Generate wgmma_bn (register-sourced A) overloads for m64nBNk16 bf16."""
        out = []
        for bn in bns:
            nacc = bn // 2
            regs = ", ".join(f"%{i}" for i in range(nacc))
            a0 = nacc
            cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
            pi = a0 + 5
            out.append(f"""
    template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
    wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
      asm volatile("{{\\n"
          ".reg .pred p;\\n"
          "setp.ne.b32 p, %{pi}, 0;\\n"
          "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
          "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
          "}}\\n"
          : {cons}
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
    }}""")
        return "\n".join(out)


    TMA_SRC = __cuda__("swapab_tma.cu")

    MAP_SRC = __cuda__("swapab_map.cu")

    GEMM_SRC = __cuda__("swapab_gemm.cu")


    SK2_SRC = __cuda__("swapab_sk2.cu")


    WIDE_SRC = __cuda__("swapab_wide.cu")


    def wgmma_ss_defs(bns):
        out = []
        for bn in bns:
            nacc = bn // 2
            regs = ", ".join(f"%{i}" for i in range(nacc))
            cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
            a, b, p = nacc, nacc + 1, nacc + 2
            out.append(f"""
    template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
    wgmma_ss(float* d, uint64_t da, uint64_t db) {{
      asm volatile("{{\\n"
          ".reg .pred p;\\n"
          "setp.ne.b32 p, %{p}, 0;\\n"
          "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
          "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
          "}}\\n"
          : {cons}
          : "l"(da), "l"(db), "r"(1));
    }}""")
        return "\n".join(out)


    CPP_SRC = """
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void rgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M);
void sgemv(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t N);
void cgemv(torch::Tensor L, torch::Tensor C, torch::Tensor x, torch::Tensor etab, int64_t pmax, torch::Tensor out, int64_t base4, int64_t rows, int64_t nth);
void gqx(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
         int64_t cl, int64_t nstage);
void rk(torch::Tensor Wc, torch::Tensor x, torch::Tensor out, int64_t S, torch::Tensor zc);
"""

    # Weight rows per layer (the full matrices; some routes run on a prefix, e.g.
    # the unpooler slice at m <= 8).
    WEIGHT_ROWS = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
    COMPRESSED_LAYERS = {"qkv_gate", "o", "pooler", "unpooler"}
    # At small m the unpooler runs on its first rows only (the slice SGLang scores).
    UNPOOLER_SLICE_ROWS = 1024

    _ext = None


    # ---------------------------------------------------------------------------
    # Skinny (TMA) route config
    # ---------------------------------------------------------------------------
    TMA_CONFIGS = {  # (layer, mt) -> (splits, RG, WS, KS); x chunk M*Kc*2 small, ~128 CTAs; mt 0 = m == 1
        ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
        ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
        ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
        ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
        ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
    }
    USE_WGM = True   # m >= 9 compressed configs with RG % 4 == 0 use RS-wgmma (sm_90)
    WGM_SM80_TEST = False  # local check of the wgmma path's data layout (mma.sync emulation on sm_80)


    def _tma_nstage(nw, comp, sm90, ks, hl=False):
        """Pipeline stages for a skinny config: as many as fit the smem budget."""
        chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
        budget = SM90_SMEM_BUDGET_BYTES if sm90 else SM80_SMEM_BUDGET_BYTES
        return max(TMA_MIN_STAGES, min(TMA_MAX_STAGES, budget // (nw * chunk)))


    # alternative skinny configs (more splits -> fewer x bytes per CTA); the prepare()-time autotuner times them
    # against the default config and cuBLAS on every scored shape
    TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
               ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
               ("o", 32): (4, 16, 1, 4)}


    def _tma_cfg_items():
        items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(TMA_CONFIGS.items())]
        items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(TMA_ALT.items())]
        return items


    def _tma_srcs(sm90):
        lines = {}
        for key, (splits, rg, wsn, ks) in _tma_cfg_items():
            if key[0] == "alt":
                layer, mtk = key[1], _mt_key(key[2])
            else:
                layer, mtk = key
            mt = max(mtk, 1)
            f32 = layer == "router"
            ot = "float" if f32 else "__nv_bfloat16"
            comp = 0 if f32 else 1
            hl = f32
            ns = _tma_nstage(rg * wsn, comp, sm90, ks, hl)
            if f32 and mt >= 4:
                ns = min(ns, 3)  # x chunk (32 x 1024) + ring must fit in shared memory
            n_cfg = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and mtk <= 1) else WEIGHT_ROWS[layer]
            wgm = bool((sm90 or WGM_SM80_TEST) and comp and USE_WGM and rg % 4 == 0 and mtk >= 2)
            line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                    f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                    f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
            lines.setdefault(line, []).append(key)
        return TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(lines)).replace("@@TMA_NAME@@", "skinny_tma_a")


    GEMM_CONFIGS = {  # (layer, bn) -> (nwg, splits, stages); grid = (N / (64 nwg), splits) <= 132 (co-resident)
        ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
        ("router", 64): (2, 32, 8),
        ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
        ("router", 128): (2, 32, 6),
        ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
        ("router", 256): (2, 32, 4),
    }


    GEMM_DIAG = False   # diagnostic: also time gemm90 variants (TMA-2D A vs contiguous pre-swizzled A) on m 65..256


    def _gemm_src(sm90):
        lines = []
        for (layer, bn), (nwg, splits, stages) in sorted(GEMM_CONFIGS.items()):
            f32 = layer == "router"
            ot = "float" if f32 else "__nv_bfloat16"
            for a_s in ((0, 1) if (not f32 and GEMM_DIAG) else (0,)):
                lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                             f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
        bns = sorted({bn for (_, bn) in GEMM_CONFIGS})
        return (GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", wgmma_ss_defs(bns))
                .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


    GQ_SRC = __cuda__("swapab_gq.cu")

    GQ_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): BM = 64*nwg rows per CTA, grid = N/BM * cl CTAs (cl-CTA clusters split K)
        # measured on H200 against cuBLAS (graph replay, cold L2): only the pooler shape wins (64 row blocks x 2 K halves)
        ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 8),
    }
    USE_GQ = True

    GQX_SRC = __cuda__("swapab_gqx.cu")

    GQX_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): grid = N / (64 nwg) * cl CTAs
        ("qkv_gate", 128): (1, 3, 8),   # m 33..128: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
    }
    USE_GQX = True
    TORCHT_LAYERS = ("unpooler", "o")   # layers that also get a K-major BF16 copy for the "torchT" (mm(x, W^T)) route


    def _gqx_src(sm90):
        lines = []
        for (layer, bn), (nwg, cl, ns) in sorted(GQX_CONFIGS.items()):
            lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                         f"{{ launch_gqx<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
        return GQX_SRC.replace("@@GQX_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


    def _gqx_bn(m):
        return 128 if m <= 128 else 256


    def _run_gq_family(x, layer, m, n, k, state, configs, bn_of_m, method):
        """Run a gq/gqx cluster GEMM: same call shape, different config table and
    entry point (gq = plain BF16 weights, gqx = big tiles)."""
        bn = bn_of_m(m)
        nwg, cl, ns = configs[(layer, bn)]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        getattr(_get_ext(), method)(state["raw"][layer], x, out, m, n, k, bn, nwg, cl, ns)
        return out


    def _run_gqx(x, layer, m, n, k, state):
        return _run_gq_family(x, layer, m, n, k, state, GQX_CONFIGS, _gqx_bn, "gqx")


    RGEMV_SRC = __cuda__("swapab_rgemv.cu")


    def _gq_src(sm90):
        lines = []
        for (layer, bn), (nwg, cl, ns) in sorted(GQ_CONFIGS.items()):
            lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                         f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
        return GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


    def _gq_bn(m):
        return 64 if m <= 64 else 128 if m <= 128 else 256


    def _run_gq(x, layer, m, n, k, state):
        return _run_gq_family(x, layer, m, n, k, state, GQ_CONFIGS, _gq_bn, "gq")


    WIDE_CONFIGS = {  # (layer, tok, nz) -> (nwg, splits, stages); grid = (N / (64 nwg), splits, nz) <= 132 (co-resident)
        ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
        ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
        ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
        # router: BF16 hi|lo tiles (plain), 64-row blocks
        ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
    }


    def _wide_key(m):
        return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


    def _wide_src(sm90):
        lines = []
        for (layer, tok, nz), (nwg, splits, stages) in sorted(WIDE_CONFIGS.items()):
            hl = layer == "router"
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                         f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                         f"M, N, K, splits); return; }}")
        return WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


    RK_SRC = __cuda__("swapab_rk.cu")


    def _rk_src(sm90):
        if sm90:
            return RK_SRC


    # ---------------------------------------------------------------------------
    # Extension build: everything above is assembled into one CUDA translation
    # ---------------------------------------------------------------------------
    def _get_ext():
        """Compile everything as one CUDA translation unit with load_inline.

    no_implicit_headers: the CUDA file includes only the ATen headers it needs
    (torch/types.h alone costs ~45 s of nvcc time).
    """
        nonlocal _ext
        if _ext is None:
            major, minor = torch.cuda.get_device_capability()
            sm90 = major >= 9
            arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
            skinny = CUDA_SRC.replace("@@MBAR@@", MBAR_HELPERS).replace("@@BODY@@", WG_HELPERS if sm90 else "")
            rs_defs = wgmma_rs_defs([8, 16, 32, 64, 128]) if sm90 else ""
            # Order matters only in that each route's helpers precede its kernel;
            # every route is self-contained apart from the shared helpers above.
            cuda_src = (skinny + rs_defs + _tma_srcs(sm90) + MAP_SRC + _gemm_src(sm90)
                        + _wide_src(sm90) + _sk2_src(sm90) + _gq_src(sm90) + _gqx_src(sm90)
                        + RGEMV_SRC + _rk_src(sm90))
            _ext = load_inline(
                name=f"okc_bf16proj_v141_{arch}" + (f"_dly{_SK2_DELAY}" if _SK2_DELAY else ""),
                cpp_sources="#include <torch/extension.h>\n" + CPP_SRC,
                cuda_sources=cuda_src,
                functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "gqx", "rgemv", "sgemv", "cgemv", "rk"],
                extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={_SK2_DELAY}"] if _SK2_DELAY else []),
                no_implicit_headers=True,
                verbose=False,
            )
        return _ext


    # ---------------------------------------------------------------------------------------------- CUTLASS route (m 129..256)
    # CUTLASS 3.x sm90 TMA warp-specialized PINGPONG GEMM, persistent tile scheduler, swap-AB (D^T = W x^T: A = W [N, K]
    # row-major, B = x [M, K] as a K x M column-major matrix, D column-major [N, M] = out [M, N] row-major), BF16 in/out,
    # FP32 accumulate, tile 128 (W rows) x 64 (tokens) x 64, cluster 1x1. Headers: CUTLASS bundled with flashinfer (in the
    # SGLang image). Sweep (pod/cutsweep.py, 54 configs x 8 shapes): it beats cuBLAS only for pooler m=256 (31.3 vs 34.5 us
    # raw); a missing header or a failed build just leaves the route out.
    CUTLASS_LAYERS = set() if _PRUNE else {"pooler"}   # layers that get the CUTLASS candidate (scored m=256 only)
    CUTLASS_BUILD_TIMEOUT = 170   # seconds prepare waits for the CUTLASS build (main build ~25 s; prepare's limit is 300 s)
    CUTLASS_SRC = __cuda__("swapab_cutlass.cpp")
    _cut_ext = None
    _cut_err = None


    def _build_cutlass():
        """Build the CUTLASS extension (sm_90 only); on any failure record it and leave the route out."""
        nonlocal _cut_ext, _cut_err
        try:
            major, minor = torch.cuda.get_device_capability()
            spec = importlib.util.find_spec("flashinfer")
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError("flashinfer (CUTLASS headers) not installed")
            root = os.path.join(list(spec.submodule_search_locations)[0], "data", "cutlass")
            inc = [os.path.join(root, "include"), os.path.join(root, "tools", "util", "include")]
            if not all(os.path.isdir(d) for d in inc):
                raise RuntimeError("CUTLASS headers missing under " + root)
            _cut_ext = load_inline(
                name=f"okc_bf16proj_v141_cutlass_{major}{minor}a",
                cpp_sources="#include <torch/extension.h>\nint64_t cut_ws(at::Tensor W, at::Tensor x, at::Tensor out);\n"
                            "void cut_gemm(at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws);",
                cuda_sources=CUTLASS_SRC, functions=["cut_ws", "cut_gemm"], extra_include_paths=inc,
                extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a", "-DNDEBUG"],
                verbose=False)
        except Exception as e:   # noqa: BLE001 - any failure only drops this optional route
            _cut_ext, _cut_err = None, f"{type(e).__name__}: {str(e)[:200]}"


    # ---------------------------------------------------------------------------
    # Weight preparation (host side, runs in prepare()): fragment orders and the
    # lossless 12-bit packing. The packing helpers share _exponent_window /
    # _escapes; the per-kernel stream/table builders stay separate because their
    # layouts and orderings differ (per-CTA streams vs k-ordered escape tables).
    # ---------------------------------------------------------------------------
    def _exponent_window(e71):
        """Best EXPONENT_WINDOW-wide window of BF16 exponent bits 7..1 for a matrix.

    Returns (base, offset): ``base`` is the window's lowest exponent, and
    ``offset`` is each weight's exponent clamped into [0, EXPONENT_WINDOW).
    The window is chosen to cover the most weights, so the fewest weights
    need exact correction as escapes.
    """
        hist = torch.bincount(e71.flatten(), minlength=EXPONENT_VALUES)
        win = torch.stack(
            [hist[b:b + EXPONENT_WINDOW].sum() for b in range(EXPONENT_VALUES - EXPONENT_WINDOW + 1)]
        )
        base = int(win.argmax())
        return base, (e71 - base).clamp(0, EXPONENT_WINDOW - 1)


    def _escapes(e71, base):
        """Weights whose exponent falls outside [base, base + EXPONENT_WINDOW)."""
        return (e71 < base) | (e71 > base + EXPONENT_WINDOW - 1)


    def pack_fragments(w: torch.Tensor) -> torch.Tensor:
        """W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order."""
        n, k = w.shape
        t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  # [rg, hr, g, s, hk, t, e]: row = hr*8+g, col = hk*8+2t+e
        t = t.permute(0, 3, 2, 5, 4, 1, 6)  # [rg, s, g, t, hk, hr, e]
        return t.contiguous().view(n // 16, k // 16, 32, 8)


    def pack_compressed(w: torch.Tensor):
        """Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it."""
        n, k = w.shape
        if n % 16 or k % 64:
            return None
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        hi = bits >> 8                  # sign | exponent bits 7..1
        e71 = hi & 0x7F
        base, o = _exponent_window(e71)
        code = ((hi >> 7) << 3) | o
        dec_hi = ((hi >> 7) << 7) | (base + o)
        dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
        esc = _escapes(e71, base)
        n_esc = int(esc.sum())
        if n_esc > n * k // MAX_ESCAPE_DENOMINATOR:
            return None
        rows, cols = esc.nonzero(as_tuple=True)
        wtrue = w[rows, cols].float()
        wdec = dec[rows, cols].float()
        delta = wtrue - wdec
        if not bool(((wdec + delta) == wtrue).all()):
            return None
        # fragment order for low bytes and codes: [rg, s, lane, 8]
        lo_f = pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
        cd_f = pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
        nrg, S = n // 16, k // 16
        lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   # [rg, u, ks, lane, 8]
        cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
        # parts 0,1: lane's low bytes for k-steps (0,1) and (2,3) -> [rg, u, 2, lane, 16]
        lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
        # part 2: word i (k-step i): byte b = code[8i + b] | code[8i + 4 + b] << 4
        cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           # [rg, u, ks, lane, 4]
        cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
        packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [u, rg, 3, lane, 16]
        order = torch.argsort(rows * k + cols)
        rows, cols, delta = rows[order], cols[order], delta[order]
        rg_idx = rows // 16
        poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
        poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
        return {
            "packed": packed,
            "base4": base * BASE4_REPLICATE,
            "poff": poff,
            "pk": cols.to(torch.int32).contiguous(),
            "pr": (rows % 16).to(torch.int32).contiguous(),
            "pd": delta.to(torch.float32).contiguous(),
            "n_esc": n_esc,
        }


    def build_patch_table(c, n_cfg, k, splits, rg):
        """Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits)."""
        dev = c["pk"].device
        rows_cta = rg * 16
        nrb = n_cfg // rows_cta
        nunits = k // 64
        poff = c["poff"].long()
        nrg_all = poff.numel() - 1
        rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
        rows = rgi * 16 + c["pr"].long()
        cols = c["pk"].long()
        keep = rows < n_cfg
        rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
        unit = cols // 64
        spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          # split owning this 64-k unit
        k_start = (spl * nunits // splits) * 64
        cta = spl * nrb + rows // rows_cta
        ncta = nrb * splits
        counts = torch.bincount(cta, minlength=ncta)
        pmax = int(counts.max().item()) + 1 if rows.numel() else 1
        pmax += pmax % 2
        order = torch.argsort(cta, stable=True)
        cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
        start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
        start[1:] = torch.cumsum(counts, 0)
        slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
        tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
        tab[:, 0, 0] = counts.to(torch.int32)
        tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
        tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
        return tab.contiguous(), pmax


    def build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
        """Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16."""
        dev = src.device
        nw = rg * wsn
        nrb = (n_cfg // 16) // rg
        ar = lambda n: torch.arange(n, device=dev)
        if wsn == 1:
            s_all = (k // 16) // ks
            rb = ar(nrb).view(-1, 1, 1, 1)
            st = ar(s_all).view(1, -1, 1, 1)
            w = ar(nw).view(1, 1, -1, 1)
            rg_idx = rb * rg + w
            if comp:
                u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
                return src[rg_idx, u_idx].contiguous()
            s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
            return src[rg_idx, s_idx].contiguous()
        kc = k // splits
        ns = kc // (16 * wsn)
        nst = ns // ks
        sp = ar(splits).view(-1, 1, 1, 1, 1)
        rb = ar(nrb).view(1, -1, 1, 1, 1)
        st = ar(nst).view(1, 1, -1, 1, 1)
        w = ar(nw).view(1, 1, 1, -1, 1)
        rg_idx = rb * rg + w % rg
        if comp:
            ku = ar(ks // 4).view(1, 1, 1, 1, -1)
            u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
            out = src[rg_idx, u_idx]
        else:
            j = ar(ks).view(1, 1, 1, 1, -1)
            s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
            out = src[rg_idx, s_idx]
        return out.contiguous()


    # ---------------------------------------------------------------- sk2: m <= 32, swapped operands, no cross-CTA reduction
    SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL); a name ending in _rNNN caps the ring at NNN KB
        ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
        ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
        ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
        ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
        ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
        ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
        ('slice8_r192', 'unpooler', UNPOOLER_SLICE_ROWS, 8, 1, 1, 8, 2, 2, 1),
        ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
        ('pooler16t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),      # tokr16 [0.973 vs 1.0]
        ('unpooler16t_r72', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # tokr16 [0.961 vs 0.969]
        ('qkv8t_r144', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),         # tokr8 [0.969 vs 0.979]
        ('qkv16t_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),        # tokr16 [0.985 vs 0.994]
        ('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1),                  # tokr16 [0.937 vs 0.957]
    ]
    SK2_PICK = {  # (layer, tokr) -> config name
        ('o', 8): 'o8h_r88',
        ('o', 16): 'o16t_r68',
        ('o', 32): 'o32h_r100',
        ('pooler', 8): 'pooler8h_r72',
        ('pooler', 16): 'pooler16t_r192',
        ('pooler', 32): 'pooler32_r144',
        ('qkv_gate', 8): 'qkv8t_r144',
        ('qkv_gate', 16): 'qkv16t_r200',
        ('qkv_gate', 32): 'qkv32_r208',
        ('router', 8): 'router',
        ('router', 16): 'router',
        ('router', 32): 'router',
        ('unpooler', 8): 'slice8_r192',
        ('unpooler', 16): 'unpooler16t_r72',
        ('unpooler', 32): 'unpooler32_r144',
    }
    SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
    USE_SK2 = True
    WIDE_BF16 = False  # the wide (RS-wgmma) kernel never beats cuBLAS for BF16 m 33..256


    def _sk2_name(layer, m):
        return SK2_PICK[(layer, _sk2_tokr(m))]


    def _sk2_tokr(m):
        return 8 if m <= 8 else 16 if m <= 16 else 32


    def _sk2_geom(cfg, tokr):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        hl = layer == "router"
        tu = 2048 if hl else 768
        ups = WSK * UPW
        wby = NWR * WSK * UPW * NT * tu
        xby = ups * tokr * 128
        sst = (xby + wby + 1023) // 1024 * 1024
        nst = -(-(SK2_K[layer] // (ups * 64)) // CL)   # (the largest rank's stages)
        mt = 2 if tokr == 32 else 1
        toks = 16 * mt
        cp = R + 1
        nesc = 4 if not hl else 1   # (3 when the ring has 3 stages; sized for 4 here)
        rp = R + ((40 - (R % 32)) % 32)
        # everything but the ring: alignment pad, barriers, corrections, peer sums, escape table (<= 64 entries), zero row
        extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
                 + 64 * 8 + 128)
        cap_kb = SK2_DEFAULT_RING_CAP_KB
        tail = name.rsplit("_r", 1)
        if len(tail) == 2 and tail[1].isdigit():
            cap_kb = int(tail[1])
        ring = min(cap_kb * 1024, SK2_RING_SMEM_BUDGET_BYTES - extra)
        nstage = max(SK2_MIN_STAGES, min(SK2_MAX_STAGES, nst, ring // sst))
        return hl, nstage


    def _sk2_src(sm90):
        lines = []
        for ci, cfg in enumerate(SK2_CONFIGS):
            name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
            for tokr in (8, 16, 32):
                if SK2_PICK.get((layer, tokr)) != name:
                    continue
                hl, nstage = _sk2_geom(cfg, tokr)
                mt = 2 if tokr == 32 else 1
                ot = "float" if hl else "__nv_bfloat16"
                lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                             f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
        return SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))


    def sk2_tileunits(w):
        """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9."""
        n, k = w.shape
        t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
        return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


    def sk2_compress(w):
        """Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable."""
        n, k = w.shape
        tu = sk2_tileunits(w)
        bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        hi = bits >> 8
        low = bits & 0xFF
        e71 = hi & 0x7F
        sign = hi >> 7
        base, o = _exponent_window(e71)
        code = (sign << 3) | o
        esc = _escapes(e71, base)
        if int(esc.sum()) > n * k // MAX_ESCAPE_DENOMINATOR:
            return None
        T, U = n // 8, k // 64
        c8 = code.to(torch.uint8)
        w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
        w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
        codes = torch.cat([w0, w1], dim=-1)                  # [T, U, 32, 8]
        packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
        idx = esc.nonzero()                                   # [E, 4]: tile, unit, lane, v
        dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
        wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
        wtrue = tu[esc].float()
        delta = wtrue - wdec
        if not bool(((wdec + delta) == wtrue).all()):
            return None
        lane, v = idx[:, 2], idx[:, 3]
        rows = idx[:, 0] * 8 + (lane >> 2)
        cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
        wbits = (tu[esc].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).contiguous()   # the original BF16 weights
        return {"packed": packed, "base4": base * BASE4_REPLICATE, "rows": rows, "cols": cols, "wbits": wbits}


    CG_CONFIGS = {"qkv_gate": (2, 256), "o": (2, 64), "unpooler": (2, 128), "pooler": (2, 512)}   # layer -> (rows per CTA, threads); m = 1
    # (v137: pooler m=1 cgemv 2x512 measured -0.34 us vs sk2; 2x256 ties sk2, 4/8 rows or 1 row lose)


    def cg_compress(w, rows_per_cta):
        """Row-major 12-bit planes for cgemv (lossless; escapes corrected exactly from a per-CTA table)."""
        n, k = w.shape
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        hi, low = bits >> 8, bits & 0xFF
        e71, sign = hi & 0x7F, hi >> 7
        base, o = _exponent_window(e71)
        code = ((sign << 3) | o).to(torch.uint8)
        esc = _escapes(e71, base)
        if int(esc.sum()) > n * k // MAX_ESCAPE_DENOMINATOR:
            return None
        g = code.view(n, k // 8, 8)
        C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
        L = low.to(torch.uint8).contiguous()
        rows, cols = esc.nonzero(as_tuple=True)          # row-major order
        dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
        wtrue = w[rows, cols].float()
        delta = (wtrue - dec).contiguous()
        if not bool(((dec + delta) == wtrue).all()):
            return None
        ncta = n // rows_per_cta
        cta = rows // rows_per_cta
        counts = torch.bincount(cta, minlength=ncta)
        pmax = int(counts.max()) + 1 if rows.numel() else 1
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
        tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
        tab[:, 0, 0] = counts.to(torch.int32)
        tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
        tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)   # the original BF16 weights; the kernel forms the correction
        return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * BASE4_REPLICATE}


    def sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
        """Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows."""
        dev = tub.device
        U = tub.shape[1]
        ups = WSK * UPW
        ar = lambda v: torch.arange(v, device=dev)
        s = ar(U // ups).view(-1, 1, 1, 1, 1)
        c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
        w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
        uu = ar(UPW).view(1, 1, 1, -1, 1)
        nt = ar(NT).view(1, 1, 1, 1, -1)
        tile = c * (R // 8) + (w // WSK) * NT + nt
        unit = s * ups + (w % WSK) * UPW + uu
        return tub[tile, unit].contiguous()


    SK2_BALANCE = True   # v135: deal 4-row groups to sk2 row blocks so every block carries ~the same number of escapes
    SK2_BALANCE_LAYERS = {"qkv_gate"}


    def sk2_balance_perm(w, n_rows, R):
        """Row order for R-row sk2 blocks over the first n_rows rows of w: rows sorted by escape count and snake-dealt, so
    every block holds ~the same number of escapes (the escape warps' work sets the CTA tail). perm[p] = original row."""
        wl = w[:n_rows]
        bits = wl.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        e71 = (bits >> 8) & 0x7F
        base, _ = _exponent_window(e71)
        cnt = _escapes(e71, base).sum(1)
        # 4-row groups are dealt (keeps 4-column output groups contiguous: vector stores in the epilogue)
        G4 = 4
        tcnt = cnt.view(n_rows // G4, G4).sum(1)
        order = torch.argsort(tcnt, descending=True, stable=True)
        TB = R // G4                                   # groups per block
        nb = n_rows // R
        i = torch.arange(n_rows // G4, device=w.device)
        rnd, pos = i // nb, i % nb
        blk = torch.where(rnd % 2 == 0, pos, nb - 1 - pos)
        tperm = torch.empty(n_rows // G4, dtype=torch.long, device=w.device)
        tperm[blk * TB + rnd] = order
        return (tperm.view(-1, 1) * G4 + torch.arange(G4, device=w.device).view(1, G4)).reshape(-1)


    def sk2_table_add_perm(tab, pmax, perm, R):
        """Append each CTA's row map (R int32 output columns) after its escapes; entry 0 .y = the map's int2 offset."""
        ncta = tab.shape[0]
        pm2 = pmax + R // 2
        pm2 += pm2 % 2
        out = torch.zeros(ncta, pm2, 2, dtype=torch.int32, device=tab.device)
        out[:, :pmax] = tab
        out[:, pmax:pmax + R // 2] = perm.to(torch.int32).view(ncta, R // 2, 2)
        out[:, 0, 1] = pmax
        return out.contiguous(), pm2


    def sk2_table(comp, n_cfg, R):
        """Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, the escaped weight's
    original BF16 bits); the kernel forms the exact correction (w - decode(w)) itself."""
        dev = comp["rows"].device
        rows, cols, delta = comp["rows"], comp["cols"], comp["wbits"]
        keep = rows < n_cfg
        rows, cols, delta = rows[keep], cols[keep], delta[keep]
        ncta = n_cfg // R
        cta = rows // R
        counts = torch.bincount(cta, minlength=ncta)
        pmax = int(counts.max().item()) + 1 if rows.numel() else 1
        pmax += pmax % 2
        order = torch.argsort(cta.long() * 65536 + cols.long())   # per CTA in k order (the kernel walks K once)
        cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
        start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
        start[1:] = torch.cumsum(counts, 0)
        slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
        tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
        tab[:, 0, 0] = counts.to(torch.int32)
        tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
        tab[cta, slot, 1] = delta.to(torch.int32)
        return tab.contiguous(), pmax


    def sk2_prepare(weights, state):
        state["sk2"] = {}
        dev = weights["o"].device
        shared = {}
        for ci, cfg in enumerate(SK2_CONFIGS):
            name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
            w = weights[layer]
            if layer == "router":
                hi = w.to(torch.bfloat16)
                lo = (w - hi.float()).to(torch.bfloat16)
                T, U = w.shape[0] // 8, w.shape[1] // 64
                tub = torch.cat([sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                                 sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
                stream = sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
                tab, pmax, b4 = state["dummy"], 0, 0
            else:
                perm = None
                if SK2_BALANCE and CL == 1 and layer in SK2_BALANCE_LAYERS:
                    pkey = (layer, n_rows, R)
                    bal = state.setdefault("sk2_bal", {})
                    if pkey not in bal:
                        pr = sk2_balance_perm(w, n_rows, R)
                        bal[pkey] = (pr, sk2_compress(w[:n_rows][pr].contiguous()))
                    perm, comp = bal[pkey]
                    if comp is None:
                        perm = None
                if perm is None:
                    comp = state.setdefault("sk2_comp", {}).get(layer)
                    if comp is None:
                        comp = sk2_compress(w.contiguous())
                        state["sk2_comp"][layer] = comp
                if comp is None:
                    continue
                lkey = (layer, n_rows, R, NT, NWR, WSK, UPW, perm is not None)   # configs differing only in NACC / ring depth share a stream
                if lkey not in shared:
                    tab, pmax = sk2_table(comp, n_rows, R)
                    if perm is not None:
                        tab, pmax = sk2_table_add_perm(tab, pmax, perm, R)
                    shared[lkey] = (sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tab, pmax)
                stream, tab, pmax = shared[lkey]
                b4 = comp["base4"]
            state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
        state.pop("sk2_comp", None)
        state.pop("sk2_bal", None)


    def _run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
        name = _sk2_name(layer, m)
        ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
        out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        _get_ext().sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, _sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
        return out


    def _mt_key(m):
        return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


    # ---------------------------------------------------------------------------
    # prepare(): build every weight layout once into `state`. The CUTLASS
    # extension builds on a background thread while the weights are compressed.
    # ---------------------------------------------------------------------------
    def _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg):
        """Ensure one skinny decomposition's prepared inputs exist in ``state``:
    the per-CTA weight stream, its split-K counter, and (compressed only) its
    per-CTA escape table. Idempotent: shared decompositions are built once."""
        key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
        if key not in state["streams"]:
            state["streams"][key] = build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
        ckey = (name, splits, rgc, wsn, ks, n_cfg)
        if ckey not in state["cnt"]:
            state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  # per-config counters
        if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
            state["tabs"][(name, splits, rgc, n_cfg)] = build_patch_table(c, n_cfg, kk, splits, rgc)


    def prepare(weights):
        _get_ext()  # compile now, so a build failure surfaces in prepare(), not mid-run
        # the CUTLASS extension builds in the background (nvcc is a subprocess) while the weights are compressed; started
        # after the main build so the two load_inline calls never overlap
        cut_thread = None
        if CUTLASS_LAYERS and torch.cuda.get_device_capability()[0] >= 9 and _cut_ext is None and _cut_err is None:
            cut_thread = threading.Thread(target=_build_cutlass, daemon=True)
            cut_thread.start()
        dev = weights["qkv_gate"].device
        sm90_ = torch.cuda.get_device_capability()[0] >= 9
        # on sm_90 the sk2 kernels replace the older skinny kernels and the BF16 wide kernel (never faster than cuBLAS)
        old_paths = not (sm90_ and USE_SK2)
        state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
        state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
        # split-K partials: kernels run one at a time, so every layer shares one workspace
        ws_floats = 0
        for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
            mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
            ws_floats = max(ws_floats, splits * mmax * WEIGHT_ROWS[lname])
        for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
            ws_floats = max(ws_floats, splits * mm * WEIGHT_ROWS[lname])
        state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
        gemm_floats = 0
        for (lname, bn), (nwg, splits, stages) in GEMM_CONFIGS.items():
            gemm_floats = max(gemm_floats, (WEIGHT_ROWS[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
        for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
            gemm_floats = max(gemm_floats, nz * (WEIGHT_ROWS[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
        state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
        for name, w in weights.items():
            if name == "router":
                hi = w.to(torch.bfloat16)
                lo = (w - hi.float()).to(torch.bfloat16)
                state["router_hi_t"] = hi.t().contiguous()
                state["router_f32"] = w.float().contiguous()   # the router GEMV (m = 1) reads the FP32 weights as given
                state["router_lo_t"] = lo.t().contiguous()
                state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   # gemm90 fallback (m 33..256), rk
                state["rk_zc"] = torch.zeros(8, dtype=torch.int32, device=w.device)   # rk: grid counters, one per token-tile count
                # hi and lo tiles of the same k-step back to back: [rg, s, 2, 32, 8]
                src = torch.stack([pack_fragments(hi), pack_fragments(lo)], dim=2).contiguous()
                comp, kk = False, w.shape[1]
                c = None
                # wide kernel stream: [rb][unit][warp][hi 4 k-steps x 512 B | lo 4 k-steps x 512 B], 64-row blocks
                nrg_r = w.shape[0] // 16
                u_all = w.shape[1] // 64
                t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   # [rg, u, hl, 4, 32, 8]
                nwg_r = WIDE_CONFIGS[("router", 128, 1)][0]
                t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
                del t
                for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
                    if lname == "router":
                        state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
            else:
                w = w.contiguous()
                state["raw"][name] = w
                if name in CG_CONFIGS:
                    wc = w[:UNPOOLER_SLICE_ROWS] if name == "unpooler" else w   # the unpooler at m = 1 is the slice
                    cg = cg_compress(wc.contiguous(), CG_CONFIGS[name][0])
                    if cg is not None:
                        state.setdefault("cg", {})[name] = cg
                if name in TORCHT_LAYERS and sm90_:
                    state.setdefault("rawT", {})[name] = w.t().contiguous()
                c = pack_compressed(w) if name in COMPRESSED_LAYERS else None
                if c is not None:
                    src, comp = c["packed"].transpose(0, 1).contiguous(), True  # [rg, u, 3, 32, 16]
                    state["base4"][name] = c["base4"]
                    del c["packed"]
                else:
                    src, comp = pack_fragments(w), False
                kk = w.shape[1]
            n = w.shape[0]
            state["comp"][name] = comp
            for (lname, mtk), (splits, rgc, wsn, ks) in TMA_CONFIGS.items():
                if lname != name or not old_paths:
                    continue
                n_cfg = UNPOOLER_SLICE_ROWS if (name == "unpooler" and mtk <= 1) else n
                _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg)
            if comp and old_paths:
                for (lname, mm), (splits, rgc, wsn, ks) in TMA_ALT.items():
                    if lname != name:
                        continue
                    n_cfg = UNPOOLER_SLICE_ROWS if (name == "unpooler" and mm <= 8) else n
                    _ensure_skinny_stream(state, name, src, comp, c, kk, n, dev, splits, rgc, wsn, ks, n_cfg)
            for (lname, tok, nz), (nwg, splits, stages) in WIDE_CONFIGS.items():
                if lname != name or not comp or not (old_paths or WIDE_BF16):   # unused on sm_90 (cuBLAS / gq / gqx)
                    continue
                rgc = 4 * nwg
                key = (name, 0, rgc, 1, 4, n)
                if key not in state["streams"]:
                    state["streams"][key] = build_stream(src, comp, n, kk, splits, rgc, 1, 4)
                if (name, splits, rgc, n) not in state["tabs"]:
                    state["tabs"][(name, splits, rgc, n)] = build_patch_table(c, n, kk, splits, rgc)
                state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
            del src
            state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        state["sm90"] = sm90_
        state["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count   # rk: one resident CTA per SM
        state["old_paths"] = old_paths
        sk2_prepare(weights, state)
        if cut_thread is not None:
            cut_thread.join(timeout=CUTLASS_BUILD_TIMEOUT)   # ~70-80 s on an H200 pod; never let it threaten prepare's limit
            if cut_thread.is_alive():
                print(f"[okc] CUTLASS route off: build still running after {CUTLASS_BUILD_TIMEOUT} s", file=sys.stderr)
        if _cut_ext is not None and (cut_thread is None or not cut_thread.is_alive()):
            # Persistent scheduler workspace: a few KB at most, sized with headroom.
            state["cut_ws"] = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
            state["cut_layers"] = set(CUTLASS_LAYERS) & set(state["raw"])
        elif _cut_err is not None:
            print(f"[okc] CUTLASS route off: {_cut_err}", file=sys.stderr)
        _selftest(weights, state)
        _autotune(weights, state)
        torch.cuda.empty_cache()
        return state


    # ---------------------------------------------------------------------------
    # Routing, selftest, autotune. The static route maps (layer, m) to a family;
    # _candidates adds the autotuner alternatives; _autotune times them as CUDA
    # graph replays from a cold L2 and keeps the fastest per (layer, m) in
    # state["choice"]. An untuned m in _nearby_choice takes its bucket's choice.
    # ---------------------------------------------------------------------------
    # Which implementation runs a (layer, m): "skinny" (TMA-streamed tensor-core GEMV, m <= 32),
    # "gemm" (TMA + wgmma split-K GEMM, sm_90, m <= 256) or "torch" (cuBLAS, as SGLang).
    ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
    WIDE_SM80_TEST = False  # local check of the wide kernel's sm_80 path
    ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


    def _route(layer, m, sm90):
        for mmax, r in (ROUTE_ROUTER if layer == "router" else ROUTE_BF16):
            if m <= mmax:
                return r
        return "torch"


    USE_RK = True   # v141: router m 33..256 candidate 'rk' (order-dependent FP32 adds; stress compares it by tolerance)


    def _route_key(layer, m, route):
        if route == "sk2":
            return (layer, route, _sk2_name(layer, m), _sk2_tokr(m))
        if route == "skinny":
            return (layer, route, _mt_key(m))
        if route == "wide":
            return (layer, route) + _wide_key(m)
        if route == "rk":
            return (layer, route, (m + 63) // 64)
        return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


    SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


    def _selftest(weights, state):
        """Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on."""
        state["disabled"] = set()
        g = torch.Generator(device=weights["o"].device)
        g.manual_seed(SELFTEST_SEED)
        for layer, w in weights.items():
            n, k = w.shape
            for m in SELFTEST_M:
                if _PRUNE and _TABLE.get('%s:%d' % (layer, m), _DEFAULT) != 'swapab':
                    continue
                nn = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and m <= 8) else n
                x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
                ref = x.double() @ w[:nn].double().T
                if layer != "router":
                    ref = ref.to(torch.bfloat16).double()
                tol = 2e-5 if layer == "router" else 5e-3
                for ch in _candidates(layer, m, state):
                    if ch[0] == "torch":
                        continue
                    key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                           else _route_key(layer, m, ch[0]))
                    try:
                        out = _run(ch, x, layer, m, nn, k, state).double()
                        torch.cuda.synchronize()
                        err = ((out - ref).norm() / ref.norm()).item()
                    except RuntimeError as e:  # launch failure -> fall back
                        err = float("inf")
                        print(f"[okc] selftest {layer} m={m} {ch[0]}: {e}", file=sys.stderr)
                    if not err < tol:
                        state["disabled"].add(key)
                        print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)


    # ---------------------------------------------------------------------------
    # Running one projection: each _run_* allocates a fresh output and launches
    # one route. Nothing is cached across calls.
    # ---------------------------------------------------------------------------
    def _run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
        splits, rg, wsn, ks = cfg
        mtk = _mt_key(m)
        out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
        cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
        fn = _get_ext().skinny_tma_a
        if state["comp"][layer]:
            tab, pmax = state["tabs"][(layer, splits, rg, n)]
            fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
               max(mtk, 1), rg, wsn, 1, WEIGHT_ROWS[layer] // 16, ks, dg)
        else:
            # Uncompressed: no escapes, so the patch table is the dummy and
            # pmax/base4 are 0 (the kernel skips exact correction).
            fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
               max(mtk, 1), rg, wsn, 0, WEIGHT_ROWS[layer] // 16, ks, dg)
        return out


    def _run_wide(x, layer, m, n, k, state):
        tok, nz = _wide_key(m)
        nwg, splits, _ = WIDE_CONFIGS[(layer, tok, nz)]
        rgc = 4 * nwg
        if layer == "router":
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
        wst = state["router_cat"] if layer == "router" else state["streams"][(layer, 0, rgc, 1, 4, n)]
        _get_ext().wide(wst, x, out, state["mid_ws_shared"],
                        state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
        return out


    def _run_gemm(x, layer, m, n, k, state):
        bn = 64 if m <= 64 else 128 if m <= 128 else 256
        nwg, splits, _ = GEMM_CONFIGS[(layer, bn)]
        out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        w = state["router_cat"] if layer == "router" else state["raw"][layer]
        _get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                          splits, nwg, bn, 0)
        return out


    def _run_torch(x, layer, m, n, k, state):
        if layer == "router":
            acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
            acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
            return acc
        return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


    def _run(choice, x, layer, m, n, k, state):
        kind = choice[0]
        if kind == "skinny":
            return _run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
        if kind == "wide":
            return _run_wide(x, layer, m, n, k, state)
        if kind == "sk2":
            return _run_sk2(x, layer, m, n, k, state)
        if kind == "gemm":
            return _run_gemm(x, layer, m, n, k, state)
        if kind == "gq":
            return _run_gq(x, layer, m, n, k, state)
        if kind == "gqx":
            return _run_gqx(x, layer, m, n, k, state)
        if kind == "cgemv":   # m = 1: 12-bit compressed GEMV (qkv, o, unpooler slice, pooler)
            cg = state["cg"][layer]
            rows, nth = CG_CONFIGS[layer]
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _get_ext().cgemv(cg["L"], cg["C"], x, cg["tab"], cg["pmax"], out, cg["base4"], rows, nth)
            return out
        if kind == "sgemv":   # unpooler slice, m = 1: BF16 weights straight from memory (4 rows per CTA)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _get_ext().sgemv(state["raw"][layer], x, out, n)
            return out
        if kind == "rgemv":   # router, m = 1: FP32 weights straight from memory (one row per CTA)
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            _get_ext().rgemv(state["router_f32"], x, out, m)
            return out
        if kind == "cutlass":   # m 129..256: CUTLASS pingpong 128x64 (see CUTLASS_SRC)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _cut_ext.cut_gemm(state["raw"][layer], x, out, state["cut_ws"])
            return out
        if kind == "rk":   # router m 33..256: dedicated split-K GEMM with order-dependent FP32 adds (see RK_SRC)
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            _get_ext().rk(state["router_cat"], x, out, _rk_splits(m, state["nsm"]), state["rk_zc"])
            return out
        if kind == "torchT":   # cuBLAS on a K-major copy of W: a different (faster) kernel choice for unpooler m=128
            return torch.mm(x, state["rawT"][layer])
        return _run_torch(x, layer, m, n, k, state)


    RK_SPLITS = {1: 16, 2: 16, 3: 10, 4: 8}   # token tiles of 64 -> K splits (grid 4 x ntt x S <= 128; pod/rk.py sweep)


    def _rk_splits(m, nsm=132):
        """K splits for m tokens: the measured table, capped so the grid (4 x ntt x S, one CTA per SM, all resident: the
    in-kernel clear waits for every CTA) fits the GPU's SMs."""
        ntt = (m + 63) // 64
        return max(1, min(RK_SPLITS[ntt], nsm // (4 * ntt)))


    def _static_choice(layer, m, state):
        route = _route(layer, m, state["sm90"])
        if not state.get("old_paths", True):
            if route == "skinny":
                route = "sk2" if (_sk2_name(layer, m) in state.get("sk2", {})
                                  and _route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
            elif route == "wide" and layer != "router" and not WIDE_BF16:
                route = "torch"
        if route != "torch" and _route_key(layer, m, route) in state.get("disabled", ()):
            # fall back: router -> the wgmma GEMM (validated earlier) when possible, else cuBLAS
            route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                               and _route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
        if route == "skinny":
            mtk = _mt_key(m)
            return ("skinny", TMA_CONFIGS[(layer, mtk)], (layer, mtk))
        return (route,)


    def _candidates(layer, m, state):
        """Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS."""
        c = [_static_choice(layer, m, state)]
        if c[0][0] == "skinny" and (layer, m) in TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
            c.append(("skinny", TMA_ALT[(layer, m)], ("alt", layer, m)))
        if (c[0][0] == "wide" and layer == "router" and state["sm90"]
                and _route_key(layer, m, "gemm") not in state["disabled"]):
            c.append(("gemm",))
        if (m <= 32 and c[0][0] != "sk2" and _sk2_name(layer, m) in state.get("sk2", {})
                and _route_key(layer, m, "sk2") not in state["disabled"]):
            c.insert(0, ("sk2",))
        if (USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gq_bn(m)) in GQ_CONFIGS
                and _route_key(layer, m, "gq") not in state["disabled"]):
            c.insert(0, ("gq",))
        if (USE_GQX and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, _gqx_bn(m)) in GQX_CONFIGS
                and _route_key(layer, m, "gqx") not in state["disabled"]):
            c.insert(0, ("gqx",))
        if layer == "router" and m == 1 and "router_f32" in state:
            c.insert(0, ("rgemv",))
        if (USE_RK and layer == "router" and 32 < m <= 256 and state["sm90"] and "rk_zc" in state and state.get("nsm", 0) >= 16
                and _route_key(layer, m, "rk") not in state["disabled"]):
            c.insert(0, ("rk",))
        if layer == "unpooler" and m == 1:
            c.insert(0, ("sgemv",))
        if m == 1 and layer in state.get("cg", {}) and _route_key(layer, m, "cgemv") not in state["disabled"]:
            c.insert(0, ("cgemv",))
        if layer in TORCHT_LAYERS and 32 < m <= 256 and layer in state.get("rawT", {}):
            c.append(("torchT",))
        if (128 < m <= 256 and layer in state.get("cut_layers", ()) and _cut_ext is not None
                and _route_key(layer, m, "cutlass") not in state["disabled"]):
            c.append(("cutlass",))
        if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
            c.append(("torch",))
        return c


    AUTOTUNE = True
    TUNE_M = (1, 16, 32, 8, 128, 256)   # the scored token counts


    def _autotune(weights, state):
        """Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m)."""
        state["choice"] = {}
        if not AUTOTUNE:
            return
        dev = weights["o"].device
        flush = torch.ones(L2_FLUSH_ROWS, 1024, dtype=torch.float32, device=dev)  # read-only sweep
        sink = torch.empty(flush.shape[1], dtype=torch.float32, device=dev)
        g = torch.Generator(device=dev)
        g.manual_seed(AUTOTUNE_SEED)
        reps = AUTOTUNE_REPS
        for m in TUNE_M:
            for layer, w in weights.items():
                if _PRUNE and _TABLE.get('%s:%d' % (layer, m), _DEFAULT) != 'swapab':
                    continue
                n = UNPOOLER_SLICE_ROWS if (layer == "unpooler" and m <= 8) else w.shape[0]
                k = w.shape[1]
                cands = _candidates(layer, m, state)
                if len(cands) < 2:
                    continue
                x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
                graphs = []
                for ch in cands:
                    _run(ch, x, layer, m, n, k, state)
                    torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        _run(ch, x, layer, m, n, k, state)
                    graphs.append(gr)
                evs = []
                for r in range(reps):
                    order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                    for i in order:
                        torch.sum(flush, dim=0, out=sink)
                        e0 = torch.cuda.Event(enable_timing=True)
                        e1 = torch.cuda.Event(enable_timing=True)
                        e0.record()
                        graphs[i].replay()
                        e1.record()
                        evs.append((i, e0, e1))
                torch.cuda.synchronize()
                per = [[] for _ in cands]
                for i, e0, e1 in evs:
                    per[i].append(e0.elapsed_time(e1))
                med = [sorted(v)[len(v) // 2] for v in per]
                best = min(range(len(cands)), key=lambda i: med[i])
                state["choice"][(layer, m)] = cands[best]
                # CUTLASS is tuned for the scored m only (its partial token tiles are slow: m=129 +14 us, m=200 +4 us vs
                # cuBLAS): other m of the bucket take the best other candidate
                alt = [i for i in range(len(cands)) if cands[i][0] != "cutlass"]
                if alt:
                    state.setdefault("choice_alt", {})[(layer, m)] = cands[min(alt, key=lambda i: med[i])]
                state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
                del graphs
        del flush
        torch.cuda.synchronize()
        # one line per shape in the run's log: candidate times (us, raw graph replay incl. ~4 us replay cost) and the pick
        for layer, m, tl in state.get("tune_log", []):
            ch = state["choice"][(layer, m)]
            desc = " ".join(f"{name}{'/' + 'x'.join(map(str, c[1])) if c[0] == 'skinny' else ''}={t}"
                            for (name, t), c in zip(tl, _candidates(layer, m, state)))
            pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")
            print(f"[okc] tune {layer} m={m}: {desc} -> {pick}", file=sys.stderr)
        print(f"[okc] disabled: {sorted(state['disabled'])}", file=sys.stderr)


    def _nearby_choice(layer, m, state):
        """An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket."""
        tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
        if not tm:
            return None
        if tm == 256 and m != 256 and layer != "router":
            # an untuned m in 129..255 runs SGLang's own call (F.linear): the m=256 picks (CUTLASS, gqx, torchT) were
            # only timed at m=256 and some lose to cuBLAS at partial tiles (pooler CUTLASS m=129 +14 us, unpooler
            # torchT m=240 +11%)
            return ("torch",)
        ch = state.get("choice", {}).get((layer, tm))
        if ch is None:
            return None
        if ch[0] == "cutlass" and m != tm:
            ch = state.get("choice_alt", {}).get((layer, tm))
            if ch is None:
                return None
        if ch[0] == "gq" and (layer, _gq_bn(m)) not in GQ_CONFIGS:
            return None
        return ch


    # ---------------------------------------------------------------------------
    # Entry point: the harness calls prepare(weights) once, then kernel_fn per call.
    # ---------------------------------------------------------------------------
    def kernel_fn(data, state):
        x = data["x"]
        cfg = data["config"]
        layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
        choice = state.get("choice", {}).get((layer, m)) or _nearby_choice(layer, m, state) or _static_choice(layer, m, state)
        out = _run(choice, x, layer, m, n, k, state)
        return out


    return {'prepare': prepare, 'kernel_fn': kernel_fn, 'warm': _get_ext}

_NB_C12 = False


