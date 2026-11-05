# okc: problem=muse_prefill_layer
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
# This file has been modified from the original.
"""Muse Spark prefill layer: one transformer layer step over a batch of tokens.

Algorithm (one ``kernel_fn`` call, in order; every stage is one launch or a
few launches, and the stage order is the data dependency order):

1. **QKV projection.** Dense GEMM (cuBLASLt-tuned, else ``torch.mm``) of the
   attention input, producing ``[Q | K | V | gate]`` per token.
2. **Q/K prep.** Per-token RMS-norm of Q and K, RoPE rotation, FP8
   (E4M3) quantisation of Q, and stores of K/V into the paged KV cache,
   scaled by the cache scales. A CUDA kernel (``qk_prep``) does this; the
   Triton ``_qk_prep_kernel`` is the same computation and is kept as the
   fallback path.
3. **Attention.** Either the FA3 paged FP8 path followed by a fused
   ``norm * sigmoid(gate)`` epilogue, or the file's own Hopper attention
   kernel (``attn_fwd``) with that epilogue fused in. The own kernel is
   picked per shape class in ``prepare`` after validating it against the
   FA3 path. On non-Hopper devices a slow FP32 reference attention runs
   instead (testing only, never on the scored path).
4. **O projection + first boundary.** Dense GEMM, then the boundary mixes
   the branch into the fp32 residual stream:
   ``stream = alpha * stream + beta * rms(branch)`` and emits the next
   block's BF16 input as ``rms(stream) * (1 + g_pre)``. The boundary
   kernels also zero the MoE work-counters as a side effect.
5. **Router + pooler, overlapped.** The router GEMM and top-8 routing run
   on a side stream while the pooler GEMM runs on the main stream; the
   main stream then joins the side stream. Routing picks ``TOP_K`` experts
   per token, per local expert.
6. **Grouped MoE.** The pooler output is gathered (with the pre-expert
   norm) into per-expert token rows, then the fused persistent kernel
   runs both expert GEMMs: NVFP4 weights are dequantised in registers as
   the wgmma A-operand, BF16 activations come from shared memory, and
   the epilogue is the fused SiLU*up (``silu(gate) * up``) for GEMM1.
7. **Combine + unpooler + second boundary.** The per-route rows are
   combined (per-expert RMS-norm, weighted by routing weights) back into
   one ``[T, D_EXPERT]`` tensor, the unpooler GEMM brings it back to
   ``HIDDEN``, and the second boundary mixes it into the stream and
   emits the layer output. When the token count is large the second
   boundary instead rebuilds the fp32 stream from the first boundary's
   saved RMS factor, saving a 32 KB-per-row round trip.

Design rules this file follows:

* ``prepare(weights)`` runs once: it builds the CUDA extension, repacks
  the NVFP4 expert weights into the kernel's layout, and tunes the two
  choices that depend on the machine (cuBLASLt algorithm per dense GEMM,
  own-attention variant per shape class). Those tuning results are the
  only state carried between calls, plus the compiled extension and one
  side stream.
* ``kernel_fn(data, state)`` allocates every activation fresh on every
  call and returns fresh tensors; nothing computed from one call's
  inputs is reused by another.
* The timed path performs no host synchronisation: no ``.item()``,
  ``.tolist()``, ``torch.cuda.synchronize()`` or data-dependent host
  branch. (The FP32 fallback attention and the self-tests do sync; they
  are not on the Hopper path.)
* The CUDA/C++ sources at the bottom (``CPP``, ``GLUE``, ``_CUDA_SRC``,
  ``LT_CPP``, ``LT_GLUE``) are the tuned algorithm with every dead
  preprocessor branch, dead function and dead define removed (the
  taken path is unchanged: the kernels, tile sizes and launch
  configs are identical). The one host-supplied switch, MOE_FAST_SCALE
  (0 when the weights have non-normal block scales), is kept with
  both of its branches. Everything above
  them -- constants, Triton kernels, ``prepare``, ``kernel_fn`` and the
  helpers -- is the Python layer. Anything in that layer
  whose equivalence is not provable from the source alone needs GPU
  gating before it is treated as the submission.
"""

import hashlib

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
from triton.language.extra import libdevice

try:  # FA3 (Hopper) from the benchmark image
    from sgl_kernel.flash_attn import flash_attn_with_kvcache as _fa3
except Exception:  # local testing
    _fa3 = None


# --------------------------------------------------------------------------------------
# Model geometry. Every other size in the file derives from these.
# --------------------------------------------------------------------------------------

HIDDEN = 8192
D_EXPERT = 4096
TOP_K = 8
LOCAL_EXPERTS = 32
HQ, HKV, HD = 16, 2, 64
Q_SIZE, KV_SIZE = HQ * HD, HKV * HD
# QKV gate output per token: [Q (Q_SIZE) | K (KV_SIZE) | V (KV_SIZE) | gate (Q_SIZE)].
QKV_SIZE = 2 * Q_SIZE + 2 * KV_SIZE
GATE_OFFSET = Q_SIZE + 2 * KV_SIZE  # gate starts after Q, K, V
RMS_EPS = 1e-5  # RMS-norm epsilon (post-norm and Q/K-norm; the CUDA kernels use the same)
POST_NORM_EPS = 1e-8  # boundary / combine RMS-norm epsilon (the CUDA kernels use the same)
ROPE_THETA = 500_000.0
FP8 = torch.float8_e4m3fn
LOG2E = 1.4426950408889634  # converts a natural-log softmax scale for exp2

# NVFP4 expert weights: K is walked in stages of KS elements; each
# (n-tile, k-stage) weight block holds KS * 128 code bytes + KS * 8 scale bytes.
KS, WBLK = 64, 4608


# --------------------------------------------------------------------------------------
# Tuning knobs. Each one is a measured default, not a guess: the comment says
# what it changes and why the value is what it is. All of them were formerly
# scattered through the file (several after their first use); they are
# collected here so the whole launch configuration is visible at once.
# --------------------------------------------------------------------------------------

# --- MoE launch ---
# Tile slots per weight n-tile in the (non-persistent) expert GEMMs.
P_GEMM1 = 8
P_GEMM2 = 16
MOE_LARGE_T = 8192  # token count from which the "large" launch mode/order apply
# K permuted inside every 64-wide k-stage of the expert GEMMs (MOE_KPERM in _CUDA_SRC): a lane's 16 weights per row are one
# block-scale, so it builds one scale pair per stage instead of four. _build passes it to the compiler; the weight repack
# (_repack) and the gather kernel apply the same permutation. Measured: fused expert kernel -2.5% at T=753, -0.3% at T=3012, even at 16384.
KPERM = True
# Launch mode per token count: "fused" = both GEMMs in one persistent launch,
# "persist" = two persistent launches, anything else = two per-tile-grid launches.
MOE_MODE_SMALL = "fused"
MOE_MODE_LARGE = "fused"
# Tile order for the persistent work queue: 0 expert order, 1 biggest first,
# 2 big first + small tiles spread between.
MOE_ORDER_SMALL = 1
MOE_ORDER_LARGE = 1
MOE_FINE_ORDER_SHIFT = 8  # bit 8 of the plan order / fused flags: fine remainder tiles
MIN_TILE_ROWS = 32  # smallest token tile; the plan's tile-count bound divides by this
MOE_TILE_SLACK = 64  # extra tile slots in the plan buffer, beyond the bound
MAX_PADDED_ROWS_PER_EXPERT = 256  # per-expert row allowance in the padded row buffer
FINE_MIN_T = 512  # from this token count, remainder tiles round to 32 rows
SKINNY_TEST = False  # feasibility-test build: the fused MoE kernel at 2 CTAs/SM
CIT = False  # combine in the tail of the fused expert kernel (no combine launch)
GATHER_PLAN = True  # the gather kernel also writes the plan (one launch fewer)
MOE_SMEM_SM90 = 227 * 1024  # dynamic shared memory for the fused/persistent MoE kernels
MOE_SMEM_SM80 = 163 * 1024
MOE_SMEM_SKINNY_TEST = 100 * 1024

# --- Boundary ---
BND_RECOMPUTE = True  # second boundary rebuilds the fp32 stream from the first's RMS factor
BND_RC_MIN_T = 8192  # ...from this token count (below it, the stored stream is cheaper)
BND_CUDA_MIN = 1  # token count from which the CUDA boundary kernel replaces the Triton one

# --- Router / combine ---
ROUTER_SERIAL_MAX = 0  # T <= this: router GEMM on the main stream before the pooler
ROUTER_SERIAL_MIN = 1 << 30  # T >= this: the same (never, at the scored sizes)
ROUTER_LT = True  # False: the router GEMM always runs as torch.mm (no tuned split-K algorithm)
COMBINE_NW_SMALL = 8  # Triton combine warps at T <= COMBINE_SMALL_T
COMBINE_SMALL_T = 4096
COMBINE_PW_EARLY = tl.constexpr(False)  # Triton combine: norm weight loaded before the PDL wait
CUDA_COMBINE = False  # CUDA combine kernel (PDL): slower than Triton at the scored sizes
TRITON_PDL = False  # qk_prep and combine (Triton) launched with programmatic dependent launch

# --- Attention ---
QK_PREP_CUDA = True  # CUDA q/k prep instead of the Triton kernel (same numerics)
PREP_SPLIT_MAX_T = 1024  # stage cache-only attention tiles on the side stream during the qkv GEMM
PREP_SPLIT = {"slide": False, "full": False}  # split-staging by shape class (off: it cost time)
PREP_SPLIT_AFTER = True  # capture the side branch after the qkv GEMM node, not before it
ATTN_SK = {"slide": 0, "full_long": 0, "full_short": 0, "full_batch": 0}  # stream-K grid per class (0: off)
ATTN_SK_MAX_PAIRS = 1536  # stream-K applies only up to this many (block, pair) pairs; must match ATTN_SK_MAXPAIRS in _CUDA_SRC
ATTN_PF_MAX_T = 0  # tokens <= this: the attention kernel prefetches the o-proj weight into L2 (0: off)
ATTN_SPLIT_MODE = 1  # 0: fill two waves; 1: minimize waves / splits (wave-quantization aware)
# Own attention variant per shape class when it validates (None: time all variants
# in prepare). Variant 43 is ~4% faster than 8 on long full attention and ~3%
# slower on sliding windows, hence the split by class.
ATTN_FIXED = {"slide": 8, "full_long": 43, "full_short": 43, "full_batch": 43}
# Own-kernel variants: (consumer warpgroups, exponentials per thread per tile on
# the FMA pipe). The first element sizes the split heuristic; the GLUE launch
# table maps the variant to the kernel template parameters.
ATTN_VARIANTS = {0: (2, 0), 1: (2, 16), 8: (3, 0), 9: (3, 16), 16: (2, 0), 24: (3, 0), 43: (3, 0)}
OWN_ATTN_LOCAL = False  # local (sm_80) runs use the FP32 fallback attention unless set
ATTN_ASTAGES = 4  # must match `#define ATTN_ASTAGES` in _CUDA_SRC (it sets the staging depth)
# Dynamic shared memory for the own attention kernel: Q/K staging + per-stage
# K/V tiles (a K tile is 128 keys x 64, a V tile 72 x 128). Must match the
# kernel's staging in _CUDA_SRC.
ATTN_SMEM = 1024 + 1024 + 4096 * 3 + ATTN_ASTAGES * (8192 + 72 * 128)
ATTN_SMEM_SM80_EXTRA = 3 * 64 * 128  # sm_80 test path stages P in shared memory
ATTN_VALIDATE_TOL = 0.03  # max relative error for an own-kernel variant to validate
ATTN_Q_ROWS_PER_WARPGROUP = 8  # query rows one warpgroup owns in the split heuristic
ATTN_KEY_TILE = 128  # keys per staging tile
ATTN_MAX_SPLITS = 16  # cap on attention key-range splits
ATTN_SPLIT_PENALTY = 0.02  # cost added per extra split (the merge pass is not free)
ATTN_SPLIT_SLOT = HD + 2  # per (split, token, head) partial: 64 unnormalised O + max + sum
ATTN_SK_Q_BLOCK = 24  # stream-K: query rows per (block, pair) -- 3 warpgroups x 8 rows
FALLBACK_Q_BLOCK = 1024  # query rows per step in the FP32 fallback attention

# --- Dense GEMMs (cuBLASLt) ---
# name -> (state weight [N, K], fp32 output). The router GEMM returns both
# halves (hi | lo partial logits) in one GEMM.
LT_GEMMS = {"qkv": ("qkv_gate", False), "o": ("o", False), "router": ("router_hilo", True),
            "pooler": ("pooler", False), "unpooler": ("unpooler", False)}
LT_TOKENS = (753, 3012, 16384)  # token counts of the scored shapes; others use torch.mm
# T -> SMs left free for the router chain by sizing the pooler down (0/absent:
# torch.mm or the tuned algorithm). The router chain runs on a side stream
# during the pooler GEMM; at small T the pooler holds every SM, so the chain
# would wait for the pooler's end and land the top-8 on the critical path.
POOL_SM_RESERVE = {}
# (T, gemm) -> cuBLASLt heuristic signature (tile id, split-K, reduction,
# custom option): a fixed pick, so prepare never races near-equal algorithms.
LT_PIN = {(753, "router"): (33, 4, 2, 3), (16384, "o"): (201, 1, 0, 2),
          (16384, "unpooler"): (201, 1, 0, 2), (16384, "pooler"): (29, 1, 0, 1),
          (3012, "qkv"): (496, 1, 0, 5), (3012, "pooler"): (24, 1, 0, 7)}
# (T, gemm) -> cuBLASLt algo-66 sweep winners, best first: (layout, [algo id, tile, split-K, reduction, swizzle, custom option, stages,
# inner shape, cluster shape]); layout bit 0 = row-major descriptors (x as A), bit 1 = W pre-transposed. Each candidate is checked and
# raced against torch.mm in prepare; none kept -> the LT_PIN / tuning path below.
LT_PIN2 = {
    (753, "pooler"): [(2, [66, 32, 1, 0, 0, 7, 35, 0, 5]), (2, [66, 32, 1, 0, 0, 5, 35, 0, 2]), (2, [66, 32, 1, 0, 0, 5, 35, 0, 3])],
    (753, "unpooler"): [(2, [66, 32, 1, 0, 0, 5, 35, 0, 2]), (2, [66, 32, 1, 0, 0, 5, 35, 0, 5])],
    (753, "o"): [(0, [66, 495, 1, 0, 0, 7, 35, 0, 5])],
    (3012, "o"): [(0, [66, 189, 1, 0, 0, 2, 35, 0, 5]), (0, [66, 197, 1, 0, 0, 2, 35, 0, 5])],
    (3012, "qkv"): [(0, [66, 496, 1, 0, 0, 5, 35, 0, 2])],
    (16384, "o"): [(3, [66, 29, 1, 0, 0, 1, 35, 0, 3])],
}
LT_ALGO_TOL = 2e-2  # a tuned algorithm must match torch.mm this closely to be kept
LT_MIN_SPEEDUP = 0.97  # ...and must beat torch.mm by at least this factor to be kept
LT_WORKSPACE_BYTES = 64 << 20  # cuBLASLt workspace per stream (main + side)

# --- Self-tests (check runs only, printed from prepare) ---
SELFTEST = False
SYNTH_SM_SCALE = 0.125  # 1 / sqrt(HD), the softmax scale of the synthetic attention cases
SYNTH_QK_SCALE_FACTOR = 5.4730064863875  # qk scale factor of the synthetic attention cases

# The side stream for the router GEMM + top-8 (created in prepare). It is a
# module global, not part of the state dict, because the state holds only
# tensors and plain data.
SIDE_STREAM = None

# --------------------------------------------------------------------------------------
# Triton kernels. The CUDA kernels in _CUDA_SRC do the same jobs on the hot
# paths; these cover the boundary/combine epilogues and the q/k prep
# fallback. Shapes here are spelled with the model constants: a QKV row is
# [Q | K | V | gate], so the gate starts at GATE_OFFSET.
# --------------------------------------------------------------------------------------


@triton.jit
def _to_e4m3(x):
    """fp32 -> float8_e4m3fn bits (uint8), round-to-nearest-even, saturating at 448."""
    a = tl.abs(x)
    u = a.to(tl.int32, bitcast=True)
    normal = ((u + 0x7FFFF + ((u >> 20) & 1)) >> 20) - 960
    sub = libdevice.rint(a * 512.0).to(tl.int32)
    code = tl.where(a < 0.015625, sub, normal)
    code = tl.minimum(code, 126)
    code = tl.where(x < 0, code | 128, code)
    return code.to(tl.uint8)


@triton.jit
def _qk_prep_kernel(qkv_ptr, pos_ptr, loc_ptr, invf_ptr, q8_ptr, kc_ptr, vc_ptr, ksc_ptr, vsc_ptr,
                     ROPE: tl.constexpr, SCALED: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:  # programmatic dependent launch: let the successor launch, wait for the predecessor's results
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    row = qkv_ptr + t.to(tl.int64) * 2304  # Triton does not capture plain-int module globals
    h = tl.arange(0, 16)[:, None]  # Triton does not capture plain-int module globals
    i = tl.arange(0, 32)[None, :]  # Triton does not capture plain-int module globals
    qe = tl.load(row + h * 64 + 2 * i).to(tl.float32)  # Triton does not capture plain-int module globals
    qo = tl.load(row + h * 64 + 2 * i + 1).to(tl.float32)  # Triton does not capture plain-int module globals
    rq = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / 64.0 + 1e-5)[:, None]  # Triton does not capture plain-int module globals
    qe = qe * rq
    qo = qo * rq
    kh = tl.arange(0, 2)[:, None]  # Triton does not capture plain-int module globals
    ke = tl.load(row + 1024 + kh * 64 + 2 * i).to(tl.float32)  # Triton does not capture plain-int module globals
    ko = tl.load(row + 1024 + kh * 64 + 2 * i + 1).to(tl.float32)  # Triton does not capture plain-int module globals
    rk = tl.rsqrt((tl.sum(ke * ke, 1) + tl.sum(ko * ko, 1)) / 64.0 + 1e-5)[:, None]  # Triton does not capture plain-int module globals
    ke = ke * rk
    ko = ko * rk
    if ROPE:
        pos = tl.load(pos_ptr + t).to(tl.float32)
        ang = pos * tl.load(invf_ptr + i)
        c = libdevice.cos(ang)
        s = libdevice.sin(ang)
        qe, qo = qe * c - qo * s, qe * s + qo * c
        ke, ko = ke * c - ko * s, ke * s + ko * c
    qrow = q8_ptr + t.to(tl.int64) * 1024  # Triton does not capture plain-int module globals
    tl.store(qrow + h * 64 + 2 * i, _to_e4m3(qe))  # Triton does not capture plain-int module globals
    tl.store(qrow + h * 64 + 2 * i + 1, _to_e4m3(qo))  # Triton does not capture plain-int module globals
    loc = tl.load(loc_ptr + t).to(tl.int64)
    j = tl.arange(0, 64)[None, :]  # Triton does not capture plain-int module globals
    v = tl.load(row + 1152 + kh * 64 + j).to(tl.float32)  # Triton does not capture plain-int module globals
    if SCALED:
        ks = tl.load(ksc_ptr)
        vs = tl.load(vsc_ptr)
        ke = ke / ks
        ko = ko / ks
        v = v / vs
    tl.store(kc_ptr + loc * 128 + kh * 64 + 2 * i, _to_e4m3(ke))  # Triton does not capture plain-int module globals
    tl.store(kc_ptr + loc * 128 + kh * 64 + 2 * i + 1, _to_e4m3(ko))  # Triton does not capture plain-int module globals
    tl.store(vc_ptr + loc * 128 + kh * 64 + j, _to_e4m3(v))  # Triton does not capture plain-int module globals


@triton.jit
def _norm_gate_kernel(o_ptr, qkv_ptr, out_ptr):
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]  # Triton does not capture plain-int module globals
    j = tl.arange(0, 64)[None, :]  # Triton does not capture plain-int module globals
    o = tl.load(o_ptr + t.to(tl.int64) * 1024 + h * 64 + j).to(tl.float32)  # Triton does not capture plain-int module globals
    g = tl.load(qkv_ptr + t.to(tl.int64) * 2304 + 1280 + h * 64 + j).to(tl.float32)  # Triton does not capture plain-int module globals
    r = tl.rsqrt(tl.sum(o * o, 1) / 64.0 + 1e-5)[:, None]  # Triton does not capture plain-int module globals
    y = o * r * tl.sigmoid(g)
    tl.store(out_ptr + t.to(tl.int64) * 1024 + h * 64 + j, y.to(tl.bfloat16))  # Triton does not capture plain-int module globals


@triton.jit
def _boundary_kernel(branch_ptr, stream_ptr, gpost_ptr, alpha_ptr, beta_ptr, gpre_ptr, sout_ptr, nout_ptr,
                     HAS_GPOST: tl.constexpr, H: tl.constexpr, STORE_S: tl.constexpr = True):
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    base = t.to(tl.int64) * H
    b = tl.load(branch_ptr + base + offs).to(tl.float32)
    if HAS_GPOST:
        b = b * (1.0 + tl.load(gpost_ptr + offs).to(tl.float32))
    r1 = tl.rsqrt(tl.sum(b * b, 0) / H + 1e-8)  # Triton does not capture plain-int module globals
    s = tl.load(alpha_ptr + offs) * tl.load(stream_ptr + base + offs) + tl.load(beta_ptr + offs) * (b * r1)
    if STORE_S:
        tl.store(sout_ptr + base + offs, s)
    r2 = tl.rsqrt(tl.sum(s * s, 0) / H + 1e-5)  # Triton does not capture plain-int module globals
    tl.store(nout_ptr + base + offs, (s * r2 * (1.0 + tl.load(gpre_ptr + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _combine_kernel(y_ptr, rrow_ptr, lid_ptr, w_ptr, gs_ptr, pw_ptr, out_ptr, H: tl.constexpr, PDL: tl.constexpr = False):
    offs = tl.arange(0, H)
    if COMBINE_PW_EARLY:  # the norm weight is a constant: load it before waiting for the expert kernel
        pw = 1.0 + tl.load(pw_ptr + offs).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    if not COMBINE_PW_EARLY:
        pw = 1.0 + tl.load(pw_ptr + offs).to(tl.float32)  # BF16 weight, applied as 1 + w
    acc = tl.zeros([H], dtype=tl.float32)
    for k in tl.static_range(8):  # Triton does not capture plain-int module globals
        lid = tl.load(lid_ptr + t * 8 + k)  # Triton does not capture plain-int module globals
        if lid >= 0:
            row = tl.load(rrow_ptr + t * 8 + k)  # Triton does not capture plain-int module globals
            y = tl.load(y_ptr + row.to(tl.int64) * H + offs).to(tl.float32) * (tl.load(gs_ptr + lid) * pw)
            rr = tl.rsqrt(tl.sum(y * y, 0) / H + 1e-8)  # Triton does not capture plain-int module globals
            acc += y * (rr * tl.load(w_ptr + t * 8 + k))  # Triton does not capture plain-int module globals
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(tl.bfloat16))


# --------------------------------------------------------------------------------------
# prepare: build the per-weights state once (extension, repacked weights,
# tuned GEMM / attention choices). kernel_fn only reads this state.
# --------------------------------------------------------------------------------------


def _residual(gate):
    """Residual-stream mixing coefficients from a residual gate: the branch
    weight is beta = sigmoid(gate / 0.3), alpha = sqrt(sigmoid(-gate/0.3) * (1 + beta))."""
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha.contiguous(), beta.contiguous()


def _split_router(router):
    """Split the fp32 router weight into hi/lo BF16 halves stacked as one
    [512, 8192] GEMM: one GEMM gives both partial-logit halves."""
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return torch.cat([hi, lo], 0).contiguous()


def _fast_scales_ok(weights):
    # The fast block-scale decode is exact only for positive normal E4M3
    # scales (bytes 0x08..0x7E: no zero, subnormal, negative or NaN
    # encodings); otherwise the extension is built with the exact decode.
    return all(bool(((weights[k].view(torch.uint8) >= 0x08) & (weights[k].view(torch.uint8) <= 0x7E)).all())
               for k in ("w13_sf", "w2_sf"))


def prepare(weights):
    global SIDE_STREAM
    dev = weights["qkv_gate"].device
    st = {}
    for n in ("qkv_gate", "o", "pooler", "unpooler"):
        st[n] = weights[n].contiguous()
    st["router_hilo"] = _split_router(weights["router"].float())
    st["router_bias"] = weights["router_bias"].float().contiguous()
    st["a1"], st["b1"] = _residual(weights["attn_residual_gate"])
    st["a2"], st["b2"] = _residual(weights["ffn_residual_gate"])
    # Norm weights stay BF16 exactly as given (applied as 1 + w): the kernels form 1 + w in fp32 when they load them.
    for n in ("pre_ffn_norm", "pre_expert_norm", "post_expert_norm", "post_ffn_norm", "next_pre_attn_norm"):
        st["g_" + n] = weights[n].contiguous()
    st["inv_freq"] = (1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2, device=dev).float() / HD))).contiguous()
    st["mod"] = _build(_fast_scales_ok(weights))
    st["w13r"] = _repack(weights["w13"], weights["w13_sf"], True, KPERM)
    st["w2r"] = _repack(weights["w2"], weights["w2_sf"], False, KPERM)
    st["w13_gs"] = weights["w13_gs"].float().contiguous()
    st["w2_gs"] = weights["w2_gs"].float().contiguous()
    st["ones32"] = torch.ones(LOCAL_EXPERTS, dtype=torch.float32, device=dev)
    st["no_zero"] = torch.empty((0,), dtype=torch.int32, device=dev)
    st["no_zero_f"] = torch.empty((0,), dtype=torch.float32, device=dev)
    st["no_zero_b"] = torch.empty((0,), dtype=torch.bfloat16, device=dev)
    props = torch.cuda.get_device_properties(dev)
    st["smem"] = MOE_SMEM_SM90 if props.major >= 9 else MOE_SMEM_SM80
    if SKINNY_TEST:  # feasibility test build (MOE_SKINNY_TEST): 2 CTAs per SM
        st["smem"] = MOE_SMEM_SKINNY_TEST
    st["mod"].moe_init(st["smem"])
    st["attn_smem"] = ATTN_SMEM + (0 if props.major >= 9 else ATTN_SMEM_SM80_EXTRA)
    st["mod"].attn_init(st["attn_smem"])
    st["no_pf"] = torch.empty((0,), dtype=torch.uint8, device=dev)
    st["nsm"] = props.multi_processor_count
    st["moe_dbg"] = torch.empty((0,), dtype=torch.int64, device=dev)  # a timeline buffer here turns on item tracing
    st["moe_flags"] = 0  # experiments only
    if SIDE_STREAM is None:
        SIDE_STREAM = torch.cuda.Stream(device=dev)
    st["lt_choice"] = _init_lt(st, dev)
    torch.cuda.synchronize()
    st["attn_choice"] = _choose_attention(st, props)
    _fmha_prepare(st)
    if SELFTEST:
        try:
            _selftest(st, weights)
        except Exception as exc:  # noqa: BLE001
            print("SELFTEST failed:", repr(exc), flush=True)
        for fn in (_selftest_attn2, _selftest_boundary):
            try:
                fn(st)
            except Exception as exc:  # noqa: BLE001
                print("SELFTEST", fn.__name__, "failed:", repr(exc), flush=True)
    return st


# --------------------------------------------------------------------------------------
# Dense GEMMs: torch.mm, or a cuBLASLt algorithm tuned in prepare when it
# beats torch.mm (same library, different kernel choice). The tuner runs
# once in prepare; kernel_fn only looks the choice up.
# --------------------------------------------------------------------------------------


