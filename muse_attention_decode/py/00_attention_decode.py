# okc: problem=muse_attention_decode
# This file has been modified from the original.
"""Gated attention decode for Muse Spark Open: one fused CUDA kernel per call.

Algorithm
---------
For each request, the kernel computes gated attention over the request's cached
keys/values plus the block's own new tokens:

    O = rmsnorm(softmax(Q K^T * sm_scale) V) * sigmoid(gate)

with Q/K first RMS-normalized (Q additionally scaled by qk_scale_factor), RoPE
applied on sliding-window layers, and K/V quantized to FP8 (E4M3) in the
cache. QK^T runs on FP8 tensor-core MMAs (queries rounded to E4M3, as FA3
does) and PV on FP16 MMAs (P in FP16, V exact in FP16) -- except q_len 8 on
sm_90, where PV is an FP8 wgmma (P in E4M3, as SGLang's FA3 FP8 path) -- with
an online softmax, so no full score matrix is ever materialized.

Design
------
* One CTA per (request, pair of KV heads, split-KV split). A "pair" is two KV
  heads, i.e. 16 query heads: the kernel's unit of work.
* Each CTA builds the block's new K/V (normalized, RoPE, FP8) and writes them
  to the cache, then streams its range of the paged FP8 cache through shared
  memory in 64-key tiles (cp.async, multi-stage pipeline).
* Splits are merged with FP32 atomics after rescaling to a fixed reference
  (QK-norm bounds the logits); the last CTA of each (request, KV pair)
  normalizes, applies rmsnorm * sigmoid(gate), writes the output, then zeroes
  the workspace for the next call.
* Launch shapes: per-request split-KV, per-request clusters (DSMEM
  reduction), a flat kernel (one CTA pass over all requests' keys, for full
  layers), and a cooperative finalize (every CTA finalizes a slice) for grids
  known resident. The host picks one in _plan(); the kernel itself is built
  once per launch mode (the MODE template parameter), so each build carries
  only its own epilogue.
* Fixed build: sm_90a only. There are no tuning switches left in the
  CUDA source; the taken path is the only path.
* CUDA-graph capturable: no host syncs in the timed path. The module-level
  caches hold only device properties, derived constants (RoPE frequencies),
  and scratch buffers (the workspace is zeroed by the kernel itself between
  calls), never call inputs or results.

Entry point: kernel_fn(data) -> output tensor. Shapes the CUDA path does not
cover fall back to _fallback(), a plain PyTorch implementation of the same
math.

CUDA layout: Cfg/Params/Smem (launch config, kernel
arguments, shared-memory layout), small device helpers (cp.async, mbarrier,
wgmma, conversions, cluster/DSMEM), the per-segment main loop (segment()),
the finalizes (sum_copies, fin_nc1, flat_finalize), attn_kernel, then the
host launchers (launch, run) at the end.
"""

from typing import TypedDict

import torch


# ---- shapes the kernel supports (see _supported) ----
HEAD_DIM = 64            # head_dim of the supported layout
Q_HEADS_PER_PAIR = 16    # query heads per KV-head pair (the kernel's unit of work)
SUPPORTED_Q_LENS = (1, 8)
MAX_BATCH = 512
MAX_REQUEST_PAIRS = 4096  # batch * (num_kv_heads // 2)

# ---- indexing ----
TILE_KEYS = 64            # keys per pipeline tile
INDEX_LIMIT = 1 << 31    # 32-bit indexing bound checked by _fits

# ---- scratch buffers ----
WORKSPACE_MIN_ROWS = 1 << 15  # _workspace grows to at least this many rows
WS_ROW = 72  # floats per workspace row (the q_len 1 per-request kernels pad their rows to 72; the others use 64)

LOG2E = 1.4426950408889634  # log2(e), to full double precision

# ---- packed kernel-mode bitfield (the `tma` argument to run) ----
# bit 1: cooperative finalize; bit 16: PDL.
COOP_BIT = 1   # bit 1: cooperative finalize
PDL_BIT = 16   # bit 16: programmatic dependent launch

# ---- per-(request, KV pair) launch grid: the per-request grids run slightly
# below the CTA target, leaving headroom for the finalize tail ----
PLAN_HEADROOM_NUM = 21
PLAN_HEADROOM_DEN = 22


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    # scheduler_metadata, cache_seqlens and cu_seqlens_q are carried for the
    # harness interface; the CUDA path uses seq_lens / cache_loc / page_table
    # instead, and positions feeds only the fallback.
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


CUDA_SRC = __cuda__("attention.cu")


CPP_SRC = r"""
void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl,
         int64_t tma);
"""


_MOD = None
_INV_FREQ = {}
_WS = {}
_NUM_SMS = {}


def _cached(cache, key, make):
    """Memoized builder for device properties and small derived tensors."""
    value = cache.get(key)
    if value is None:
        value = make()
        cache[key] = value
    return value


# Per-shape launch overrides measured on the benchmark's H200:
# (batch, q_len, sliding, kv pairs) -> plan fields.
_TUNED = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}


def _build(tag="", flags=()):
    # Fixed build: sm_90a only.
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name=f"okc_attn_v298bulk{tag}_90a",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["run"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a",
                           "--expt-relaxed-constexpr", *flags],
        verbose=False,
    )


def _module():
    global _MOD
    if _MOD is None:
        _MOD = _build()
    return _MOD


def _inv_freq(theta, device, n_pairs=HEAD_DIM // 2):
    """RoPE inverse frequencies, 1 / theta^(2i / head_dim), one per pair."""

    def make():
        if theta:
            t = 1.0 / (theta ** (torch.arange(0, 2 * n_pairs, 2, device=device).float() / (2 * n_pairs)))
        else:
            t = torch.zeros(n_pairs, device=device)
        return t.float().contiguous()

    return _cached(_INV_FREQ, (float(theta), str(device), n_pairs), make)


def _workspace(n_rows, device):
    """Persistent FP32 accumulators (zero between calls: the last CTA resets them)."""
    key = str(device)
    ws = _WS.get(key)
    if ws is None or ws[0].numel() < n_rows * WS_ROW:
        n = max(n_rows, WORKSPACE_MIN_ROWS)
        ws = (torch.zeros(n * WS_ROW, dtype=torch.float32, device=device),
              torch.zeros(n, dtype=torch.float32, device=device),
              torch.zeros(n, dtype=torch.int32, device=device))
        _WS[key] = ws
    return ws


def _num_sms(device):
    return _cached(_NUM_SMS, str(device), lambda: torch.cuda.get_device_properties(device).multi_processor_count)


def _splits(tiles, quota):
    """Splits per (request, KV pair): at least 1, at most one per tile."""
    return max(1, min(tiles, quota))


def _copies(n, cap):
    """Workspace copies: at least 1, at most cap."""
    return max(1, min(n, cap))


def _headroom(target):
    """The per-request grids run slightly below the CTA target."""
    return (target * PLAN_HEADROOM_NUM) // PLAN_HEADROOM_DEN


def _plan(B, pairs, L, window, max_pages, P, device):
    """Launch plan for a shape: n_splits CTAs per (request, KV pair) or, flat, per KV pair over all
    requests; clusters of cs CTAs reduce through DSMEM (ncl clusters per request); nc workspace copies."""
    sms = _num_sms(device)
    target = 2 * sms
    if not window and 1 < B <= MAX_BATCH:
        # Full layer, several requests: equal shares of all requests' keys, one CTA pass per SM pair.
        return dict(n_splits=max(1, target // pairs), flat=1, cs=0, ncl=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + TILE_KEYS - 1) // TILE_KEYS)
    reqs = B * pairs
    if not window:
        # One long request: plain split-KV over 2 CTAs per SM (clusters slower here).
        if L == 8:
            n = _splits(tiles, _headroom(target) // reqs)
            return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=_copies(n, 8))
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=_copies(n, 4), coop=0)
    if L == 1 and reqs < 24:
        # Few requests, sliding: plain split-KV with one tile per CTA and a single workspace copy.
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    if L == 8 and reqs < 16:
        # q_len 8, few requests: plain split-KV with one tile per CTA and the cooperative finalize
        # (every CTA normalizes a slice of the rows) beats clusters.
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if L == 8 and reqs >= 16:
        # q_len 8, many requests: plain split-KV with about one CTA per SM, plus the cooperative finalize.
        n = _splits(tiles, (target if reqs >= 24 else sms) // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if reqs >= 24:
        # Many requests: clusters no longer fit one CTA per SM, so plain split-KV spreads evenly.
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    # Sliding window: one cluster per (request, KV pair); keep the grid small enough that all
    # clusters are resident at once.
    cs = 16
    limit = 2 * sms if L == 1 else sms
    if L == 8 and target == sms:
        limit = (sms * 15) // 16   # one CTA per SM: clusters of 8 fit 15 per 132 SMs, so stay below
    while cs > 4 and reqs * cs > limit:
        cs //= 2
    ncl = 1
    while cs * ncl > tiles and cs > 4:
        cs //= 2
    if cs * ncl > tiles or cs < 4:
        n = _splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=_copies(n, 4))
    return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=_copies(ncl, 4))


def _launch(data, dbg=None, _mod=None, pdl=None, **override):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    qkv = data["qkv"]
    device = qkv.device
    pairs = hkv // 2
    window = cfg["window"]
    page_table = data["page_table"]
    out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
    plan = _plan(B, pairs, L, window, page_table.shape[1], P, device)
    tuned = _TUNED.get((B, L, bool(window), pairs))
    if tuned and not override:
        override = tuned
    if override:
        plan = dict(plan, **override)
        if "cs" in override and override["cs"] > 1:
            plan["flat"] = 0
            if "ncl" not in override:
                plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
            plan["n_splits"] = plan["cs"] * plan["ncl"]
            if "nc" not in override:
                plan["nc"] = _copies(plan["ncl"], 4)
        elif "n_splits" in override and "nc" not in override:
            plan["nc"] = 1 if plan["flat"] else _copies(plan["n_splits"], 4)
    n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
    # Cooperative finalize (all CTAs resident): only for the long single-request grids, whose last-CTA
    # finalize reads and zeroes tens of KB from one SM; the short sliding grids finalize faster in one CTA.
    if flat:   # grid-wide finalize after all segments (the kernel re-checks residency)
        coop = int(plan.get("coop", int(L == 8 and
                                        n_splits * pairs <= 2 * _num_sms(device))))
    else:
        coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else
                            int(n_splits * B * pairs <= 2 * _num_sms(device))))
    use_pdl = bool(pdl)
    # tma packs the kernel-mode bitfield: bit 1 = cooperative finalize, bit 16 = PDL.
    tma = int(plan.get("tma", 0)) | (coop << COOP_BIT) | (int(use_pdl) << PDL_BIT)
    if flat or cs <= 1:
        cs, ncl = 0, 1
    ws_acc, ws_l, ws_cnt = _workspace(B * pairs * nc * L * Q_HEADS_PER_PAIR, device)
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    (_mod or _module()).run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                  data["k_scale"], data["v_scale"], _inv_freq(cfg["rope_theta"], device), out, ws_acc, ws_l,
                  ws_cnt, B, L, hq, hkv, P, window, 1 if cfg["rope_theta"] else 0, n_splits, flat, nc, scale_log2,
                  dbg, cs, ncl, tma)
    return out


def _supported(cfg):
    P = cfg["page_size"]
    return (cfg["head_dim"] == HEAD_DIM and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
            and cfg["q_len"] in SUPPORTED_Q_LENS and cfg["batch"] <= MAX_BATCH and P > 0 and (P & (P - 1)) == 0
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= MAX_REQUEST_PAIRS)


def _rms(x, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope(x, pos, theta):
    d = x.shape[-1]
    inv_freq = _inv_freq(theta, x.device, d // 2)
    ang = pos.float()[..., None] * inv_freq
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def _fallback(data):
    """Plain PyTorch path for shapes the CUDA kernel does not cover (other q_len / head layouts)."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    q_in, k_in, v_in, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)
    pos = data["positions"].view(batch, length)
    q = _rms(q_in.reshape(batch, length, hq, d)) * cfg["qk_scale_factor"]
    k = _rms(k_in.reshape(batch, length, hkv, d))
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    q = q.to(torch.bfloat16)
    table = data["page_table"].long()
    slots = data["cache_loc"].view(batch, length)
    k_cache = data["k_cache"].view(-1, hkv, d)
    v_cache = data["v_cache"].view(-1, hkv, d)
    k_scale, v_scale = data["k_scale"], data["v_scale"]
    # The new K/V land in the cache, exactly as the CUDA path writes them.
    k_cache[slots] = (k / k_scale).to(torch.float8_e4m3fn)
    v_cache[slots] = (v_in.reshape(batch, length, hkv, d).float() / v_scale).to(torch.float8_e4m3fn)
    t = torch.arange(table.shape[1] * page, device=q.device)
    out = torch.empty((batch, length, hq, d), dtype=torch.bfloat16, device=q.device)
    for b in range(batch):
        rows = table[b].clamp(min=0)[t // page] * page + t % page
        keys = k_cache[rows].float() * k_scale
        vals = v_cache[rows].float() * v_scale
        visible = t[None, :] <= pos[b][:, None]
        if cfg["window"]:
            visible &= t[None, :] > pos[b][:, None] - cfg["window"]
        att = torch.einsum("lkgd,skd->lkgs", q[b].view(length, hkv, hq // hkv, d).float(), keys) * cfg["sm_scale"]
        att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
        out[b] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(length, hq, d)
    out = _rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, d)
    return out.reshape(batch, length, hq * d).to(torch.bfloat16)


def _fits(data):
    """Sizes the CUDA path's 32-bit indexing covers (caches of any size are fine: 64-bit offsets past 4 GB)."""
    cfg = data["config"]
    out_elems = cfg["batch"] * cfg["q_len"] * cfg["num_q_heads"] * cfg["head_dim"]
    return (data["qkv"].numel() < INDEX_LIMIT and data["page_table"].numel() < INDEX_LIMIT
            and out_elems < INDEX_LIMIT)


def _base_kernel_fn(data: Inputs) -> torch.Tensor:
    if not _supported(data["config"]) or not _fits(data):
        return _fallback(data)
    return _launch(data)


def kernel_fn(data: Inputs) -> torch.Tensor:
    return _base_kernel_fn(data)
