# okc: problem=muse_decode_layer
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
# This file has been modified from the original.
"""Muse Spark decode layer: one CUDA-graph-capturable kernel_fn.

Pipeline (one decode step, T = batch * q_len tokens)
-----------------------------------------------------
1.  qkv projection (cuBLAS, the bf16_proj kernel, or a Triton split-K GEMM --
    prepare() times the three and pins the winner per (T, matrix)).
2.  Attention: the muse_attention_decode fused kernel (QK-norm,
    RoPE, FP8 cache write, split-KV FP8 attention, merge, per-head
    RMSNorm * sigmoid(gate)) for the shapes it supports; otherwise the glue's
    own QK-prep Triton kernel plus SGLang's FA3 (or a Triton split-KV
    attention fallback when SGLang is unavailable).
3.  o projection, then the first residual boundary (residual add, residual
    gate, pre-FFN norm) -- one-program-per-token kernels or parallel
    atomic-sum kernels, per T, timed in prepare().
4.  Pooler + router (bf16_proj's fused launch, its two separate launches, or
    the Triton split-K GEMM), routing (top-8 of biased sigmoid scores), and
    the pre-expert norm.
5.  Experts: the muse_moe_w4a16 NVFP4 kernel (three compile-time
    builds by token count), or SGLang's Marlin NVFP4 path, whichever
    prepare() timed faster per T.
6.  Unpooler, then the second residual boundary (post-FFN norm, next layer's
    pre-attention norm).

Precision: BF16 weights stay BF16 (or bf16_proj's lossless 12-bit encoding
with exact escape correction); NVFP4 experts stay NVFP4; norms apply
(1 + w) formed in FP32 at use; FP8 queries / FP8 KV cache as SGLang;
softmax P in f16, or e4m3 where SGLang's FA3 rounds it to e4m3 too;
FP32 accumulation throughout; activations BF16 wherever SGLang keeps BF16.

Selection: prepare() times each backend per (token count, matrix) and the
expert kernels per token count, on the benchmark's own GPU with synthetic
activations (it reads only weights, never data), then fixes the choices;
untuned token counts take the next tuned bucket's.

OKC rules: one Python file; native CUDA only via
torch.utils.cpp_extension.load_inline; CUDA-graph capturable (no host syncs
in timed paths -- prepare() and the tuners sync, kernel_fn does not); fresh
inputs each call, no caching across calls (module-level caches hold only
compiled extensions and per-device scratch, never activations).
Safety: every scratch/workspace buffer is followed by sentinel words;
prepare() checks them after the tuner and the attention self-test.
"""

import os
import threading
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

# Model shape (the task's dimensions, not tunables). The CUDA sources carry
# their own copies because they compile standalone; keep them in sync by hand.
HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
QKV_ROWS = 2304
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 64
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # 1024: q / o-projection width
ROPE_THETA = 500000.0
ROUTER_MULT = 0.1767766952966369  # router logit scale (1 / sqrt(32))
LOG2E = 1.4426950408889634

_TUNE_LOG = False
FORCE_PBMAX = None  # override the tuner's pbmax (no consumer; key kept for state shape)
# Programmatic Dependent Launch for this file's Triton glue: each kernel launches
# while its producer drains and waits (griddepcontrol.wait) right before its
# first read of producer data; weights and older inputs load before the wait.
PDL = True
PDL_MAX_T = 128  # a launch-latency win at decode sizes, a ~1% loss at T = 256


def _pdl(T):
    return PDL and T <= PDL_MAX_T


# The native NVFP4 expert kernel wins at every tuned token count; the tuner may only
# depart from it when another backend is clearly faster, so a noisy measurement
# cannot flip a pick.
DEFAULT_MOE = "native"
TUNE_MARGIN = 0.95
# Dense picks pinned to the winners measured in the whole layer: the isolated
# tuner sometimes gets them wrong because it cannot see PDL overlap in the chain.
PIN_PICK = {(1, "un"): "bfp", (8, "un"): "bfp", (8, "o"): "bfp", (32, "o"): "bfp", (32, "pr"): "bfp",
            (128, "qkv"): True, (256, "qkv"): True}
FORCE_GLUEV2 = None  # None: tuner's choice; True/False: force the parallel boundary/router kernels


ATT7_CUDA_SRC = __cuda__("att7.cu")
ATT7_CPP_SRC = r"""
void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl,
         int64_t tma, int64_t pf_ptr, int64_t pf_bytes);
"""
_ATT7 = None


def _att7():
    global _ATT7
    if _ATT7 is None:
        import os
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major >= 9 else "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major == 9 else "")
        _ATT7 = load_inline(name=f"muse_att_pv2_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,
                            functions=["run"], extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"], verbose=False)
    return _ATT7


_WS7 = {}
_PF7 = None
_NUM_SMS7 = {}
_WS_ROW7 = 72


def _workspace7(n_rows, device):
    """Persistent FP32 accumulators (zero between calls: the last CTA resets them), 72-float rows,
    each followed by sentinel words -- see _guarded / _check_guards."""
    key = str(device)
    ws = _WS7.get(key)
    if ws is None or ws[0].numel() < n_rows * _WS_ROW7:
        n = max(n_rows, 1 << 15)
        ws = tuple(_guarded(t) for t in (
            torch.zeros(n * _WS_ROW7, dtype=torch.float32, device=device),
            torch.zeros(n, dtype=torch.float32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device)))
        _WS7[key] = ws
    return ws


def _num_sms7(device):
    key = str(device)
    if key not in _NUM_SMS7:
        _NUM_SMS7[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS7[key]


_TUNED7 = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}


OKC_PDL = 0


def _plan7(B, pairs, L, window, max_pages, P, device):
    """Launch plan for a shape: n_splits CTAs per (request, KV pair) or, flat, per KV pair over all requests; nc workspace copies."""
    sms = _num_sms7(device)
    target = 2 * sms
    splits = lambda tiles, quota: max(1, min(tiles, quota))
    copies = lambda n, cap: max(1, min(n, cap))
    if not window and 1 < B <= 512:
        return dict(n_splits=max(1, target // pairs), flat=1, cs=0, ncl=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + 63) // 64)
    reqs = B * pairs
    if not window:
        if L == 8:
            n = splits(tiles, ((target * 21) // 22) // reqs)
            return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=copies(n, 8))
        n = splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=copies(n, 4), coop=0)
    if L == 1 and reqs < 24:
        n = splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    if L == 8 and reqs < 16:
        n = splits(tiles, target // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if L == 8 and reqs >= 16:
        n = splits(tiles, (target if reqs >= 24 else sms) // reqs)
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    n = splits(tiles, target // reqs)
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)


