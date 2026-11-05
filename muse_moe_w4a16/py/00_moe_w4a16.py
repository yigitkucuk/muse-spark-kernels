# okc: problem=muse_moe_w4a16
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
# This file has been modified from the original.
"""Routed experts (NVFP4 W4A16 MoE) as one persistent CUDA kernel.

Algorithm
---------
For each token, up to ``TOP_K`` of the 256 experts are active, and this rank
owns ``LOCAL_EXPERTS`` of them. The work is a gated MLP per expert:

1.  Routing (``route`` in the CUDA source): every CTA redundantly builds the
    same per-expert token lists from ``topk_ids`` -- counts, offsets, and a
    sorted ``(token, slot)`` entry list -- plus a job list. An expert with
    more than ``SKINNY_MAX_TOKENS`` tokens becomes one or more *fat* jobs
    (chunks of ``FAT_MAX_TOKENS`` tokens); the rest become *skinny* jobs.
    Fat jobs are listed first, largest first.
2.  Phase A (gate + up, ``GEMM1 + SwiGLU``): each job's tokens are
    multiplied by the expert's packed NVFP4 gate/up weights. A *skinny* unit
    is 16 rows x full K with warps splitting K and ``mma.sync``; a *fat*
    unit is 256 rows x full K streamed through shared memory in K-stages.
    The SwiGLU output lands in ``inter`` (bf16).
3.  Phase B (down, ``GEMM2``): the same jobs run again with the expert's
    down weights, reading ``inter``. The epilogue multiplies by
    ``1 + post_norm_w`` (the post-expert RMSNorm weight, kept bf16 as given)
    and writes fp32 rows to ``ybuf`` plus each route's sum of squares to
    ``ss`` (double-buffered per launch generation).
4.  Final pass: ``out[t] = sum_s topk_w[t, s] * rsqrt(ss[pos] / HIDDEN + eps)
    * ybuf[pos]``. A grid barrier between the phases and the final pass
    makes every CTA's phase-B writes visible to every CTA.

Scheduling: one persistent grid (one CTA per SM) runs both phases. Fat
units go first (round-robin over CTAs); skinny units are split by cost so
every CTA carries the same total cost, with a small run-time pool of tail
units claimed atomically (stream-K). With no fat jobs, phase A's last ring
refills already prefetch this CTA's first phase-B units (cross-phase
prefill).

Which kernel runs depends on the batch: ``kernel_fn`` picks one of four
compile-time variants of the same source (``MAIN`` / ``TINY`` / ``TINY2`` /
``MID``), selected by token count; the large-batch variant additionally
switches to an RS-wgmma implementation (``OKC_WGS``) for the main band.
Weights are only repacked (bit/byte permutations) in ``prepare`` -- never
requantized -- into the four layouts the kernel reads: packed
mma.sync order, fat order, unit-contiguous order, and natural-k / WGS
order.

Constraints the code preserves (OKC rules)
------------------------------------------
* One Python file; native CUDA only, via ``torch.utils.cpp_extension.
  load_inline``.
* CUDA-graph capturable: no host synchronisation in the timed path.
  ``kernel_fn`` allocates a fresh ``out`` every call, reads no tensor
  back to host, and launches only ``okc_run``. Host syncs exist only in
  ``prepare`` (one-time setup) and in the ``DIAG`` diagnostics, both
  outside the timed path.
* Fresh inputs each call, no caching across calls: the only
  module-level cache is ``_EXTS``, which maps a compile-time define
  tuple to its compiled extension module. It caches compiled code,
  never activations, weights, or outputs.
"""

from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


# ---------------------------------------------------------------- model shape
# These are the task's model dimensions, not tunables. The CUDA source has
# its own copies (HD / ID) because it is compiled standalone; keep them in
# sync by hand -- there is no mechanism to share a constant across the
# Python/CUDA boundary in a load_inline submission.
HIDDEN = 4096            # d_hidden and d_expert (the MoE is square)
TOP_K = 8                # routes per token
N_EXPERTS_TOTAL = 256    # experts across all ranks
LOCAL_START = 0          # first expert owned by this rank
LOCAL_EXPERTS = 32       # experts owned by this rank (ep8 sharding)
SHARDING = "ep8"

# ------------------------------------------------------- per-call chunking
# kernel_fn runs the kernel in slices of at most this many tokens: the
# kernel's scratch (inter / ybuf / ss) is sized for this bound.
CHUNK_TOKENS = 384
EMAX = CHUNK_TOKENS * TOP_K          # max (token, route) entries per chunk
JMAX = 32 + EMAX // 64 + 8           # max jobs per chunk (upper bound)

# ------------------------------------------------------------- WGS scratch
# Stream-K fixup area: mirror of the CUDA-side WGS_FIXF / WGS_XROWS.
WGS_SLOTS = 160                      # fixup slots per phase (>= grid + 1)
WGS_FIXW = 4096                      # floats per (slot, warpgroup) fixup
WGS_FIXF = 2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16
WGS_XROWS = EMAX + 8 * (32 + EMAX // 32 + 8)

# ------------------------------------------- compile-time variant selection
# The four variants are the same CUDA source compiled with different
# -D flags -- i.e. different tile / scheduling choices, not different code.
# Flag tuples are part of the submission's tuned behaviour; do not edit
# them without re-gating on GPU. kernel_fn picks by token count M.
MAIN_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1")
# M <= TINY_MAX_M: skinny-only build; a job never has more than M tokens,
# so there are no fat units.
TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2",
                "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1",
                "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1")
TINY_MAX_M = 4
# TINY_MAX_M < M <= TINY2_MAX_M: a second skinny-only build with a shorter
# helper-warp tail.
TINY2_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_CPF=0", "-DOKC_EPI4=0",
                 "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1",
                 "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=1", "-DOKC_FINPF=1")
TINY2_MAX_M = 8
# TINY2_MAX_M < M <= MID_MAX_M: tiny-style build with experts of more than
# SKINNY_MAX_TOKENS tokens split into <= SKINNY_MAX_TOKENS-token jobs.
MID_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0",
               "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0",
               "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2",
               "-DOKC_FINPF=1")
MID_MAX_M = 16
# M 17..63 runs on the WGS MAIN build; M >= 64 is MAIN by default below.
# The skinny ring's slot count when the caller does not pin -DOKC_NS.
MAIN_NS = 3

# The M >= 64 build used when the scale bytes rule out the PRMT decode
# (the WGS kernel needs it): the mma.sync kernel with the same shapes.
MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7",
                         "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0")

# ------------------------------------------------------------- feature flags
# Programmatic dependent launch for embedders whose graph runs this kernel
# right after other kernels (the kernel waits on griddepcontrol.wait before
# touching their outputs). Off: one launch per call gains nothing.
PDL = False
# Scale decode with PRMT + IMAD instead of the smem LUT; exact only when
# every block-scale byte of the local experts is a positive normal e4m3
# value (0x08..0x7E). prepare() checks the actual weights and keeps the
# LUT build otherwise.
PRMT_SCALES = True
# DQ7: the bit layout whose decode is 7 ALU ops + 2 IMAD.SHL per 8 weights
# (the kernel gets -DOKC_DQ7=1 via _ext).
DQ7 = True

# SC6: lossless 6-bit block scales (checked on the actual weights in prepare(): every byte in 0x40..0x7F)
SC6_WGS = True      # WGS stages (bs >= 17)
SC6_SKINNY = True      # skinny units (TINY / TINY2 / MID builds)
SC6_ATTRS = True    # prepare() prints registers / local bytes of every build

# ---------------------------------------------------------------- diagnostics
# All off in a submission run. DIAG makes prepare() print an H200 timing
# table (untimed); DIAG_BW additionally runs the BW_SRC microbenchmarks.
DIAG = False
DIAG_VARIANTS = {}         # name -> defines, timed in diag mode
DIAG_BW = False
DIAG_BW_MODES = (5, 8, 12, 16) + tuple(range(17, 25))

# Extension-point for out-of-tree experiments: "@name" in a defines tuple
# swaps the compiled source to ALT_SRCS[name]. Empty in a submission.
ALT_SRCS = {}

# Bit layout of a 32-bit word holding 8 E2M1 codes for one MMA k-step of
# one lane: register R (0..3) is (h, j0) = (0,0) (1,0) (0,2) (1,2): rows
# g+8h, k = base + 4s + j0 (+1 for the upper bf16). Code bits (m, e0, e1, s)
# land at output bits (6, 7, 8, 15) (+16 for the upper half) after the
# per-register rotation used by dq4().
_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


CUDA_SRC = __cuda__("moe.cu")

# CUDA source of the small builds (TINY / TINY2 / MID): the base source plus this variant's patches
VAR_SRC = __cuda__("moe_var.cu")


BW_SRC = __cuda__("moe_bw.cu")

CPP_SRC = r'''
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
std::vector<int64_t> okc_attrs();
'''


# ------------------------------------------------------------------ extension
# Compiled extensions, keyed by their define tuple. This caches compiled
# code only -- never weights, activations, or outputs.
_EXTS = {}


