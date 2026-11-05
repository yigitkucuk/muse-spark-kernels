muse-spark-okc
==============

CUDA kernels for one Muse Spark transformer layer on H200 (sm_90a): decode, prefill, and the three components
they are built from (routed experts, attention decode, dense projections).

  muse_decode_layer      full layer step at decode / verify sizes, T = 1..256 tokens
  muse_prefill_layer     full layer step over a prefill batch
  muse_moe_w4a16         routed experts: NVFP4 weights, BF16 activations, one persistent kernel
  muse_attention_decode  gated attention over a paged FP8 KV cache, one fused kernel
  muse_bf16_proj         the five dense projections (qkv_gate, o, pooler, unpooler, FP32 router)

Each kernel is a single Python module with two entry points:

  prepare(weights) -> state    compile the extensions, repack weights (lossless only), time the candidate
                               paths on this GPU and pin the fastest per shape
  kernel_fn(data, state)       one call on fresh inputs; CUDA-graph capturable, no host syncs, no caching

Layout
------
  <kernel>/py/NN_*.py     consecutive sections of the module, in order
  <kernel>/cuda/*.cu|cpp  CUDA / C++ sources, compiled at runtime with torch.utils.cpp_extension.load_inline;
                          referenced from the Python as __cuda__("name")
  <kernel>/DESIGN.txt     algorithm, sections, sources
  build.py                joins the sections and inlines the sources into build/<kernel>.py

Build
-----
  python3 build.py            # all five -> build/
  python3 build.py --check    # also verify each output's sha256

Requirements: Hopper GPU (sm_90a), CUDA 12.8+, PyTorch, Triton. When sgl_kernel / SGLang are installed,
muse_decode_layer also times their FA3 attention and Marlin experts, and muse_prefill_layer their FA3 attention;
without them the native kernels run.

Numerics, all kernels: BF16 weights stay exact (dense paths may store them in a lossless 12-bit code with escape
correction); NVFP4 experts are repacked, never requantized; FP32 accumulation; FP8 E4M3 KV cache and queries;
router in FP32 or as an exact BF16 hi + lo pair.
