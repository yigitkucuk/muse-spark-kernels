def _build(fast_scales=True):
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}{cap[1]}" + ("a" if cap[0] == 9 else "")
    src = ("" if fast_scales else "#define MOE_FAST_SCALE 0\n") + f"#define MOE_KPERM {int(KPERM)}\n" + _CUDA_SRC + GLUE
    name = "okc_prefill_moe_" + hashlib.sha1((src + CPP + arch).encode()).hexdigest()[:16]
    return load_inline(name, cpp_sources=CPP, cuda_sources=src,
                       functions=["qk_prep", "moe_route", "moe_plan", "moe_gather", "moe_gather_plan", "moe_combine",
                                  "moe_gemm", "moe_init", "moe_gemm_p", "moe_fused", "boundary", "boundary_rc",
                                  "kv_gather", "attn_prep", "attn_fwd", "attn_init", "attn_set_prof"],
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-std=c++20",
                                          "--use_fast_math"])


def _build_lt():
    name = "okc_prefill_lt_" + hashlib.sha1((LT_GLUE + LT_CPP).encode()).hexdigest()[:16]
    # Host-only code: compiled by the host compiler (no nvcc), linked against cuBLASLt.
    return load_inline(name, cpp_sources=[LT_CPP, LT_GLUE],
                       functions=["lt_plan", "lt_plan_sm", "lt_nalgo", "lt_info", "lt_run", "lt2_add", "lt2_run"],
                       extra_cflags=["-O3", "-std=c++20"], extra_ldflags=["-lcublasLt"], with_cuda=True)


CPP = __cuda__("prefill_bindings.cpp")

GLUE = __cuda__("glue.cu")

_CUDA_SRC = __cuda__("prefill.cu")

LT_CPP = r'''
#include <torch/extension.h>
int64_t lt_plan(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos);
int64_t lt_plan_sm(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos, int64_t sm_count);
int64_t lt_nalgo(int64_t id);
std::vector<int64_t> lt_info(int64_t id, int64_t algo);
int64_t lt2_add(int64_t m, int64_t n, int64_t k, bool out_f32, int64_t lay, std::vector<int64_t> cfg, int64_t ws_bytes);
void lt2_run(int64_t h, torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out, torch::Tensor ws);
void lt_run(int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws);
'''

LT_GLUE = __cuda__("lt_glue.cpp")


