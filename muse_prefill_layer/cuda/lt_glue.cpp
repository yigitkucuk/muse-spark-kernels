
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <vector>
struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  std::vector<cublasLtMatmulHeuristicResult_t> res;
};
static std::vector<LtPlan*> g_lt_plans;
static cublasLtHandle_t g_lt = nullptr;

struct Lt2Algo { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; cublasLtMatmulAlgo_t algo; int lay; size_t ws; };
static std::vector<Lt2Algo> g_lt2;
static bool lt2_set(cublasLtMatmulAlgo_t& a, int attr, int64_t v) {
  if (attr >= 7) { uint16_t x = (uint16_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS; }
  uint32_t x = (uint32_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS;
}
// cfg: CUBLASLT_ALGO_CONFIG_{ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID, INNER_SHAPE_ID, CLUSTER_SHAPE_ID}
// returns a handle, or -1 when this cuBLASLt does not offer / accept the configuration
int64_t lt2_add(int64_t m, int64_t n, int64_t k, bool out_f32, int64_t lay, std::vector<int64_t> cfg, int64_t ws_bytes) {
  if (cfg.size() != 9) return -1;
  if (!g_lt && cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return -1;
  Lt2Algo r{};
  r.lay = (int)lay;
  const bool row = lay & 1, pk = lay & 2;
  const cudaDataType_t ct = out_f32 ? CUDA_R_32F : CUDA_R_16BF;
  if (cublasLtMatmulDescCreate(&r.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return -1;
  cublasOperation_t ta = !row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N, tb = row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  if (row) {
    cublasLtMatrixLayoutCreate(&r.a, CUDA_R_16BF, m, k, k);
    cublasLtMatrixLayoutCreate(&r.b, CUDA_R_16BF, pk ? k : n, pk ? n : k, pk ? n : k);
    cublasLtMatrixLayoutCreate(&r.c, ct, m, n, n);
    cublasLtOrder_t o = CUBLASLT_ORDER_ROW;
    for (auto d : {r.a, r.b, r.c}) cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &o, sizeof(o));
  } else {
    cublasLtMatrixLayoutCreate(&r.a, CUDA_R_16BF, pk ? n : k, pk ? k : n, pk ? n : k);
    cublasLtMatrixLayoutCreate(&r.b, CUDA_R_16BF, k, m, k);
    cublasLtMatrixLayoutCreate(&r.c, ct, n, m, n);
  }
  cublasLtMatmulAlgo_t a;
  if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, (int)cfg[0], &a) != CUBLAS_STATUS_SUCCESS) return -1;
  for (int i = 1; i < 9; i++) if (!lt2_set(a, i, cfg[i]) && cfg[i]) return -1;
  if (cfg[2] > 1 && cfg[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return -1;
  uint64_t fl = 0; size_t w = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &fl, sizeof(fl), &w) != CUBLAS_STATUS_SUCCESS) return -1;
  if ((fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F || (fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)) return -1;
  cublasLtMatmulHeuristicResult_t hr{};
  if (cublasLtMatmulAlgoCheck(g_lt, r.op, r.a, r.b, r.c, r.c, &a, &hr) != CUBLAS_STATUS_SUCCESS || hr.state != CUBLAS_STATUS_SUCCESS ||
      (int64_t)hr.workspaceSize > ws_bytes) return -1;
  r.algo = a;
  r.ws = hr.workspaceSize;
  g_lt2.push_back(r);
  return (int64_t)g_lt2.size() - 1;
}
void lt2_run(int64_t h, torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out, torch::Tensor ws) {
  const Lt2Algo& r = g_lt2.at(h);
  const float alpha = 1.f, beta = 0.f;
  const void* wp = (r.lay & 2) ? wt.data_ptr() : w.data_ptr();
  const void* a = (r.lay & 1) ? x.data_ptr() : wp;
  const void* b = (r.lay & 1) ? wp : x.data_ptr();
  auto s = cublasLtMatmul(g_lt, r.op, &alpha, a, r.a, b, r.b, &beta, out.data_ptr(), r.c, out.data_ptr(), r.c, &r.algo,
                          ws.data_ptr(), (size_t)ws.numel(), c10::cuda::getCurrentCUDAStream());
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul (lt2) failed: ", (int)s);
}
int64_t lt_plan_sm(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos, int64_t sm_count) {
  if (!g_lt) TORCH_CHECK(cublasLtCreate(&g_lt) == CUBLAS_STATUS_SUCCESS, "cublasLtCreate");
  auto* p = new LtPlan();
  TORCH_CHECK(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "desc");
  if (sm_count > 0) {  // size the kernel for fewer SMs: a concurrent stream keeps the rest
    const int32_t smc = (int32_t)sm_count;
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &smc, sizeof(smc));
  }
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  // column-major views: A = W^T (K x N, ld K) used transposed, B = X^T (K x M, ld K), C = out^T (N x M, ld N)
  cublasLtMatrixLayoutCreate(&p->la, CUDA_R_16BF, K, N, K);
  cublasLtMatrixLayoutCreate(&p->lb, CUDA_R_16BF, K, M, K);
  cublasLtMatrixLayoutCreate(&p->lc, out_f32 ? CUDA_R_32F : CUDA_R_16BF, N, M, N);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  uint64_t ws = (uint64_t)ws_bytes;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p->res.resize(max_algos);
  int n = 0;
  cublasLtMatmulAlgoGetHeuristic(g_lt, p->op, p->la, p->lb, p->lc, p->lc, pref, (int)max_algos, p->res.data(), &n);
  p->res.resize(n);
  cublasLtMatmulPreferenceDestroy(pref);
  g_lt_plans.push_back(p);
  return (int64_t)g_lt_plans.size() - 1;
}
int64_t lt_plan(int64_t M, int64_t N, int64_t K, bool out_f32, int64_t ws_bytes, int64_t max_algos) {
  return lt_plan_sm(M, N, K, out_f32, ws_bytes, max_algos, 0);
}
int64_t lt_nalgo(int64_t id) { return (int64_t)g_lt_plans[id]->res.size(); }
std::vector<int64_t> lt_info(int64_t id, int64_t algo) {
  const auto& r = g_lt_plans[id]->res[algo];
  int tile = 0, splitk = 0, stages = 0, cluster = 0, aid = 0, red = 0, swz = 0, cust = 0, inner = 0;
  size_t sz = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof(cluster), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_ID, &aid, sizeof(aid), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof(swz), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &cust, sizeof(cust), &sz);
  cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof(inner), &sz);
  // [0..3] as before; [4] workspace; [5..9] algo id, reduction scheme, CTA swizzling, custom option, inner shape
  return {tile, splitk, stages, cluster, (int64_t)r.workspaceSize, aid, red, swz, cust, inner};
}
void lt_run(int64_t id, int64_t algo, torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws) {
  auto* p = g_lt_plans[id];
  const float alpha = 1.f, beta = 0.f;
  auto st = c10::cuda::getCurrentCUDAStream();
  auto s = cublasLtMatmul(g_lt, p->op, &alpha, w.data_ptr(), p->la, x.data_ptr(), p->lb, &beta, out.data_ptr(), p->lc,
                          out.data_ptr(), p->lc, &p->res[algo].algo, ws.data_ptr(), (size_t)ws.numel(), st);
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed: ", (int)s);
}
