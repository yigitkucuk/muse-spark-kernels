
#include <cuda_runtime.h>
using namespace okc_moe;

template <typename... KArgs, typename... Args>
static inline void okc_launch(void (*k)(KArgs...), dim3 g, dim3 b, size_t smem, cudaStream_t st, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k, ((KArgs)args)...);
}
void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                 int T, cudaStream_t st, int* tok_left, int* cq) {
  okc_launch(moe_route_kernel, dim3((T + ROUTE_WARPS - 1) / ROUTE_WARPS), dim3(ROUTE_WARPS * 32), 0, st, logits, bias, lstart, ids, w, lid, pos, cnt, T,
             tok_left, cq);
}
void k_moe_plan(const int* counts, int* offsets, int* tiles, int* ntiles, int* rows, int order, cudaStream_t st) {
  okc_launch(moe_plan_kernel, dim3(1), dim3(32), 0, st, counts, offsets, (int4*)tiles, ntiles, rows, order);
}
static int g_attn_merge_host = 0;
void k_moe_gather(const void* p, const void* gw, const int* lid, const int* pos, const int* offsets, const int* counts,
                  void* xt, int* rrow, int T, cudaStream_t st) {
#define MOE_PADFILL 1
  okc_launch(moe_gather_kernel<false>, dim3(T + (MOE_PADFILL ? 32 * MOE_PADFILL_PARTS : 0)), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, offsets, counts, (uint8_t*)xt, rrow, T, (int*)nullptr, (int4*)nullptr, (int*)nullptr, (int*)nullptr, 0,
             (int*)nullptr, (int*)nullptr, (int*)nullptr);
}
void k_moe_gather_plan(const void* p, const void* gw, const int* lid, const int* pos, const int* counts, void* xt, int* rrow,
                       int* offsets, int* tiles, int* ntiles, int* rows, int order, int T, cudaStream_t st,
                       int* row_tok, int* cq, int* qtail) {
  okc_launch(moe_gather_kernel<true>, dim3(T + (MOE_PADFILL ? 32 * MOE_PADFILL_PARTS : 0)), dim3(256), 0, st, (const __nv_bfloat16*)p, (const __nv_bfloat16*)gw, lid, pos, (const int*)nullptr, counts, (uint8_t*)xt, rrow, T,
             offsets, (int4*)tiles, ntiles, rows, order, row_tok, cq, qtail);
}
void k_moe_combine(const void* y, const int* rrow, const int* lid, const float* topw, const void* pw, void* out, int T,
                   cudaStream_t st) {
  okc_launch(moe_combine_kernel, dim3(T), dim3(256), 0, st, (const __nv_bfloat16*)y, rrow, lid, topw, (const __nv_bfloat16*)pw, (__nv_bfloat16*)out);
}
static Args make_args(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out,
                      int NT, int smem, int* counter) {
  Args a;
  a.wrep = (const uint8_t*)wrep; a.xt = (const uint8_t*)xt; a.tiles = (const int4*)tiles; a.ntiles = ntiles;
  a.gscale = gscale; a.out = (uint8_t*)out; a.NT = NT; a.smem_bytes = smem; a.counter = counter; a.flags = 0; a.mul16 = 16u; a.mul64 = 64u;
  return a;
}
void k_moe_gemm(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                int P, bool gated, int smem, cudaStream_t st) {
  Args a = make_args(wrep, xt, tiles, ntiles, gscale, out, NT, smem, nullptr);
  dim3 grid((unsigned)NT, (unsigned)P);
  if (gated) moe_gemm_kernel<true><<<grid, 384, smem, st>>>(a);
  else moe_gemm_kernel<false><<<grid, 384, smem, st>>>(a);
}
void k_moe_gemm_p(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                  int nctas, bool gated, int smem, int* counter, cudaStream_t st) {
  Args a = make_args(wrep, xt, tiles, ntiles, gscale, out, NT, smem, counter);
  if (gated) moe_gemm_persistent<true><<<(unsigned)nctas, 384, smem, st>>>(a);
  else moe_gemm_persistent<false><<<(unsigned)nctas, 384, smem, st>>>(a);
}
#define BND_TPB 512
#define BND_HINT false
template <bool R, bool S, typename PT, typename LT>
static void qk_prep_launch(const void* qkv, const void* pos, const void* loc, const float* invf, void* q8, void* kc, void* vc,
                           const float* ksc, const float* vsc, int T, cudaStream_t st) {
  okc_launch(qk_prep_kernel<R, S, PT, LT>, dim3(T), dim3(128), 0, st, (const __nv_bfloat16*)qkv, (const PT*)pos, (const LT*)loc, invf,
             (uint8_t*)q8, (uint8_t*)kc, (uint8_t*)vc, ksc, vsc);
}
void k_qk_prep(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* invf, void* q8, void* kc,
               void* vc, const float* ksc, const float* vsc, int T, bool rope, bool scaled, cudaStream_t st) {
#define QKP(R, S) \
  if (pos64 && loc64) qk_prep_launch<R, S, long long, long long>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st); \
  else if (pos64) qk_prep_launch<R, S, long long, int>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st); \
  else if (loc64) qk_prep_launch<R, S, int, long long>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st); \
  else qk_prep_launch<R, S, int, int>(qkv, pos, loc, invf, q8, kc, vc, ksc, vsc, T, st);
  if (rope && !scaled) { QKP(true, false) }
  else if (rope) { QKP(true, true) }
  else if (!scaled) { QKP(false, false) }
  else { QKP(false, true) }
#undef QKP
}
void k_boundary(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                const void* gpre, float* sout, void* nout, int T, bool has_gpost, cudaStream_t st, int* zp, int zn) {
  if (has_gpost)
    okc_launch(boundary_kernel<BND_TPB, true, BND_HINT, 0>, dim3(T), dim3(BND_TPB), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn, (const __nv_bfloat16*)nullptr, (float*)nullptr, (const float*)nullptr, (const float*)nullptr);
  else
    okc_launch(boundary_kernel<BND_TPB, false, BND_HINT, 0>, dim3(T), dim3(BND_TPB), 0, st, (const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn, (const __nv_bfloat16*)nullptr, (float*)nullptr, (const float*)nullptr, (const float*)nullptr);
}
void k_boundary_rc(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                   const void* gpre, float* sout, void* nout, int T, int mode, cudaStream_t st, int* zp, int zn,
                   const void* br1, float* rio, const float* alpha1, const float* beta1) {
  if (mode == 1)
    boundary_kernel<128, false, false, 1><<<T, 128, 0, st>>>((const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn,
                                                                        (const __nv_bfloat16*)br1, rio, alpha1, beta1);
  else
    boundary_kernel<BND_TPB, true, BND_HINT, 2><<<T, BND_TPB, 0, st>>>((const __nv_bfloat16*)branch, stream, (const __nv_bfloat16*)gpost, alpha, beta, (const __nv_bfloat16*)gpre, sout, (__nv_bfloat16*)nout, zp, zn,
                                                                       (const __nv_bfloat16*)br1, rio, alpha1, beta1);
}
void k_kv_gather(const void* kc, const void* vc, const int* page_table, const int* seq_lens, void* ko, void* vo, int B, int max_pages,
                 int page, int L, int window, int Smax, cudaStream_t st) {
  dim3 grid((unsigned)((Smax + 31) / 32), (unsigned)B);
  kv_gather_kernel<<<grid, 256, 0, st>>>((const uint8_t*)kc, (const uint8_t*)vc, page_table, seq_lens, (uint8_t*)ko, (uint8_t*)vo,
                                         max_pages, page, L, window, Smax);
}
void k_moe_fused(const void* w13, const void* w2, const void* xt, void* inter, void* y, const int* tiles, const int* ntiles,
                 const float* gs13, const float* gs2, int nctas, int smem, int* counter, int* done, void* dbg, int flags,
                 cudaStream_t st, int* cit, int* qctl, int* tok_left, int* cq, const int* row_tok, const int* rrow,
                 const int* lid, const float* topw, const void* pw, void* out, int T) {
  FusedArgs fa;
  fa.cit = cit; fa.qctl = qctl; fa.tok_left = tok_left; fa.cq = cq; fa.row_tok = row_tok; fa.rrow = rrow; fa.lid = lid;
  fa.topw = topw; fa.pw = (const __nv_bfloat16*)pw; fa.out = (__nv_bfloat16*)out; fa.T = T;
  fa.dbg = (unsigned long long*)dbg;
  fa.flags = flags;
  fa.g1 = make_args(w13, xt, tiles, ntiles, gs13, inter, 64, smem, counter);
  fa.g2 = make_args(w2, inter, tiles, ntiles, gs2, y, 32, smem, counter);
  fa.g1.flags = flags;
  fa.g2.flags = flags;
  fa.done = done;
  if ((fa.flags >> 8) & 1)  // bit 8: fine remainder tiles (the plan used the same flag)
    okc_launch(moe_fused_persistent<1>, dim3((unsigned)nctas), dim3(MOE_FUSED_THREADS), (size_t)smem, st, fa);
  else
    okc_launch(moe_fused_persistent<0>, dim3((unsigned)nctas), dim3(MOE_FUSED_THREADS), (size_t)smem, st, fa);
}
void k_moe_init(int smem) {
  cudaFuncSetAttribute(moe_fused_persistent<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_fused_persistent<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_persistent<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_persistent<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(moe_gemm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}
// ---- attention ----
#define ATTN_HALF_MAX 640  // per-head prep grids below this use half-tile blocks
void k_attn_prep(const void* kc, const void* vc, const int* page_table, int max_pages, int page, const int* seq_lens, int B, int L,
                 int window, int ntk, void* kt, void* vt, cudaStream_t st, int part) {
  if (ntk * B >= 264) {  // long contexts: both heads per block (full cache rows, one page-table read per key)
    dim3 grid((unsigned)ntk, 1, (unsigned)B);
    okc_launch(okc_attn::attn_prep2_kernel, grid, dim3(256), 0, st, (const uint8_t*)kc, (const uint8_t*)vc, page_table, max_pages, page, seq_lens, L,
               window, ntk, (uint8_t*)kt, (uint8_t*)vt, part);
  } else if (2 * ntk * B >= ATTN_HALF_MAX) {  // one head per block
    dim3 grid((unsigned)ntk, 2, (unsigned)B);
    okc_launch(okc_attn::attn_prep_kernel<1>, grid, dim3(256), 0, st, (const uint8_t*)kc, (const uint8_t*)vc, page_table, max_pages, page, seq_lens, L,
               window, ntk, (uint8_t*)kt, (uint8_t*)vt, part);
  } else {  // few tiles (short sliding windows): one head and half a tile per block
    dim3 grid((unsigned)(2 * ntk), 2, (unsigned)B);
    okc_launch(okc_attn::attn_prep_kernel<2>, grid, dim3(256), 0, st, (const uint8_t*)kc, (const uint8_t*)vc, page_table, max_pages, page, seq_lens, L,
               window, ntk, (uint8_t*)kt, (uint8_t*)vt, part);
  }
}
#define ATTN_CR 2  // (token, head) rows per warp in the split merge (Oct 26 search: 2 +0.03..0.06% over 4)
void k_attn_fwd(const void* q, const void* kt, const void* vt, const int* seq_lens, const void* qkv, void* out, int B, int L,
                int window, int ntk, float c, float v_scale, int smem, int nsplit, float* ws, int variant, cudaStream_t st,
                const void* pf, unsigned long long pf_bytes) {
  okc_attn::AttnArgs a;
  a.q = (const uint8_t*)q; a.kt = (const uint8_t*)kt; a.vt = (const uint8_t*)vt; a.seq_lens = seq_lens;
  a.qkv = (const __nv_bfloat16*)qkv; a.out = (__nv_bfloat16*)out;
  const int nwg = variant == 99 ? 1 : (variant == 64 ? 4 : ((variant & 8) ? 3 : 2));
  a.L = L; a.window = window; a.ntk = ntk; a.nqb = (L + 8 * nwg - 1) / (8 * nwg); a.c = c; a.v_scale = v_scale;
  a.zero = 0;
  a.pf = (const uint8_t*)pf; a.pf_bytes = pf_bytes;
  a.sk = 0; a.nb = B; a.skbuf = nullptr;
  if (nsplit < 0) {  // stream-K: grid -nsplit; ws = [int header: W, pair prefix, first pairs | partials]
    const int G = -nsplit, npairs = B * a.nqb * 2;
    const int hdr = ((2 + npairs + 1 + G) + 63) & ~63;
    a.sk = G; a.skbuf = reinterpret_cast<int*>(ws); a.ws = ws + hdr; a.nsplit = 1;
    if (variant == 43) okc_launch(okc_attn::attn_fwd_sk_kernel<3, 0, true, true>, dim3((unsigned)G), dim3(512), (size_t)smem, st, a);
    else okc_launch(okc_attn::attn_fwd_sk_kernel<3, 0, false, false>, dim3((unsigned)G), dim3(512), (size_t)smem, st, a);
    const int T = B * L;
    okc_launch(okc_attn::attn_combine_sk_kernel<ATTN_CR, 24>, dim3((T * (16 / ATTN_CR) + 7) / 8), dim3(256), 0, st, (const float*)(ws + hdr),
               (const int*)ws, T, L, a.nqb, (const __nv_bfloat16*)qkv, (__nv_bfloat16*)out, c, v_scale);
    return;
  }
  a.nsplit = nsplit; a.ws = nsplit > 1 ? ws : nullptr;
  dim3 grid((unsigned)a.nqb, (unsigned)(2 * nsplit), (unsigned)B);
  switch (variant) {
    case 0: okc_launch(okc_attn::attn_fwd_kernel<2, 0, false>, grid, dim3(384), (size_t)smem, st, a); break;
    case 1: okc_launch(okc_attn::attn_fwd_kernel<2, 16, false>, grid, dim3(384), (size_t)smem, st, a); break;
    case 8: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false>, grid, dim3(512), (size_t)smem, st, a); break;
    case 9: okc_launch(okc_attn::attn_fwd_kernel<3, 16, false>, grid, dim3(512), (size_t)smem, st, a); break;
    case 16: okc_launch(okc_attn::attn_fwd_kernel<2, 0, true>, grid, dim3(384), (size_t)smem, st, a); break;
    case 43: okc_launch(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, grid, dim3(512), (size_t)smem, st, a); break;
    default: okc_launch(okc_attn::attn_fwd_kernel<3, 0, true>, grid, dim3(512), (size_t)smem, st, a); break;
  }
  if (nsplit > 1 && !g_attn_merge_host) {  // (with the in-kernel merge the last split's CTA writes the output)
    const int T = B * L;
    okc_launch(okc_attn::attn_combine2_kernel<ATTN_CR>, dim3((T * (16 / ATTN_CR) + 7) / 8), dim3(256), 0, st, ws, nsplit, T, (const __nv_bfloat16*)qkv, (__nv_bfloat16*)out,
               c, v_scale);
  }
}
void k_attn_set_prof(void* p) {
}
void k_attn_init(int smem) {
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<2, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_kernel<3, 0, false, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_sk_kernel<3, 0, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(okc_attn::attn_fwd_sk_kernel<3, 0, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}
