
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
void k_moe_plan(const int* counts, int* offsets, int* tiles, int* ntiles, int* rows, int order, cudaStream_t st);
void k_moe_route(const float* logits, const float* bias, const int* lstart, int* ids, float* w, int* lid, int* pos, int* cnt,
                 int T, cudaStream_t st, int* tok_left, int* cq);
void k_moe_gather(const void* p, const void* gw, const int* lid, const int* pos, const int* offsets, const int* counts,
                  void* xt, int* rrow, int T, cudaStream_t st);
void k_moe_gather_plan(const void* p, const void* gw, const int* lid, const int* pos, const int* counts, void* xt, int* rrow,
                       int* offsets, int* tiles, int* ntiles, int* rows, int order, int T, cudaStream_t st,
                       int* row_tok, int* cq, int* qtail);
void k_moe_combine(const void* y, const int* rrow, const int* lid, const float* topw, const void* pw, void* out, int T,
                   cudaStream_t st);
void k_moe_gemm(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                int P, bool gated, int smem, cudaStream_t st);
void k_moe_gemm_p(const void* wrep, const void* xt, const int* tiles, const int* ntiles, const float* gscale, void* out, int NT,
                  int nctas, bool gated, int smem, int* counter, cudaStream_t st);
void k_boundary(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                const void* gpre, float* sout, void* nout, int T, bool has_gpost, cudaStream_t st, int* zp, int zn);
void k_qk_prep(const void* qkv, const void* pos, bool pos64, const void* loc, bool loc64, const float* invf, void* q8, void* kc,
               void* vc, const float* ksc, const float* vsc, int T, bool rope, bool scaled, cudaStream_t st);
void k_boundary_rc(const void* branch, const float* stream, const void* gpost, const float* alpha, const float* beta,
                   const void* gpre, float* sout, void* nout, int T, int mode, cudaStream_t st, int* zp, int zn,
                   const void* br1, float* rio, const float* alpha1, const float* beta1);
void k_kv_gather(const void* kc, const void* vc, const int* page_table, const int* seq_lens, void* ko, void* vo, int B, int max_pages,
                 int page, int L, int window, int Smax, cudaStream_t st);
void k_moe_init(int smem);
void k_moe_fused(const void* w13, const void* w2, const void* xt, void* inter, void* y, const int* tiles, const int* ntiles,
                 const float* gs13, const float* gs2, int nctas, int smem, int* counter, int* done, void* dbg, int flags,
                 cudaStream_t st, int* cit, int* qctl, int* tok_left, int* cq, const int* row_tok, const int* rrow,
                 const int* lid, const float* topw, const void* pw, void* out, int T);
void k_attn_prep(const void* kc, const void* vc, const int* page_table, int max_pages, int page, const int* seq_lens, int B, int L,
                 int window, int ntk, void* kt, void* vt, cudaStream_t st, int part);
void k_attn_fwd(const void* q, const void* kt, const void* vt, const int* seq_lens, const void* qkv, void* out, int B, int L,
                int window, int ntk, float c, float v_scale, int smem, int nsplit, float* ws, int variant, cudaStream_t st,
                const void* pf, unsigned long long pf_bytes);
void k_attn_init(int smem);
void k_attn_set_prof(void* p);

static cudaStream_t cur() { return c10::cuda::getCurrentCUDAStream(); }

void moe_route(torch::Tensor logits, torch::Tensor bias, torch::Tensor lstart, torch::Tensor ids, torch::Tensor w,
               torch::Tensor lid, torch::Tensor pos, torch::Tensor cnt, torch::Tensor tok_left, torch::Tensor cq) {
  k_moe_route(logits.data_ptr<float>(), bias.data_ptr<float>(), lstart.data_ptr<int>(), ids.data_ptr<int>(), w.data_ptr<float>(),
              lid.data_ptr<int>(), pos.data_ptr<int>(), cnt.data_ptr<int>(), (int)logits.size(0), cur(),
              tok_left.numel() ? tok_left.data_ptr<int>() : nullptr, cq.numel() ? cq.data_ptr<int>() : nullptr);
}
void moe_plan(torch::Tensor counts, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor rows,
              int64_t order) {
  k_moe_plan(counts.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
             (int)order, cur());
}
void moe_gather_plan(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor counts,
                     torch::Tensor xt, torch::Tensor rrow, torch::Tensor offsets, torch::Tensor tiles, torch::Tensor ntiles,
                     torch::Tensor rows, int64_t order, torch::Tensor cit, torch::Tensor tok) {
  // CIT (cit.numel() > 0): queue tail = cit[numel - 3]; tok = [tok_left (T) | cq (T) | row_tok]
  const bool on = cit.numel() > 0;
  const int T = (int)p.size(0);
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_gather_plan(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), counts.data_ptr<int>(), xt.data_ptr(),
                    rrow.data_ptr<int>(), offsets.data_ptr<int>(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), rows.data_ptr<int>(),
                    (int)order, T, cur(), on ? tk + 2 * T : nullptr, on ? tk + T : nullptr,
                    on ? cit.data_ptr<int>() + (cit.numel() - 3) : nullptr);
}
void moe_gather(torch::Tensor p, torch::Tensor gw, torch::Tensor lid, torch::Tensor pos, torch::Tensor offsets,
                torch::Tensor counts, torch::Tensor xt, torch::Tensor rrow) {
  k_moe_gather(p.data_ptr(), gw.data_ptr(), lid.data_ptr<int>(), pos.data_ptr<int>(), offsets.data_ptr<int>(),
               counts.data_ptr<int>(), xt.data_ptr(), rrow.data_ptr<int>(), (int)p.size(0), cur());
}
void moe_combine(torch::Tensor y, torch::Tensor rrow, torch::Tensor lid, torch::Tensor topw, torch::Tensor pw, torch::Tensor out) {
  k_moe_combine(y.data_ptr(), rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), pw.data_ptr(),
                out.data_ptr(), (int)out.size(0), cur());
}
void moe_gemm(torch::Tensor wrep, torch::Tensor xt, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor gscale,
              torch::Tensor out, int64_t NT, int64_t P, bool gated, int64_t smem) {
  k_moe_gemm(wrep.data_ptr(), xt.data_ptr(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), gscale.data_ptr<float>(), out.data_ptr(),
             (int)NT, (int)P, gated, (int)smem, cur());
}
void moe_gemm_p(torch::Tensor wrep, torch::Tensor xt, torch::Tensor tiles, torch::Tensor ntiles, torch::Tensor gscale,
                torch::Tensor out, int64_t NT, int64_t nctas, bool gated, int64_t smem, torch::Tensor counter) {
  k_moe_gemm_p(wrep.data_ptr(), xt.data_ptr(), tiles.data_ptr<int>(), ntiles.data_ptr<int>(), gscale.data_ptr<float>(),
               out.data_ptr(), (int)NT, (int)nctas, gated, (int)smem, counter.data_ptr<int>(), cur());
}
void boundary_rc(torch::Tensor branch, torch::Tensor stream, torch::Tensor gpost, torch::Tensor alpha, torch::Tensor beta,
                 torch::Tensor gpre, torch::Tensor sout, torch::Tensor nout, int64_t mode, torch::Tensor zero,
                 torch::Tensor br1, torch::Tensor rio, torch::Tensor alpha1, torch::Tensor beta1) {
  k_boundary_rc(branch.data_ptr(), stream.data_ptr<float>(), gpost.data_ptr(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
                gpre.data_ptr(), sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), (int)mode, cur(),
                zero.numel() ? zero.data_ptr<int>() : nullptr, (int)zero.numel(), br1.data_ptr(), rio.data_ptr<float>(),
                alpha1.data_ptr<float>(), beta1.data_ptr<float>());
}
void qk_prep(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor invf, torch::Tensor q8, torch::Tensor kc,
             torch::Tensor vc, torch::Tensor ksc, torch::Tensor vsc, bool rope, bool scaled) {
  k_qk_prep(qkv.data_ptr(), pos.data_ptr(), pos.element_size() == 8, loc.data_ptr(), loc.element_size() == 8, invf.data_ptr<float>(),
            q8.data_ptr(), kc.data_ptr(), vc.data_ptr(), ksc.data_ptr<float>(), vsc.data_ptr<float>(), (int)qkv.size(0), rope, scaled, cur());
}
void boundary(torch::Tensor branch, torch::Tensor stream, torch::Tensor gpost, torch::Tensor alpha, torch::Tensor beta,
              torch::Tensor gpre, torch::Tensor sout, torch::Tensor nout, bool has_gpost, torch::Tensor zero) {
  k_boundary(branch.data_ptr(), stream.data_ptr<float>(), gpost.data_ptr(), alpha.data_ptr<float>(), beta.data_ptr<float>(),
             gpre.data_ptr(), sout.data_ptr<float>(), nout.data_ptr(), (int)branch.size(0), has_gpost, cur(),
             zero.numel() ? zero.data_ptr<int>() : nullptr, (int)zero.numel());
}
void kv_gather(torch::Tensor kc, torch::Tensor vc, torch::Tensor page_table, torch::Tensor seq_lens, torch::Tensor ko,
               torch::Tensor vo, int64_t page, int64_t L, int64_t window, int64_t Smax) {
  k_kv_gather(kc.data_ptr(), vc.data_ptr(), page_table.data_ptr<int>(), seq_lens.data_ptr<int>(), ko.data_ptr(), vo.data_ptr(),
              (int)seq_lens.size(0), (int)page_table.size(1), (int)page, (int)L, (int)window, (int)Smax, cur());
}
void moe_init(int64_t smem) { k_moe_init((int)smem); }
void moe_fused(torch::Tensor w13, torch::Tensor w2, torch::Tensor xt, torch::Tensor inter, torch::Tensor y, torch::Tensor tiles,
               torch::Tensor ntiles, torch::Tensor gs13, torch::Tensor gs2, int64_t nctas, int64_t smem, torch::Tensor counter,
               torch::Tensor done, torch::Tensor dbg, int64_t flags, torch::Tensor cit, torch::Tensor tok, torch::Tensor rrow,
               torch::Tensor lid, torch::Tensor topw, torch::Tensor pw, torch::Tensor out) {
  // CIT (cit.numel() > 0): cit = [max tiles GEMM2 counters | head, tail, error, pad]; tok = [tok_left (T) | cq (T) | row_tok]
  const bool on = cit.numel() > 0;
  const int T = (int)rrow.size(0);
  int* c = on ? cit.data_ptr<int>() : nullptr;
  int* tk = on ? tok.data_ptr<int>() : nullptr;
  k_moe_fused(w13.data_ptr(), w2.data_ptr(), xt.data_ptr(), inter.data_ptr(), y.data_ptr(), tiles.data_ptr<int>(),
              ntiles.data_ptr<int>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), (int)nctas, (int)smem, counter.data_ptr<int>(),
              done.data_ptr<int>(), dbg.numel() ? dbg.data_ptr() : nullptr, (int)flags, cur(),
              c, on ? c + (cit.numel() - 4) : nullptr, tk, on ? tk + T : nullptr, on ? tk + 2 * T : nullptr,
              rrow.data_ptr<int>(), lid.data_ptr<int>(), topw.data_ptr<float>(), pw.data_ptr(), on ? out.data_ptr() : nullptr, T);
}
void attn_prep(torch::Tensor kc, torch::Tensor vc, torch::Tensor page_table, torch::Tensor seq_lens, int64_t page, int64_t L,
               int64_t window, int64_t ntk, torch::Tensor kt, torch::Tensor vt, int64_t part) {
  k_attn_prep(kc.data_ptr(), vc.data_ptr(), page_table.data_ptr<int>(), (int)page_table.size(1), (int)page, seq_lens.data_ptr<int>(),
              (int)seq_lens.size(0), (int)L, (int)window, (int)ntk, kt.data_ptr(), vt.data_ptr(), cur(), (int)part);
}
void attn_fwd(torch::Tensor q, torch::Tensor kt, torch::Tensor vt, torch::Tensor seq_lens, torch::Tensor qkv, torch::Tensor out,
              int64_t L, int64_t window, int64_t ntk, double c, double v_scale, int64_t smem, int64_t nsplit, torch::Tensor ws,
              int64_t variant, torch::Tensor pf) {
  k_attn_fwd(q.data_ptr(), kt.data_ptr(), vt.data_ptr(), seq_lens.data_ptr<int>(), qkv.data_ptr(), out.data_ptr(),
             (int)seq_lens.size(0), (int)L, (int)window, (int)ntk, (float)c, (float)v_scale, (int)smem, (int)nsplit,
             nsplit != 1 ? ws.data_ptr<float>() : nullptr, (int)variant, cur(), pf.numel() ? pf.data_ptr() : nullptr,
             (unsigned long long)(pf.numel() * pf.element_size()));
}
void attn_init(int64_t smem) { k_attn_init((int)smem); }
void attn_set_prof(torch::Tensor p) { k_attn_set_prof(p.numel() ? p.data_ptr() : nullptr); }
