def _gle_source():
    cfg = next(c for c in bfp_SK2_CONFIGS if c[0] == bfp_SK2_PICK[("o", 8)])
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    assert (layer, R, NWR * WSK, CL) == ("o", 32, 4, 1), cfg
    _, nstage = bfp__sk2_geom(cfg, 8)
    s = bfp_SK2_SRC
    s = s.replace("#define SK2_DIAG 0", "#define SK2_DIAG 0\n"
                  "struct GleEpi { const float *s, *a, *b, *a2; const __nv_bfloat16* p; __nv_bfloat16* uv; float* part; float* uvf; const float *x32, *w32; int fpr, ilv; };\n"
                  "static GleEpi g_gle;", 1)
    s = s.replace("           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {",
                  "           float* __restrict__ part, int* __restrict__ rcnt, int pdl, const GleEpi ep) {", 1)
    # the epilogue's input-side vectors (residual stream row, alpha / beta / alpha2 / pre) load before the wait
    pre = r"""  __shared__ __align__(16) float4 gx_s[64], gx_a[64], gx_b[64], gx_a2[64];
  __shared__ __align__(16) uint2 gx_p[64];
  // the epilogue's input-side vectors (residual stream row, alpha / beta / alpha2 / pre) go to shared memory by
  // cp.async before the wait and are awaited only in the epilogue: no registers held across the main loop (2 CTAs / SM
  // next to attention) and no stall behind the pre-wait weight stages
  // ep.s is data['stream'], the layer's input residual stream: nothing in the call writes it, which is what makes this
  // pre-wait copy safe (keep it that way: any in-call writer of data['stream'] must move this copy after the wait)
  if (!FUSE && tid < 8 * M) {   // o only (this source also holds the fused pooler)
    const size_t col = (size_t)cta * R + (tid & 7) * 4;
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&gx_s[tid]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d0), "l"(ep.s + (size_t)(tid >> 3) * N + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a[tid])), "l"(ep.a + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_b[tid])), "l"(ep.b + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a2[tid])), "l"(ep.a2 + col) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_p[tid])), "l"(ep.p + col) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
"""
    anchor = '  __syncthreads();\n  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread'
    assert s.count(anchor) == 1
    s = s.replace(anchor, pre + anchor, 1)
    epi = __cuda__("gle_epi.cpp")
    anchor = '  constexpr int R4 = R / 4;\n  for (int idx = tid; idx < R4 * M; idx += NC) {'
    assert s.count(anchor) == 1
    s = s.replace(anchor, epi, 1)
    # FUSE with ep.fpr: the router piece is an FP32 GEMV on CUDA cores (FP32 x, FP32 weight, FP32 FMAs and reductions)
    # through the same ring as the BF16 hi + lo router: 4 stages per CTA, each = 8 router rows x 512 columns of FP32
    # weights (16 KB, 8 bulk copies) + M FP32 x rows x 512 columns (bulk copies), consumed from shared memory
    fpr_stage = (
        "        } else if (ep.fpr) {   // FP32 router stage j = st - nst: columns rq * 2048 + 512 j .. + 511\n"
        "          const int kb = rq * 2048 + (st - nst) * 512;\n"
        "          mbar_expect_tx(full0 + 8 * slot, 8 * 2048 + M * 2048);\n"
        "          for (int r = 0; r < 8; ++r)\n"
        "            bulk_g2s(s0 + r * 2048, reinterpret_cast<const uint8_t*>(ep.w32 + (size_t)(rt * 8 + r) * 8192 + kb), 2048, full0 + 8 * slot);\n"
        "          for (int t = 0; t < M; ++t)\n"
        "            bulk_g2s(s0 + 16384 + t * 2048, reinterpret_cast<const uint8_t*>(ep.x32 + (size_t)t * 8192 + kb), 2048, full0 + 8 * slot);\n"
        "        } else {   // router piece: global router stage rq * 4 + j of tile rt")
    fpr_cons = (
        "  if constexpr (FUSE) {\n    if (ep.fpr) {\n"
        "      static_assert(!FUSE || (NW == 8 && SST >= 32768), \"fp32 router stages: 8 consumer warps, 32 KB slots\");\n"
        "      // thread: column group cg = tid & 127 (4 of the stage's 512 columns) x all 8 router rows x tokens 4h .. 4h + 3\n"
        "      // (h = tid >> 7): every shared-memory float4 is read by 2 threads, not 8 warps; FP32 FMAs\n"
        "      const int cg = tid & 127, th = tid >> 7;\n"
        "      float af[32];\n"
        "#pragma unroll\n"
        "      for (int i = 0; i < 32; ++i) af[i] = 0.f;\n"
        "      for (int st = nst; st < nstt; ++st) {\n"
        "        const int slot = st % NSTAGE;\n"
        "        mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n"
        "        const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + cg * 4;\n"
        "        const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384) + cg * 4;\n"
        "        float4 xa[4];\n"
        "#pragma unroll\n"
        "        for (int u = 0; u < 4; ++u)\n"
        "          xa[u] = (th * 4 + u < M) ? *reinterpret_cast<const float4*>(xs + (th * 4 + u) * 512) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
        "#pragma unroll\n"
        "        for (int r = 0; r < 8; ++r) {\n"
        "          const float4 bw = *reinterpret_cast<const float4*>(ws + r * 512);\n"
        "#pragma unroll\n"
        "          for (int u = 0; u < 4; ++u) {\n"
        "            float v = af[r * 4 + u];\n"
        "            v = fmaf(xa[u].x, bw.x, v); v = fmaf(xa[u].y, bw.y, v); v = fmaf(xa[u].z, bw.z, v); v = fmaf(xa[u].w, bw.w, v);\n"
        "            af[r * 4 + u] = v;\n"
        "          }\n"
        "        }\n"
        "        __syncwarp();\n"
        "        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n"
        "      }\n"
        "      // reduction over the 128 column groups through the (now idle) ring: red[th][cg][33] (static indices only)\n"
        "      if constexpr (COMP) mbar_wait(cbar, 0);   // as the BF16 router: the escape warps are done with the ring\n"
        "      named_bar(1, NC);                         // and every consumer is done reading its last stage\n"
        "      float* red32 = reinterpret_cast<float*>(gbase);\n"
        "#pragma unroll\n"
        "      for (int i = 0; i < 32; ++i) red32[(th * 128 + cg) * 33 + i] = af[i];\n"
        "      named_bar(1, NC);\n"
        "      float* red2 = red32 + 256 * 33;\n"
        "      {\n"
        "        const int o = tid & 63, pq = tid >> 6, h = o >> 5, e = o & 31;\n"
        "        float v = 0.f;\n"
        "        for (int c = pq * 32; c < pq * 32 + 32; ++c) v += red32[(h * 128 + c) * 33 + e];\n"
        "        red2[pq * 64 + o] = v;\n"
        "      }\n"
        "      named_bar(1, NC);\n"
        "      if (tid < 64) {\n"
        "        const int h = tid >> 5, e = tid & 31, r = e >> 2, t = h * 4 + (e & 3);\n"
        "        const float v = red2[tid] + red2[64 + tid] + red2[128 + tid] + red2[192 + tid];\n"
        "        if (t < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + t * 8 + r, v);\n"
        "      }\n"
        "      named_bar(1, NC);\n"
        "    } else {\n    // router piece:")
    # ep.ilv: the 4 FP32 router stages sit at ring positions PER - 1, 2 PER - 1, ... (PER = nstt / 4) between the pooler
    # stages, so their loads and FMAs overlap the pooler stream; ring position q holds pooler stage q - q / PER otherwise
    ilv_edits = (
        ("  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\n",
         "  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\n"
         "  const bool ilv = FUSE && ep.fpr && ep.ilv && (nstt % 4) == 0 && nstt / 4 > P;\n"
         "  const int PER = ilv ? nstt / 4 : (1 << 30);\n"
         "  const int OFF = ilv ? (ep.ilv == 2 ? P : PER - 1) : 0;   // router stage slot inside each period\n"),
        ("        if (!FUSE || st < nst) {\n"
         "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n"
         "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);\n"
         "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\n",
         "        const bool isr = ilv ? (st % PER == OFF) : (FUSE && st >= nst);\n"
         "        const int ps = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\n"
         "        if (!isr) {\n"
         "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n"
         "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + ps) * wstride, WBY, full0 + 8 * slot);\n"
         "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + ps) * UPS);\n"),
        ("          const int kb = rq * 2048 + (st - nst) * 512;\n",
         "          const int kb = rq * 2048 + (ilv ? st / PER : st - nst) * 512;\n"),
        ("        if (FUSE && st >= nst) {   // router stages:",
         "        if (ilv ? (st % PER == OFF) : (FUSE && st >= nst)) {   // router stages:"),
        ("        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;\n",
         "        const int psq = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\n"
         "        const int kbeg = (gs0 + psq) * KST, kend = kbeg + KST;\n"),
        ("        if ((st % NESC) != ew) {", "        if ((psq % NESC) != ew) {"),
        ("  for (int st = 0; st < nst; ++st) {\n    const int slot = st % NSTAGE;\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n",
         "  float afi[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};   // ilv: FP32 router accumulators (warp = router row)\n"
         "  for (int st = 0; st < (ilv ? nstt : nst); ++st) {\n    const int slot = st % NSTAGE;\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n"
         "    if constexpr (FUSE) if (ilv && st % PER == OFF) {   // an interleaved FP32 router stage\n"
         "      const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + warp * 512;\n"
         "      const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384);\n"
         "#pragma unroll\n"
         "      for (int i = 0; i < 4; ++i) {\n"
         "        const int k = lane * 4 + i * 128;\n"
         "        const float4 bw = *reinterpret_cast<const float4*>(ws + k);\n"
         "#pragma unroll\n"
         "        for (int t = 0; t < 8; ++t) if (t < M) {\n"
         "          const float4 a = *reinterpret_cast<const float4*>(xs + t * 512 + k);\n"
         "          afi[t] = fmaf(a.x, bw.x, afi[t]); afi[t] = fmaf(a.y, bw.y, afi[t]);\n"
         "          afi[t] = fmaf(a.z, bw.z, afi[t]); afi[t] = fmaf(a.w, bw.w, afi[t]);\n"
         "        }\n"
         "      }\n"
         "      __syncwarp();\n"
         "      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n"
         "      continue;\n"
         "    }\n"),
        ("  if constexpr (FUSE) {\n    if (ep.fpr) {\n",
         "  if constexpr (FUSE) {\n    if (ilv) {   // interleaved FP32 router: warp butterflies of afi, then the 4-quarter reduction below\n"
         "#pragma unroll\n"
         "      for (int t = 0; t < 8; ++t)\n"
         "#pragma unroll\n"
         "        for (int d = 16; d; d >>= 1) afi[t] += __shfl_xor_sync(0xffffffffu, afi[t], d);\n"
         "      float mine = afi[0];\n"
         "#pragma unroll\n"
         "      for (int t = 1; t < 8; ++t) mine = lane == t ? afi[t] : mine;\n"
         "      if (lane < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + lane * 8 + warp, mine);\n"
         "      if constexpr (COMP) mbar_wait(cbar, 0);\n"
         "      named_bar(1, NC);\n"
         "    } else if (ep.fpr) {\n"))
    for old, new in (
            ("        } else {   // router piece: global router stage rq * 4 + j of tile rt", fpr_stage),
            ("  if constexpr (FUSE) {\n    // router piece:", fpr_cons),
            ("    named_bar(1, NC);\n    __shared__ int s_rlast;",
             "    named_bar(1, NC);\n    }\n    __shared__ int s_rlast;")):
        assert s.count(old) == 1, old[:60]
        s = s.replace(old, new, 1)
    for old, new in ilv_edits:
        assert s.count(old) == 1, old[:70]
        s = s.replace(old, new, 1)
    s = s.replace("wsr, outr, part, rcnt, pdl));", "wsr, outr, part, rcnt, pdl, g_gle));", 1)
    s = s.replace("wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, 0, g_gle);", 1)
    host = (f"void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,\n"
            f"         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {{")
    assert s.count(host) == 1
    disp = (f"if (tokr == 8 && N == 8192 && K == 1024 && M <= 8) {{ launch_sk2<1, 8, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
            f"{NACC}, {CL}, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, 0, nullptr); return; }}")
    pcfg = next(c for c in bfp_SK2_CONFIGS if c[0] == "pooler")
    flines = []
    for tokr in (8,):   # the fused pooler + FP32 router at M <= 8
        _, pns = bfp__sk2_geom(pcfg, tokr)
        _, _, _, pR, pNT, pNWR, pWSK, pUPW, pNACC, pCL = pcfg
        flines.append(f"if (tokr == {tokr}) {{ launch_sk2<1, {tokr}, {pNT}, {pNWR}, {pWSK}, {pUPW}, {pns}, true, false, "
                      f"{pNACC}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                      f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                      f"rcnt.data_ptr<int>()); return; }}")
    s = s.replace("@@SK2_DISPATCH@@", disp).replace("@@SK2F_DISPATCH@@", "\n  ".join(flines))
    s += r"""
void gle_o(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,
           at::Tensor s, at::Tensor a, at::Tensor b, at::Tensor a2, at::Tensor p, at::Tensor uv, at::Tensor part, at::Tensor uvf) {
  TORCH_CHECK(M >= 1 && M <= 8 && s.size(0) == M && uv.size(0) == 2 * M && uvf.size(0) == 2 * M && part.numel() >= 256 * M * 8, "gle_o: shapes");
  g_gle = GleEpi{s.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), a2.data_ptr<float>(),
                 reinterpret_cast<const __nv_bfloat16*>(p.data_ptr()), reinterpret_cast<__nv_bfloat16*>(uv.data_ptr()),
                 part.data_ptr<float>(), uvf.data_ptr<float>()};
  sk2(Ws, x, out, ptab, pmax, base4, M, 8192, 1024, 0, 8, 0, ptab);
}
void gle_pr(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
            int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, at::Tensor x32, at::Tensor w32, int64_t wpf) {
  // fused pooler (BF16 x rows) + FP32 router (FP32 x32 rows, FP32 weights w32) at M <= 8
  TORCH_CHECK(M >= 1 && M <= 8 && x.size(0) == M && x32.size(0) == M && x32.scalar_type() == at::kFloat &&
              w32.scalar_type() == at::kFloat && w32.size(0) == 256 && w32.size(1) == 8192, "gle_pr: shapes");
  g_gle = GleEpi{};
  g_gle.x32 = x32.data_ptr<float>(); g_gle.w32 = w32.data_ptr<float>(); g_gle.fpr = 1; g_gle.ilv = (int)wpf;
  sk2f(Ws, Wsr, x, out, outr, ptab, pmax, base4, part, rcnt, M, 8);
  g_gle = GleEpi{};
}
"""
    shared = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", "")
    return shared + bfp_MAP_SRC + s


def _gle_ext():
    global _GLE_EXT
    if _GLE_EXT is None:
        source = _gle_source()
        tag = _hashlib.md5(source.encode()).hexdigest()[:12]
        source = (source.replace("sk2_kernel", "sk2_kernel_gle" + tag).replace("launch_sk2", "launch_sk2_gle" + tag)
                  .replace("void sk2(", "void sk2_gle" + tag + "(").replace("  sk2(Ws,", "  sk2_gle" + tag + "(Ws,")
                  .replace("void sk2f(", "void sk2f_gle" + tag + "(").replace("  sk2f(Ws,", "  sk2f_gle" + tag + "(Ws,"))
        cpp = ("void gle_o(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,"
               " int64_t M, torch::Tensor s, torch::Tensor a, torch::Tensor b, torch::Tensor a2, torch::Tensor p,"
               " torch::Tensor uv, torch::Tensor part, torch::Tensor uvf);\n"
               "void gle_pr(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr,"
               " torch::Tensor ptab, int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M,"
               " torch::Tensor x32, torch::Tensor w32, int64_t wpf);\nvoid sk2_set_pdl(int64_t n);\n")
        _GLE_EXT = load_inline("gle_o_" + tag, "#include <torch/extension.h>\n" + cpp, cuda_sources=source,
                               functions=["gle_o", "gle_pr", "sk2_set_pdl"], no_implicit_headers=True,
                               extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"], verbose=False)
    return _GLE_EXT


@triton.jit
def _gle_carry(PO, S, ALPHA, BETA, ALPHA2, t, cols, r1, ROWS1: tl.constexpr):
    # boundary 1's residual: ns = alpha * s + beta * bf16(r1 * P) (b1's FMA order), carry = alpha2 * ns
    p = tl.load(PO + t * 8192 + cols).to(tl.float32)
    s = tl.load(S + t * 8192 + cols)
    a = tl.load(ALPHA + cols)
    b = tl.load(BETA + cols)
    bn = (p * r1).to(tl.bfloat16).to(tl.float32)
    if ROWS1:
        ns = tl.fma(b, bn, a * s)
    else:
        ns = tl.where(cols < 4096, tl.fma(a, s, b * bn), tl.fma(b, bn, a * s))
    return tl.load(ALPHA2 + cols) * ns


@triton.jit
def _gle_route(POOLED, LOGITS, PART, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, RT,
               PO, S, ALPHA, BETA, ALPHA2, S1,
               MODE: tl.constexpr, PARTS: tl.constexpr, NCH: tl.constexpr, TT: tl.constexpr):
    # _sl_tile_route on the two-row (u, v) pooler / router outputs; r1 / r2 come from the o epilogue's partial sums,
    # complete before this launch (o completes before the pooler wait returns; the FP32 router is a plain launch)
    t = tl.program_id(0)
    kind = tl.program_id(1)
    k8 = tl.arange(0, 8)
    cta = tl.arange(0, 256)
    q = tl.sum(tl.load(PART + (cta[:, None] * T + t) * 8 + k8[None, :]), 0)
    q0 = tl.sum(tl.where(k8 == 0, q, 0.0), 0)
    q1 = tl.sum(tl.where(k8 == 1, q, 0.0), 0)
    q2 = tl.sum(tl.where(k8 == 2, q, 0.0), 0)
    q3 = tl.sum(tl.where(k8 == 3, q, 0.0), 0)
    r1 = tl.rsqrt(q0 / 8192.0 + 1e-8)
    r2 = tl.rsqrt((q1 + 2.0 * r1 * q2 + r1 * r1 * q3) / 8192.0 + 1e-5)
    if kind > PARTS:
        # carry programs (no wait: they read o's outputs and the inputs only): column chunk (kind - PARTS - 1) * T + t
        # of 8192 // (NCH * T) columns, for every token
        cc = ((kind - PARTS - 1) * TT + t) * (8192 // (NCH * TT)) + tl.arange(0, 8192 // (NCH * TT))
        for tok in tl.static_range(TT):
            r1t = tl.rsqrt(tl.sum(tl.load(PART + (cta * TT + tok) * 8), 0) / 8192.0 + 1e-8)
            tl.store(S1 + tok * 8192 + cc, _gle_carry(PO, S, ALPHA, BETA, ALPHA2, tok, cc, r1t, TT == 1))
    elif kind < PARTS:
        c = tl.arange(0, 4096)
        oc = kind * (4096 // PARTS) + tl.arange(0, 4096 // PARTS)
        gpre = 1.0 + tl.load(GPRE + oc).to(tl.float32)
        _gdc_wait()
        _gdc_launch()
        pooled = (tl.load(POOLED + t * 4096 + c).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + c).to(tl.float32) * r1) * r2
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5)
        if PARTS == 1:
            vals = pooled
        else:
            vals = (tl.load(POOLED + t * 4096 + oc).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + oc).to(tl.float32) * r1) * r2
        tl.store(XOUT + t * 4096 + oc, (vals * inv * gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        bias = tl.load(BIAS + e)
        ls = tl.load(LS)
        q4 = tl.sum(tl.where(k8 == 4, q, 0.0), 0)
        q5 = tl.sum(tl.where(k8 == 5, q, 0.0), 0)
        q6 = tl.sum(tl.where(k8 == 6, q, 0.0), 0)
        tail = q4 + 2.0 * r1 * q5 + r1 * r1 * q6
        _gdc_wait()
        _gdc_launch()
        tl.store(RT + t, r1)
        tl.store(RT + T + t, tail)
        logit = (tl.load(LOGITS + t * 256 + e) + tl.load(LOGITS + (T + t) * 256 + e) * r1) * r2
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - ls + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


def _gle_frt(xb, x32, state):
    """Fused BF16 pooler + FP32 router for M = xb.shape[0] <= 8 rows: (pooled bf16 [M, 4096], logits fp32 [M, 256])."""
    bst = state["bfp"]
    m = xb.shape[0]
    dev = xb.device
    if "fused_part" not in bst:
        bst["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=dev)
        bst["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
    pci, pstream, ptab_, ppmax, pb4, _ = bst["sk2"]["pooler"]
    pooled = torch.empty((m, 4096), device=dev, dtype=torch.bfloat16)
    logits = torch.empty((m, 256), device=dev, dtype=torch.float32)
    _bfp_pdl(m, "pool")
    ext = _gle_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_pr(pstream, bst["sk2"]["router"][1], xb, pooled, logits, ptab_, ppmax, pb4, bst["fused_part"], bst["fused_cnt"],
               m, x32, state["router_fp32"], int(GLE_ILV))
    return pooled, logits


_gle_prior_pool_router_fp32 = _pool_router_fp32
def _pool_router_fp32(h, hr, T, state, ppick):
    if (GLE_FRT and T <= 8 and hr is not None and ppick in ("bfp", "bfpf") and state.get("bfp") is not None
            and state.get("router_fp32") is not None and hr.is_contiguous() and h.is_contiguous()):
        return _gle_frt(h, hr, state), -1
    return _gle_prior_pool_router_fp32(h, hr, T, state, ppick)


def _gle_o_choice(state, T):
    bst = state.get("bfp")
    if bst is None or not bst.get("sm90"):
        return False
    ch = bst.get("choice", {}).get(("o", T)) or bfp__nearby_choice("o", T, bst) or bfp__static_choice("o", T, bst)
    return ch[0] == "sk2" and bfp__sk2_name("o", T) == bfp_SK2_PICK[("o", 8)]


def _gle_after_attention(data, state, o_in, gate, T):
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    g = state['link']
    bst = state["bfp"]
    ci, stream, tab, pmax, b4, n_rows = bst["sk2"][bfp__sk2_name("o", T)]
    P = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uv = torch.empty((2 * T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uvf = torch.empty((2 * T, HIDDEN), **f32)
    part = torch.empty((256 * T * 8,), **f32)
    rt = torch.empty((2, T), **f32)
    nch = GLE_RC.get(T, GLE_RC.get(str(T), 0))
    s1 = torch.empty((T, HIDDEN), **f32)
    _bfp_pdl(T, "o")
    ext = _gle_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_o(stream, o_in, P, tab, pmax, b4, T, data['stream'], g['alpha1'], g['beta1'], g['alpha2'], state['g_preffn'], uv, part,
              uvf)
    if GLE_FRT:   # fused pooler + FP32 router (one launch)
        pooled, logits = _gle_frt(uv, uvf, state)
    else:         # 0389's T <= 8 split: BF16 pooler (bf16_proj sk2), FP32 router GEMV, here on the two rows per token
        pooled = _bfp_gemm(state, "pool", uv, 2 * T, D_EXPERT, HIDDEN)
        logits = _router_fp32_gemv(uvf, state)
    if GLE_CAPTURE is not None:
        GLE_CAPTURE.update(logits=logits, pooled=pooled, part=part, P=P)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    _link_al(pooled, logits, xm, ids, wts, lids, part, rt)
    _gle_route[(T, SL_TILE_CONFIG[T][1] + 1 + nch)](pooled, logits, part, T, state["bias"], state["g_preexp"], ids, wts, xm,
                                                    data["local_start"], lids, rt, P, data['stream'], g['alpha1'], g['beta1'],
                                                    g['alpha2'], s1, MODE=SL_ROUTE_CONFIG[T][1],
                                                    PARTS=SL_TILE_CONFIG[T][1], NCH=nch, TT=T,
                                                    num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    mx = _moe(T, state)
    if MOE_HAS_PDL:
        mx.okc_set_pdl(1 if _pdl(T) else 0)
    for m0 in range(0, T, _MA_MMAX):
        m1 = min(T, m0 + _MA_MMAX)
        mx.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32, state["nsm"])
    U, S = dense(m, state["unpool_slice"], T, state, "un")
    if S != 1 or U.dtype != torch.bfloat16:
        raise RuntimeError('gle: BF16 sliced unpooler expected')
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    nw, bc = SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T, 8192 // bc)](U, T, s1, rt[1], g['beta2'], g['post2'], g['pre2'], s2, out, BC=bc, num_warps=nw,
                                        launch_pdl=True)
    return out, s2, ids


_gle_prior_after = _after_attention
def _after_attention(data, state, o_in, gate, T):
    if (GLE_T.get(T, GLE_T.get(str(T), False)) and GLE_RC.get(T, 0) > 0 and gate is None and _pdl(T) and 'link' in state
            and state.get('compose_link_enabled', {}).get(T, True) and T in SL_TILE_CONFIG and T in SL_CLUSTER_CONFIG):
        mmp = state.get("mm_pick", {})
        ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
        opick = mmp.get((T, "o"), mmp.get((_bucket(T), "o"), False))
        moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "native")),
                                   marlin_available=state["bank"] is not None, native_available=state["ma_ok"])
        gw = state.get('g_preffn')
        if (ppick in ("bfp", "bfpf") and opick == "bfp" and moe == "native" and state.get("bfp") is not None
                and state.get("router_fp32") is not None and state.get("router_fp32_ext") is not None
                and gw is not None and gw.dtype == torch.bfloat16 and gw.numel() == HIDDEN and gw.is_contiguous()
                and _gle_o_choice(state, T)):
            return _gle_after_attention(data, state, o_in, gate, T)
    return _gle_prior_after(data, state, o_in, gate, T)


_gle_prior_prepare = prepare
def prepare(weights):
    _gle_ext()
    return _gle_prior_prepare(weights)


GLE_B1X = True
_GLE_W_CUR = None


def _gle_b1x_source():
    s = _SL_GLUE_CUDA
    for old, new in (
            ("float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];\n  __shared__ float partials[3][NW];",
             "float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N],nsx[N];\n  __shared__ float partials[4][NW];"),
            ("    if constexpr(POST) glb<V>(a.post+col,post+r*V);\n",
             "    if constexpr(POST || ROUTER) glb<V>(a.post+col,post+r*V);   // ROUTER: post = w (pre_ffn_norm)\n"),
            ("        float bn=gb(gm(b[k],r1));\n",
             "        float bn=gb(gm(b[k],r1));\n"
             "        if constexpr(ROUTER) {   // the router's FP32 copy: no BF16 rounding of r1 * P\n"
             "          float bx=gm(b[k],r1);\n"
             "          nsx[k]=(SLICED || (PRELOADED_BASE && col<4096)) ? gf(alpha[k],s[k],gm(beta[k],bx)) : gf(beta[k],bx,gm(alpha[k],s[k]));\n"
             "        }\n"),
            ("  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);\n",
             "  float r2, r2x=0.f;\n"
             "  if constexpr(ROUTER) {   // the router copy's sum of squares rides the same cluster exchange\n"
             "    float zs;\n"
             "    r2=gr(cluster_squares_triple<NW>(ns,masked,nsx,partials[1],partials[2],partials[3],cluster,rank,tail,zs),1.e-5f);\n"
             "    r2x=gr(zs,1.e-5f);\n"
             "  } else {\n"
             "    r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);\n"
             "  }\n"),
            ("template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n",
             "template<int NW,int N>\n"
             "__device__ __forceinline__ float cluster_squares_triple(const float (&x)[N], const float (&y)[N], const float (&z)[N],\n"
             "    float* sm, float* smy, float* smz, cg::cluster_group cluster, int rank, float& tail, float& zsum) {\n"
             "  constexpr int warps_per_cta=NW/8;\n"
             "  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;\n"
             "  float sx=butterfly(local_squares(x),16);\n"
             "  float sy=butterfly(local_squares(y),16);\n"
             "  float sz=butterfly(local_squares(z),16);\n"
             "  if(lane==0 && warp<warps_per_cta) {\n"
             "    #pragma unroll\n"
             "    for(int dst=0;dst<8;dst++) {\n"
             "      volatile float* px=cluster.map_shared_rank(sm,dst);\n"
             "      volatile float* py=cluster.map_shared_rank(smy,dst);\n"
             "      volatile float* pz=cluster.map_shared_rank(smz,dst);\n"
             "      px[rank*warps_per_cta+warp]=sx;\n"
             "      py[rank*warps_per_cta+warp]=sy;\n"
             "      pz[rank*warps_per_cta+warp]=sz;\n"
             "    }\n"
             "  }\n"
             "  cluster.sync();\n"
             "  tail=butterfly(smy[lane&(NW-1)],NW/2);\n"
             "  zsum=butterfly(smz[lane&(NW-1)],NW/2);\n"
             "  return butterfly(sm[lane&(NW-1)],NW/2);\n"
             "}\n\n"
             "template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n"),
            ("      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);\n",
             "      if constexpr(ROUTER) {\n"
             "        float hx[V];\n"
             "        #pragma unroll\n"
             "        for(int j=0;j<V;j++) hx[j]=gm(gm(nsx[r*V+j],r2x),ga(1.f,post[r*V+j]));\n"
             "        gsf<V>(a.router+row*8192+col,hx);\n"
             "      }\n"),
            ("void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {\n"
             "  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),",
             "void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router,TV w) {\n"
             "  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)w.data_ptr(),(const BF*)pre.data_ptr(),")):
        assert s.count(old) == 1, old[:70]
        s = s.replace(old, new, 1)
    return s


@_lru_cache(None)
def _gle_b1x_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    src = _gle_b1x_source()
    return _tvm_load_inline(name='gle_b1x_' + _hashlib.md5(src.encode()).hexdigest()[:12], cuda_sources=src,
                            functions=['b1'], extra_cuda_cflags=['-O3', '-std=c++17'])


class _GleB1x:
    """_sl_cluster_module() stand-in: the router-writing b1 goes to the exact-FP32 copy (w from the calling state)."""
    def __init__(self, base):
        self.base = base

    def b1(self, branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, write_router):
        if write_router and GLE_B1X and _GLE_W_CUR is not None:
            return _gle_b1x_module().b1(branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, True, _GLE_W_CUR)
        return self.base.b1(branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, write_router)


_gle_b1x_prior_module = _sl_cluster_module
def _sl_cluster_module():
    return _GleB1x(_gle_b1x_prior_module())


_gle_b1x_prior_after = _after_attention
def _after_attention(data, state, o_in, gate, T):
    global _GLE_W_CUR
    gw = state.get('g_preffn')
    _GLE_W_CUR = gw if (gw is not None and gw.dtype == torch.bfloat16 and gw.numel() == HIDDEN and gw.is_contiguous()) else None
    return _gle_b1x_prior_after(data, state, o_in, gate, T)


_gle_b1x_prior_prepare = prepare
def prepare(weights):
    _gle_b1x_module()
    return _gle_b1x_prior_prepare(weights)


# ---- opus_glue: pin the T <= 8 chain. 0389's compose tuner picks the T = 1 / T = 8 chain by timing inside prepare; its
# older link chain feeds the FP32 router a BF16-rounded b1 copy (2e-3 rel). Force the chain with b1x / gle / the fused FP32
# router (exact FP32 router input) on every prepare.
_pin_parent_prepare = prepare
def prepare(weights):
    st = _pin_parent_prepare(weights)
    cge = st.setdefault('compose_link_enabled', {})
    for T in (1, 8):
        cge[T] = True
    return st


