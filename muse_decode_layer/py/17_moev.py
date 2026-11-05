def lab_source(cuda_src):
    def sub(s, old, new, cnt=1):
        assert s.count(old) == cnt, (s.count(old), old[:90])
        return s.replace(old, new)

    def norm(src):   # drop // comments, trailing blanks and empty lines (the decode file's lab copy is minified)
        out = []
        for l in src.split("\n"):
            i = l.find("//")
            if i >= 0 and l[:i].count('"') % 2 == 0:
                l = l[:i]
            l = l.rstrip()
            if l.strip():
                out.append(l)
        return "\n".join(out) + "\n"
    s = norm(cuda_src)
    # --- dq4 (IMADROT)
    s = sub(s, """  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
""", """#if OKC_LAB & 16
  r1 = (q * 16u) & M;
  {
    uint32_t lo, hi;
    asm("{ .reg .b64 t; mul.wide.u32 t, %2, 1024; mov.b64 {%0, %1}, t; }" : "=r"(lo), "=r"(hi) : "r"(q));
    r2 = (lo | hi) & M;
  }
  {
    uint32_t lo, hi;
    asm("{ .reg .b64 t; mul.wide.u32 t, %2, 16384; mov.b64 {%0, %1}, t; }" : "=r"(lo), "=r"(hi) : "r"(q));
    r3 = ((q * 64u) & 0x80C080C0u) | ((lo | hi) & 0x01000100u);
  }
#elif OKC_LAB & 512
  r1 = __funnelshift_l(q, q, 4) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = (__funnelshift_l(q, q, 6) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
#else
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
#endif
  return;
""")
    # --- GS per n-tile count
    s = sub(s, "  constexpr int GS = OKC_WGSGS;\n",
            "#ifndef OKC_LABGS4N\n#define OKC_LABGS4N 2\n#endif\n  constexpr int GS = ((OKC_LAB & 8) && NTT <= OKC_LABGS4N) ? 4 : OKC_WGSGS;\n")
    s = sub(s, "if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);",
            "if (FIRST && ci == (OKC_WGSFCI * 2) / GS && pd.kind) wgs_flush<PB>(p, S, pd);")
    # --- NODQ
    s = sub(s, """          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
""", """#if OKC_LAB & 2
          af[ci & 1][jj][0] = q; af[ci & 1][jj][1] = q; af[ci & 1][jj][2] = q; af[ci & 1][jj][3] = q;
#else
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
""")
    s = sub(s, """          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
""", """#if OKC_LAB & 32
          af[ci & 1][jj][0] = r0; af[ci & 1][jj][1] = r1; af[ci & 1][jj][2] = r2; af[ci & 1][jj][3] = r3;
#else
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
#endif
#endif
""")
    # the scale words are loaded before; with NODQ they are dead: wrap the sgp build
    s = sub(s, """#if OKC_SC6W
          const uint32_t sgp = sgs[js];
#else
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#endif
""", """#if OKC_LAB & 2
#elif OKC_SC6W
          const uint32_t sgp = sgs[js];
#else
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#endif
""")
    # --- NOMMA
    s = sub(s, """          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
""", """#if OKC_LAB & 1
          if (FIRST && ci == 0 && jj == 0) {
#pragma unroll
            for (int n = 0; n < NTT; ++n) { d[n][0] = 0.f; d[n][1] = 0.f; d[n][2] = 0.f; d[n][3] = 0.f; }
          }
          asm volatile("" : "+f"(d[0][0]) : "r"(af[ci & 1][jj][0]), "r"(af[ci & 1][jj][1]), "r"(af[ci & 1][jj][2]), "r"(af[ci & 1][jj][3]), "l"(xd));
#else
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
#endif
""")
    # --- NOX
    s = sub(s, """      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
""", """#if OKC_LAB & 4
      (void)bytes;
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" ::"r"(smem_u32(bar)) : "memory");
#else
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
#endif
""")
    # --- FP1: single-read final pass (pass 2 reuses the rows pass 1 loaded)
    ld1 = """          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
"""
    assert s.count(ld1) == 2, s.count(ld1)
    k1 = s.index(ld1)
    k2 = s.index(ld1, k1 + 1)
    new1 = """#if OKC_LAB & 128
          const uint4 yr = (ykeep[sl] = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid));
#else
""" + ld1 + "#endif\n"
    new2 = """#if OKC_LAB & 128
          const uint4 yr = ykeep[sl];
#else
""" + ld1 + "#endif\n"
    s = s[:k1] + new1 + s[k1 + len(ld1):k2] + new2 + s[k2 + len(ld1):]
    s = sub(s, """      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
""", """      float ps[8];
#if OKC_LAB & 128
      uint4 ykeep[8];
#endif
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
""")
    # --- PF: L2 prefetch of the next phase-A stages by the weight producer
    s = sub(s, """      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}
constexpr int WGS_XROWS""", """      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
#if OKC_LAB & 1024
      if (OKC_DEBUG && p.dbg && i == 0) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + (pb ? 11 : 8)] = gtimer();
#endif
#if OKC_LAB & 4096
      {  // continuous L2 prefetch: OKC_LABPFK stages beyond the ring (inside this phase's range)
        const int ii = i + WGS_NS + OKC_LABPFK;
        if (ii < g1 - g0) {
          int g2 = sp + ii;
          if (g2 >= g1) g2 = g0 + (g2 - g1);
          const int u2 = g2 >> 4, st2 = g2 & 15, j2 = u2 >> ubl, ub2 = u2 & ((1 << ubl) - 1);
          const unsigned char* src2 = base + ((size_t)(((S.jE[j2] << ubl) + ub2) * 16 + st2)) * (size_t)WGS_STG;
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\\n" ::"l"(src2), "r"(WGS_STG) : "memory");
        }
      }
#endif
#if OKC_LAB & 256
      if (pb == 0 && n == WGS_NS - 1) {
        for (int k = 0; k < OKC_LABPF; ++k) {
          const int ii = WGS_NS + k;
          if (ii >= g1 - g0) break;
          int g2 = sp + ii;
          if (g2 >= g1) g2 = g0 + (g2 - g1);
          const int u2 = g2 >> 4, st2 = g2 & 15, j2 = u2 >> 5, ub2 = u2 & 31;
          const unsigned char* src2 = base + ((size_t)(((S.jE[j2] << 5) + ub2) * 16 + st2)) * (size_t)WGS_STG;
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\\n" ::"l"(src2), "r"(WGS_STG) : "memory");
        }
      }
#endif
    }
  }
}
constexpr int WGS_XROWS""")
    # --- SMID stamp (OKC_DEBUG builds)
    s = sub(s, "  DBG(0);\n  asm volatile(\"griddepcontrol.wait;",
            "  DBG(0);\n#if OKC_LAB & 1024\n  if (OKC_DEBUG && tid == 0 && p.dbg) { unsigned smid_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(smid_)); p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 31] = smid_; }\n#endif\n  asm volatile(\"griddepcontrol.wait;")
    # --- EARLY: weight producer joins the routing barrier (7) and a ranges barrier (8), not the post-gather WS_JOIN
    s = sub(s, """  asm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 32) : "memory");
""", """#if OKC_LAB & 2048
  asm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 64) : "memory");  // both producers
#else
  asm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 32) : "memory");
#endif
""")
    # the x producer's prologue: one lane (pub_k) or all 32 lanes (sub_v18: warp-parallel wgs_ranges) compute the stage ranges
    for call in ("  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);\n", "  wgs_ranges(p, S);\n"):
        old_x = """  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");
""" + call + """  __syncwarp();
  WS_JOIN();
"""
        if s.count(old_x) == 1:
            new_x = """#if OKC_LAB & 2048
  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 64) : "memory");
""" + call + """  __syncwarp();
  asm volatile("bar.arrive 8, 64;\\n" ::: "memory");
  asm volatile("bar.sync 2, %0;\\n" ::"n"(NT + 32) : "memory");
#else
  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");
""" + call + """  __syncwarp();
  WS_JOIN();
#endif
"""
            s = s.replace(old_x, new_x)
            break
    else:
        raise AssertionError("x producer prologue anchor")
    s = sub(s, """__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();
""", """__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_LAB & 2048
  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 64) : "memory");
  asm volatile("bar.sync 8, 64;\\n" ::: "memory");
#else
  WS_JOIN();
#endif
""")
    s = sub(s, """  DBG(7);
  WS_JOIN();
  DBG(6);
""", """  DBG(7);
#if OKC_LAB & 2048
  asm volatile("bar.sync 2, %0;\\n" ::"n"(NT + 32) : "memory");
#if OKC_LAB & 2097152
  if (tid == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" ::"r"(smem_u32(S.bars + 23)) : "memory");
#endif
#else
  WS_JOIN();
#endif
  DBG(6);
""")
    # --- extra timeline stamps (OKC_DEBUG builds with LAB bit 1024): 8 wprod A issue, 9 xprod A copy, 10 consumers A stage ready, 11-13 the same for B
    s = sub(s, """      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
""", """#if OKC_LAB & 1024
      if (OKC_DEBUG && p.dbg && i == 0) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + (pb ? 12 : 9)] = gtimer();
#endif
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
""")
    s = sub(s, """    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
""", """    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
#if OKC_LAB & 1024
    if (OKC_DEBUG && p.dbg && tid == 0 && ((!PB && seq == 0) || (PB && seq == S.wgr[1] - S.wgr[0]))) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + (PB ? 13 : 10)] = gtimer();
#endif
""")
    # --- EARLY_AI (8192, with 2048): the weight producer also waits (barrier 9) until the consumers issued their first x-gather loads
    # --- gather stamps (1024): 14 start, 15 first batch issued, 16 after the CTA_SYNC, 17 rows of this CTA, 18 total padded rows
    s = sub(s, """  constexpr int GB = OKC_GATHB;
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[GB];
""", """  constexpr int GB = OKC_GATHB;
#if OKC_LAB & 8192
  bool ai_ = false;
#endif
#if OKC_LAB & 1024
  if (OKC_DEBUG && p.dbg && tid == 0) { p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 14] = gtimer(); p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 17] = q1 - q0; p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 18] = R; }
#endif
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[GB];
""")
    s = sub(s, """#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
  CTA_SYNC();
""", """#if OKC_LAB & 8192
    if (!ai_) { ai_ = true; asm volatile("bar.arrive 9, %0;\\n" ::"n"(NT + 32) : "memory"); }
#endif
#if OKC_LAB & 1024
    if (OKC_DEBUG && p.dbg && tid == 0 && qb == q0) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 15] = gtimer();
#endif
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
#if OKC_LAB & 8192
  if (!ai_) asm volatile("bar.arrive 9, %0;\\n" ::"n"(NT + 32) : "memory");
#endif
  CTA_SYNC();
#if OKC_LAB & 1024
  if (OKC_DEBUG && p.dbg && tid == 0) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 16] = gtimer();
#endif
""")
    s = sub(s, """  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 64) : "memory");
  asm volatile("bar.sync 8, 64;\\n" ::: "memory");
#else""", """  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 64) : "memory");
  asm volatile("bar.sync 8, 64;\\n" ::: "memory");
#if OKC_LAB & 8192
  asm volatile("bar.sync 9, %0;\\n" ::"n"(NT + 32) : "memory");
#endif
#else""")
    # --- FIN2 (131072): the final pass handles two tokens per iteration (tk0 = cta, tk1 = cta + grid): one poll round, one load round, one pair of CTA_SYNCs
    s = sub(s, """    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {""", """#if OKC_LAB & 131072
    for (int tk0 = blockIdx.x; tk0 < p.M; tk0 += 2 * gridDim.x) {
      const int tk1 = tk0 + gridDim.x;
      const bool has1 = tk1 < p.M;
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 16) {
        const int tkx = tid < 8 ? tk0 : tk1;
        if ((tid < 8 || has1) && S.pos[tkx * 8 + (tid & 7)] >= 0) {
          const int e = S.eid[tkx * 8 + (tid & 7)], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
      if (tk0 == blockIdx.x) DBG(4);
#endif
      int qv[2][8];
      float ps[2][8];
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) qv[t][sl] = (t == 0 || has1) ? S.pos[(t ? tk1 : tk0) * 8 + sl] : -1;
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) {
          ps[t][sl] = 0.f;
          if (qv[t][sl] >= 0) {
            const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[t][sl] * HD) + tid);
            const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
            const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
            const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
            ps[t][sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
          }
        }
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int sl = 0; sl < 8; ++sl)
          if (qv[t][sl] >= 0) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ps[t][sl] += __shfl_xor_sync(0xffffffffu, ps[t][sl], o);
          }
      if (lane == 0) {
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
          for (int sl = 0; sl < 8; ++sl) red[warp * 16 + t * 8 + sl] = ps[t][sl];
      }
      CTA_SYNC();
      if (tid < 16) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 16 + tid];
        red[NW * 16 + tid] = tsum;
      }
      CTA_SYNC();
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        if (t == 0 || has1) {
          const int tk = t ? tk1 : tk0;
          float acc[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
          for (int sl = 0; sl < 8; ++sl)
            if (qv[t][sl] >= 0) {
              const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 16 + t * 8 + sl] * (1.f / HD) + 1e-8f);
              const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[t][sl] * HD) + tid);
              const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
              const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
              const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
              acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
              acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
            }
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
            ow[i] = *reinterpret_cast<uint32_t*>(&v);
          }
          *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
        }
      }
      CTA_SYNC();
    }
#else
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {""")
    s = sub(s, """      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      CTA_SYNC();
    }
  }
#elif OKC_FINCOL""", """      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      CTA_SYNC();
    }
#endif
  }
#elif OKC_FINCOL""")
    # --- FP2 (4194304, use with FP1 128): the final pass loads the token's 8 route weights before the counter poll (they were loaded after the two block syncs)
    s = sub(s, """#else
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {""", """#else
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_LAB & 4194304
      float wv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) wv[sl] = __ldg(p.topk_w + tk * 8 + sl);
#endif
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {""")
    s = sub(s, """          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
""", """#if OKC_LAB & 4194304
          const float r = wv[sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
#else
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
#endif
""")
    # --- final-pass stamps (1024): 19 after pass 1 + first CTA_SYNC, 20 after the second CTA_SYNC (first token of the CTA)
    s = sub(s, """        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
""", """        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
#if OKC_LAB & 1024
      if (OKC_DEBUG && p.dbg && tid == 0 && tk == blockIdx.x) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 19] = gtimer();
#endif
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
#if OKC_LAB & 1024
      if (OKC_DEBUG && p.dbg && tid == 0 && tk == blockIdx.x) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 20] = gtimer();
#endif
""")
    # --- WGSDFIN overridable (-DOKC_WGSDFIN=3: designated finisher in phase A too); gen-atomic stamps 22 (before) / 23 (after) under bit 1024
    s = sub(s, "#define OKC_WGSDFIN 2\n", "#ifndef OKC_WGSDFIN\n#define OKC_WGSDFIN 2\n#endif\n")
    s = sub(s, "    const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;\n",
            "#if OKC_LAB & 1024\n    if (OKC_DEBUG && p.dbg) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 22] = gtimer();\n#endif\n"
            "    const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;\n"
            "#if OKC_LAB & 1024\n    if (OKC_DEBUG && p.dbg) p.dbg[gridDim.x * 8 + blockIdx.x * 32 + 23] = gtimer();\n#endif\n")
    # --- FENCE1 (262144): the gather tail issues ONE fence.acq_rel.gpu and relaxed reds (release pattern) instead of one red.release per job
    # --- GATE (524288): the weight producer waits (relaxed poll, timing only) until every CTA finished its x gather (counter xready[MF_XRS - 1])
    # --- STAG (1048576): the weight producer issues stages 1 and 2 only after stage 0 landed (first MMA does not wait for the 3-stage burst)
    s = sub(s, """  if (tid == 0) {
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
""", """  if (tid == 0) {
    int jj = 0;
#if OKC_LAB & 262144
    asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");
#endif
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
#if OKC_LAB & 262144
      asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");
#else
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");
#endif
      q = e;
    }
#if OKC_LAB & 524288
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;\\n" ::"l"(mf_xr(p, S) + (MF_XRS - 1)) : "memory");
#endif
  }
""")
    s = sub(s, """  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
""", """  if ((threadIdx.x & 31) != 0) return;
#if OKC_LAB & 524288
  {
    const int* gate = mf_xr(p, S) + (MF_XRS - 1);
    int gv;
    for (int it_ = 0; it_ < 300000; ++it_) {
      asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\\n" : "=r"(gv) : "l"(gate) : "memory");
      if (gv >= (int)gridDim.x - OKC_GATETOL) break;
      __nanosleep(32);
    }
  }
#endif
  int ws = 0, n = 0;
  uint32_t wph = 0;
""")
    s = sub(s, """      if (n >= 2 && p.M <= 64) { const int pn = n - 2; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }
""", """      if (n >= 2 && p.M <= 64) { const int pn = n - 2; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }
#if OKC_LAB & 2097152
      if (n == 1) wgs_wait(S.bars + 23, 0u, 7000 + gs);
#endif
#if OKC_LAB & 1048576
      if (n == 1 || n == 2) wgs_wait(S.bars + BW_F, 0u, 6000 + gs);
#endif
""")
    # --- EARLY0 (2097152, with 2048): bars[23] = "this CTA's x gather is done" (one arrival from consumer thread 0)
    s = sub(s, """    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
""", """    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
#if OKC_LAB & 2097152
    static_assert(BX_E + WGS_NX <= 23, "EARLY0 barrier slot");
    mbar_init(S.bars + 23, 1);
#endif
""")
    # --- OKC_DEBUG must be overridable from the command line (the source defines it unconditionally)
    s = sub(s, "#define OKC_DEBUG 0\n", "#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif\n")
    # --- SC5W: 5-bit lossless block scales per warpgroup piece (base byte in a 16 B header, 5-bit offsets; -DOKC_SC5W=1, no SC6W)
    s = sub(s, "constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;", "constexpr int WGS_PIECE = OKC_SC5W ? 8848 : OKC_SC6W ? 8960 : 9216;")
    s = sub(s, "constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;", "constexpr int WGS_STG = OKC_SC5W ? (OKC_SC5W == 3 ? 35584 : OKC_SC5W == 2 ? 35456 : 35392) : OKC_SC6W ? 35840 : 36864;")
    s = sub(s, """#if OKC_SC6W
    uint32_t sgs[4];""", """#if OKC_SC5W
    // SC5: 16 B header (C = (base * 16 + 0x5C00) in both halves), plane P (uint32 per thread: v0 | v1 << 16, v2 << 5 | v3 << 21,
    // v4 << 10 | v5 << 26, bit 15 / 31 = bit 4 of v6 / v7) and plane R (byte per thread: v6 low nibble | v7 low nibble << 4), v = scale byte - base
    uint32_t sgs[4];
    {
      const int tt = wp * 32 + lane;
      const uint32_t C = *reinterpret_cast<const uint32_t*>(pc + 8192);
      const uint32_t P = *reinterpret_cast<const uint32_t*>(pc + 8208 + tt * 4);
      const uint32_t Rb = static_cast<uint32_t>(pc[8208 + 512 + tt]);
      sgs[0] = (P & 0x001F001Fu) * k16c + C;
      sgs[1] = ((P >> 5) & 0x001F001Fu) * k16c + C;
      sgs[2] = ((P >> 10) & 0x001F001Fu) * k16c + C;
      sgs[3] = (((Rb * 0x1001u) & 0x000F000Fu) | ((P >> 11) & 0x00100010u)) * k16c + C;
    }
#elif OKC_SC6W
    uint32_t sgs[4];""")
    s = sub(s, """#if !OKC_SC6W
      const uint2 scg = scs;
#endif""", """#if !OKC_SC6W && !OKC_SC5W
      const uint2 scg = scs;
#endif""")
    s = sub(s, """#if OKC_LAB & 2
#elif OKC_SC6W
          const uint32_t sgp = sgs[js];""", """#if OKC_LAB & 2
#elif OKC_SC6W || OKC_SC5W
          const uint32_t sgp = sgs[js];""")
    # --- release probes: 16384 NOREL (every red.release.gpu -> red.relaxed.gpu: timing probe only, not exact), 32768 NOFENCE
    # (the fence.acq_rel.gpu before the split-unit fixup read-back removed: timing probe), 65536 RELSTAMP (OKC_DEBUG builds: globaltimer
    # around every release / the fence in wgs_flush, summed + 4-bin histogram per site into dbg[G*40 + cta*32 + site*5 ..])
    import re
    if "void wgs_flush(" in s:
        i0 = s.index("__device__ __forceinline__ void wgs_flush(")
        i1 = s.index("__device__ __forceinline__ void wgs_seg(")
        seg = s[i0:i1]
        pat = re.compile(r'asm volatile\("red\.release\.gpu\.global\.add\.s32 \[%0\], %1;\\n" ::[^;]*?: "memory"\)')
        cnt = [0]
        def rep(m):
            k = cnt[0]; cnt[0] += 1
            return "RS(%d, %s)" % (k, m.group(0))
        seg = pat.sub(rep, seg)
        assert cnt[0] in (4, 5), cnt[0]
        fo = r'asm volatile("fence.acq_rel.gpu;\n" ::: "memory");'
        assert seg.count(fo) == 1
        seg = seg.replace(fo, "#if !(OKC_LAB & 32768)\n  RS(5, asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\"));\n#endif")
        s = s[:i0] + seg + s[i1:]
        rs_def = r"""#if (OKC_LAB & 65536) && OKC_DEBUG
__device__ __forceinline__ unsigned gtimer_lo() { unsigned t; asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(t)); return t; }
__device__ __noinline__ void rs_rec(const Params& p, int site, unsigned dt) {
  if (!p.dbg || (threadIdx.x & 127) != 0) return;
  unsigned long long* b = p.dbg + (size_t)gridDim.x * 40 + (size_t)blockIdx.x * 32 + site * 5;
  atomicAdd(b, (unsigned long long)dt);
  atomicAdd(b + 1 + (int)(dt >= 250u) + (int)(dt >= 1000u) + (int)(dt >= 4000u), 1ull);
}
#define RS(site, ...) do { const unsigned t0_ = gtimer_lo(); __VA_ARGS__; rs_rec(p, site, gtimer_lo() - t0_); } while (0)
#else
#define RS(site, ...) do { __VA_ARGS__; } while (0)
#endif
"""
        s = sub(s, "template <bool PB>\n__device__ __forceinline__ void wgs_flush(", rs_def + "template <bool PB>\n__device__ __forceinline__ void wgs_flush(")
    s = s.replace('"red.release.gpu.global', '"red." OKC_RELSEM ".gpu.global')
    s = '#if OKC_LAB & 16384\n#define OKC_RELSEM "relaxed"\n#else\n#define OKC_RELSEM "release"\n#endif\n' + s
    s = "#ifndef OKC_SC5W\n#define OKC_SC5W 0\n#endif\n#ifndef OKC_GATETOL\n#define OKC_GATETOL 0\n#endif\n" + s
    s = "#ifndef OKC_LAB\n#define OKC_LAB 0\n#endif\n#ifndef OKC_LABPF\n#define OKC_LABPF 3\n#endif\n#ifndef OKC_LABPFK\n#define OKC_LABPFK 2\n#endif\n" + s
    return s

# SC5W host packing (lossless 5-bit block scales per warpgroup piece).
def _sc5_ok(s):
    """s: uint8 [..., 1024] (one warpgroup piece per row): True if every piece's scale bytes span <= 31."""
    return int((s.amax(dim=-1).to(torch.int32) - s.amin(dim=-1).to(torch.int32)).max()) <= 31


def _sc5_wgs(s, strict=True):
    """SC5 WGS piece scales: [..., 1024] bytes (thread t = 8 bytes s0..s7 = [j][h]) -> [..., 656] = 16 B header (u32 C = (base * 16 + 0x5C00) in both
    16-bit halves, base = the piece's smallest scale byte) | plane P (t * 4 B: v0 | v1 << 16, v2 << 5 | v3 << 21, v4 << 10 | v5 << 26,
    bit 15 / 31 = bit 4 of v6 / v7) | plane R (t * 1 B: v6 & 15 | (v7 & 15) << 4), v_n = scale byte n - base (0..31)."""
    lead = s.shape[:-1]
    b = s.view(*lead, 128, 8).to(torch.int64)
    base = b.amin(dim=(-2, -1))
    v = b - base[..., None, None]
    if strict:
        assert int(v.max()) <= 31 and int(v.min()) >= 0
    else:
        v = v.clamp(0, 31)
    P = (v[..., 0] | (v[..., 1] << 16) | (v[..., 2] << 5) | (v[..., 3] << 21) | (v[..., 4] << 10) | (v[..., 5] << 26)
         | (((v[..., 6] >> 4) & 1) << 15) | (((v[..., 7] >> 4) & 1) << 31))
    R = (v[..., 6] & 15) | ((v[..., 7] & 15) << 4)
    pb = torch.stack([(P >> (8 * i)) & 0xFF for i in range(4)], -1).to(torch.uint8).reshape(*lead, 512)
    C = ((base * 16 + 0x5C00) * 0x10001) & 0xFFFFFFFF
    hb = torch.cat([torch.stack([(C >> (8 * i)) & 0xFF for i in range(4)], -1).to(torch.uint8), torch.zeros(*lead, 12, dtype=torch.uint8, device=s.device)], -1)
    return torch.cat([hb, pb, R.to(torch.uint8)], -1).contiguous()

_MOEV = {"ctx": None}
_MOEV_BASE_MOE = _moe


def _wgs_copy5(codes, sf, rows):
    """_wgs_copy (file) with the SC5W scale pieces."""
    wn = _pack_codes_nat(_kperm_codes(codes), rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)
    s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)
    s = _sc5_wgs(s)   # asserts span <= 31
    nub = cb.shape[0]
    o = torch.cat([cb, s], dim=3).contiguous().view(nub, 16, 4 * 8848)
    o = torch.cat([o, torch.zeros(nub, 16, 64, dtype=torch.uint8, device=o.device)], dim=2)   # stage stride 35456 = 277 x 128 B (alignment matters)
    return o.contiguous().view(-1)


class _MoevExt:
    """Same okc_run contract as the file's ext; the caller's state["ma_w"] is replaced by the candidate's weight list."""
    def __init__(self, ext, w):
        self._e, self._w = ext, w

    def okc_run(self, w, *a):
        return self._e.okc_run(self._w, *a)

    def __getattr__(self, n):
        return getattr(self._e, n)

    def __deepcopy__(self, memo):
        return self


def _moev_ext(M, ctx):
    key = "b4" if 16 < M <= 32 else "main"
    ext = ctx["ext"].get(key)
    if ext is None:
        base = tuple(MOE_DEFINES) + (("-DOKC_B4FIN=1",) if key == "b4" else ())
        d = base + ("-DOKC_LAB=%d" % MOEV_BITS,) + tuple(MOEV_EXTRA) + (("-DOKC_SC6OFF=1", "-DOKC_SC5W=2") if ctx.get("sc5", MOEV_SC5W) else ()) + ("@moev",) + ctx["sfx"]
        e = _ma_ext(d)
        if MOE_HAS_PDL:
            e.okc_set_pdl(0)
        ext = ctx["ext"][key] = _MoevExt(e, ctx["w"])
    return ext


def _moev_selftest(ctx, st, dev):
    """Variant vs the base kernel on hot-expert routing at each T: relative L2 of the bf16 outputs must be < 2e-3 (the candidate is lossless; 2e-3 only absorbs atomics-order noise). A broken repack / decode gives O(1)."""
    try:
        g = torch.Generator(device=dev).manual_seed(4242)
        for T in MOEV_SELFTEST_TS:
            x = torch.randn((T, 4096), generator=g, device=dev).to(torch.bfloat16)
            sc = torch.rand((T, 256), generator=g, device=dev)
            sc[:, :32] += 0.03
            sc[:, 0] += 0.35
            sc[:, 1] += 0.25
            sc[:, 2] += 0.15
            top = sc.topk(8, dim=-1)
            ids = top.indices
            lids = torch.where(ids < 32, ids, torch.full_like(ids, -1)).to(torch.int32).contiguous()
            wts = (top.values / top.values.sum(-1, keepdim=True)).float().contiguous()
            ob = torch.zeros((T, 4096), dtype=torch.bfloat16, device=dev)
            oc = torch.zeros((T, 4096), dtype=torch.bfloat16, device=dev)
            be = _MOEV_BASE_MOE(T, st)
            be.okc_run(st["ma_w"], st["ma_scratch"], x, lids, wts, ob, 0, 32, st["nsm"])
            ce = _moev_ext(T, ctx)
            ce.okc_run(st["ma_w"], st["ma_scratch"], x, lids, wts, oc, 0, 32, st["nsm"])
            torch.cuda.synchronize()
            rel = float((oc.float() - ob.float()).norm() / ob.float().norm().clamp(min=1e-20))
            print(f"[moev] selftest T={T}: rel L2 vs the file's kernel {rel:.3e}, local routes {int((lids >= 0).sum())}", flush=True)
            if not rel < 2e-3:
                return False
        return True
    except Exception as ex:
        print(f"[moev] selftest failed with {str(ex)[-300:]}", flush=True)
        return False


def _moev_setup(st, weights, dev):
    sfx = tuple(st["ma_sfx"])
    if "-DOKC_PRMTSC=1" not in sfx or not _sc6_ok(weights["w13_sf"], weights["w2_sf"]):
        raise RuntimeError("moev: scales do not fit the 6-bit form (or no PRMT decode)")
    _MA_ALT_SRCS["moev"] = lab_source(MOE_CUDA_SRC)
    w = list(st["ma_w"])
    sc5 = bool(MOEV_SC5W)
    if sc5:
        try:   # _sc5_wgs asserts that every piece's scale bytes span <= 31; weights that do not fit keep the 6-bit form (the other patches still apply)
            I_ = 4096
            rb = torch.arange(I_ // 8, device=dev)
            g = torch.arange(8, device=dev)
            rowsA = torch.stack((8 * rb[:, None] + g[None, :], I_ + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
            rowsB = torch.arange(4096, device=dev)
            w12 = torch.cat([_wgs_copy5(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], _kperm_rows(rowsA)) for e in range(32)])
            w13 = torch.cat([_wgs_copy5(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB) for e in range(32)])
            w[12], w[13] = w12, w13
        except AssertionError:
            sc5 = False
            w = list(st["ma_w"])
            print("[moev] SC5W scale span > 31: keeping the 6-bit scale form", flush=True)
    ctx = {"st": st, "w": w, "sfx": sfx, "ext": {}, "sc5": sc5, "off": False}
    _moev_ext(MOEV_MIN_M, ctx)   # the one candidate build (key "main"), shared by every M >= MOEV_MIN_M
    ctx["off"] = not _moev_selftest(ctx, st, dev)
    return ctx


def _moev_moe(M=None, state=None):
    ctx = _MOEV["ctx"]
    if ctx is None or ctx["off"] or state is None or M is None or M < MOEV_MIN_M:
        return _MOEV_BASE_MOE(M, state)
    return _moev_ext(M, ctx)


# ---- build the variant's extension in a thread while prepare runs, then repack the scales,
# self-test it against the base MoE kernel and rebind _moe. Every failure (or a prepare that ran too long) keeps the base kernel.
import threading as _mv_thr
import time as _mv_time

_MV_PRIOR_PREPARE = prepare
_MV_GUARD_S = 225.0     # no setup after this many seconds into prepare() (the hard limit is 300 s)


def _mv_defs(weights):
    sfx = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
    return (tuple(MOE_DEFINES) + ("-DOKC_LAB=%d" % MOEV_BITS,) + tuple(MOEV_EXTRA)
            + (("-DOKC_SC6OFF=1", "-DOKC_SC5W=2") if MOEV_SC5W else ()) + ("@moev",) + tuple(sfx))


def _mv_prebuild(weights, box):
    try:
        _MA_ALT_SRCS["moev"] = lab_source(MOE_CUDA_SRC)
        _ma_ext(_mv_defs(weights))
        box["ok"] = True
    except BaseException as ex:   # noqa: BLE001
        box["err"] = ex


def prepare(weights):
    t0 = _mv_time.time()
    box, th = {}, None
    try:
        if _MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"]):
            th = _mv_thr.Thread(target=_mv_prebuild, args=(weights, box), daemon=True)
            th.start()
    except BaseException as ex:   # noqa: BLE001
        th = None
        print(f"[moev] prebuild not started: {type(ex).__name__}: {str(ex)[-200:]}", flush=True)
    st = _MV_PRIOR_PREPARE(weights)
    try:
        if th is None:
            raise RuntimeError("no PRMT scale decode")
        th.join(max(1.0, _MV_GUARD_S - (_mv_time.time() - t0)))
        if th.is_alive() or "err" in box or _mv_time.time() - t0 > _MV_GUARD_S:
            raise RuntimeError(f"build not ready in time ({_mv_time.time() - t0:.0f}s): {box.get('err')}")
        ctx = _moev_setup(st, weights, weights["qkv_gate"].device)
        _MOEV["ctx"] = ctx
        globals()["_moe"] = _moev_moe
        print(f"[moev] ready at {_mv_time.time() - t0:.0f}s, self-test {'ok' if not ctx['off'] else 'FAILED (file kernel kept)'}, sc5 {ctx['sc5']}", flush=True)
    except BaseException as ex:   # noqa: BLE001
        print(f"[moev] off: {type(ex).__name__}: {str(ex)[-300:]}", flush=True)
    return st


_nr1_prior = bfp__run_sk2


def _nr1_run(x, layer, m, n, k, state, dg=0, dbg=None):
    if layer != "qkv_gate" or m != 1:
        return _nr1_prior(x, layer, m, n, k, state, dg, dbg)
    name = bfp__sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    ext = _de_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, bfp__sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


bfp__run_sk2 = _nr1_run


# ---- NORH at T = 32 (appended after the NORH block): link chain + fused pooler/router pinned at T = 32, no-handshake kernels, kernel-level bit-identity self-test.
_N32_OK = {}


def _n32_selftest(st, dev):
    gen = torch.Generator(device=dev).manual_seed(12)
    bst = st["bfp"]
    ok = True
    for m in (32, 24):
        x = torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16)
        pb, lb = bfp_fused_pooler_router(x, bst)
        pn, ln = _nr_bfp_fused(x, bst)
        torch.cuda.synchronize()
        ok &= bool(torch.equal(_nr_logits_from(_NR_PART.pop(ln.data_ptr()), m), lb)) and bool(torch.equal(pn, pb))
    _NR_PART.clear()
    return bool(ok)


_n32_prior_prepare = prepare


def prepare(weights):
    st = _n32_prior_prepare(weights)
    st["nr32"] = False
    try:
        if st.get("nr_on") and st.get("bfp") is not None and _n32_selftest(st, st["qkv"].device):
            st.setdefault("compose_link_enabled", {})[32] = True
            st.setdefault("mm_pick", {})[(32, "pr")] = "bfpf"
            st["nr32"] = True
        print(f"[nr32] no-handshake pooler/router at T=32: {'on' if st['nr32'] else 'off'}", flush=True)
    except BaseException as ex:
        st["nr32"] = False
        print(f"[nr32] off: {type(ex).__name__}: {str(ex)[-300:]}", flush=True)
    return st


_n32_prior_kernel_fn = kernel_fn


def kernel_fn(data, state):
    if state.get("nr32") and data["config"]["tokens"] == 32:
        _NR_PART.clear()
        try:
            with _nr_rebind(_gle_frt=_nr_gle_frt, bfp_fused_pooler_router=_nr_bfp_fused, _sl_tile_route=_NR_SLT, _gle_route=_NR_GLE):
                res = _n32_prior_kernel_fn(data, state)
            if _NR_PART:
                raise RuntimeError("norh32: unconsumed router partials")
        finally:
            _NR_PART.clear()
        return res
    return _n32_prior_kernel_fn(data, state)


# ---- final prepare wrapper (append LAST, after every other block): order stress of the data-flag (DF) MoE builds, then re-arm of the DF sentinel rows.
# Runs T = 32 -> 16 -> 256 -> 8 -> 16 -> 128 -> 16 on all-local routes (the most inter rows each T can write), each step compared bit for bit with the same input's output from a clean
# state (sentinel re-armed), and prints max|d| per step. Without the DF-row move the T=16 / T=8 steps read the rows the T=32 / 256 / 128 launches wrote.
# About 20 launches on already built extensions: no compile, a few ms. It never raises; the sentinel rows are always re-armed at the end (a b3r/tuner T=384 launch writes them).
import time as _dft_time

_DFT = {"bad": None, "worst": None}


def _dft_gen(T, dev):
    g = torch.Generator(device=dev).manual_seed(4100 + T)
    x = torch.randn((T, 4096), generator=g, device=dev).to(torch.bfloat16)
    top = torch.rand((T, 32), generator=g, device=dev).topk(8, dim=-1)
    return x, top.indices.to(torch.int32).contiguous(), (top.values / top.values.sum(-1, keepdim=True)).float().contiguous()


def _dft_run(st, T, x, lids, wts):
    o = torch.zeros((T, 4096), dtype=torch.bfloat16, device=x.device)
    _moe(T, st).okc_run(st["ma_w"], st["ma_scratch"], x, lids, wts, o, 0, 32, st["nsm"])
    torch.cuda.synchronize()
    return o


def _dft_order_stress(st, dev):
    seq = (32, 16, 256, 8, 16, 128, 16)
    inter = st["ma_scratch"][2]
    jt = st.get("ma_jt")
    ins = {T: _dft_gen(T, dev) for T in sorted(set(seq))}
    ref, noise = {}, {}
    for T in ins:
        outs = []
        for _ in range(2):
            _df_inter(inter)
            if jt is not None:
                jt.zero_()
            outs.append(_dft_run(st, T, *ins[T]))
        ref[T] = outs[0]
        noise[T] = float((outs[0].float() - outs[1].float()).abs().max())
    _df_inter(inter)
    if jt is not None:
        jt.zero_()
    bad, worst, line = 0, 0.0, []
    for T in seq:
        o = _dft_run(st, T, *ins[T])
        d = float((o.float() - ref[T].float()).abs().max())
        line.append(f"T={T}:{d:.3g}")
        worst = max(worst, d)
        bad += int(not d <= noise[T])
    print(f"[dft] order stress {' '.join(line)} | clean-rerun noise {max(noise.values()):.3g} | bad steps {bad}", flush=True)
    return bad, worst


_dft_prior_prepare = prepare


def prepare(weights):
    st = _dft_prior_prepare(weights)
    t0 = _dft_time.time()
    try:
        _DFT["bad"], _DFT["worst"] = _dft_order_stress(st, st["ma_scratch"][0].device)
        print(f"[dft] done in {_dft_time.time() - t0:.2f}s", flush=True)
    except BaseException as ex:   # noqa: BLE001
        print(f"[dft] order stress skipped: {type(ex).__name__}: {str(ex)[-200:]}", flush=True)
    finally:
        try:
            _df_inter(st["ma_scratch"][2])
            if st.get("ma_jt") is not None:
                st["ma_jt"].zero_()
            torch.cuda.synchronize()
        except BaseException as ex:   # noqa: BLE001
            print(f"[dft] re-arm failed: {type(ex).__name__}: {str(ex)[-200:]}", flush=True)
    return st
