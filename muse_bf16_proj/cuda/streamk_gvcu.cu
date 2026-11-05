
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

struct G { const uint4* W; const uint4* W2; const __nv_bfloat16* x; void* out; const uint32_t* esc; float4* ws; int* cnt; int m, N, K, KC; uint32_t T0, T1, U0, U1; };

__device__ __forceinline__ uint4 gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int XP> __device__ __forceinline__ uint4 gxp(const void* p);
__device__ __forceinline__ uint4 gx(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int XP> __device__ __forceinline__ uint4 gxp(const void* p) {
  uint4 v;
  if constexpr (XP == 1) asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else if constexpr (XP == 2) asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else v = gx(p);
  return v;
}
__device__ __forceinline__ uint4 lds4(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void hmma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// V & 1 == 0: nibble v = offset of value v (byte-permute table T0:T1 = base .. base + 7), signs at bits 3, 11, 19, 27 (values 0-3)
// and 7, 15, 23, 31 (values 4-7): 11 ops / 8 values
__device__ __forceinline__ void dec0(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hl = prmt(T0, T1, cm) | ((C << 4) & 0x80808080u), hh = prmt(T0, T1, cm >> 16) | (C & 0x80808080u);
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
// V & 1 == 1: C byte i: bits 0-2 offset of value i, bit 7 its sign, bits 4-6 offset of value 4 + i, bit 3 the sign of value 4 + (i + 3) % 4,
// so (C & 0x87878787) + base and the same on C rotated by 4 are the high bytes of values 0-3 and 4-7 (window base <= 120)
__device__ __forceinline__ void dec1(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void sto(__nv_bfloat16* d, float v) { *d = __float2bfloat16(v); }
__device__ __forceinline__ void sto(float* d, float v) { *d = v; }
template <typename O>
__device__ __forceinline__ void emit(const G& p, int idx, int row0, float4 v, int MT) {
  const int l = idx & 31, r = idx >> 5, n = r % MT, row = row0 + (r / MT) * 16 + (l >> 2), tok = n * 8 + 2 * (l & 3);
  O* o = (O*)p.out + (size_t)tok * p.N + row;
  sto(o, v.x); sto(o + 8, v.z);
  if (tok + 1 < p.m) { sto(o + p.N, v.y); sto(o + p.N + 8, v.w); }
}
// split-K partial added into an output the caller zeroed (a memset node in the same graph); fp32 (router) or bf16
__device__ __forceinline__ void reda(float* d, float v) { atomicAdd(d, v); }
__device__ __forceinline__ void reda2(__nv_bfloat16* d, float a, float b) {  // native packed bf16x2 reduction (no return value)
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  asm volatile("red.global.add.noftz.bf16x2 [%0], %1;\n" :: "l"(d), "r"(*(const uint32_t*)&h) : "memory");
}
template <typename O>
__device__ __forceinline__ void redemit(const G& p, int idx, int row0, float4 v, int MT) {
  const int l = idx & 31, r = idx >> 5, n = r % MT, row = row0 + (r / MT) * 16 + (l >> 2), tok = n * 8 + 2 * (l & 3);
  O* o = (O*)p.out + (size_t)tok * p.N + row;
  if constexpr (std::is_same<O, float>::value) {
    reda(o, v.x); reda(o + 8, v.z);
    if (tok + 1 < p.m) { reda(o + p.N, v.y); reda(o + p.N + 8, v.w); }
  } else {  // bf16: rows (g, g+1) of lanes l, l ^ 4 (same tokens, same liveness) as one native packed bf16x2 red.add
    const unsigned am = __activemask();
    const float px = __shfl_xor_sync(am, v.x, 4), py = __shfl_xor_sync(am, v.y, 4), pz = __shfl_xor_sync(am, v.z, 4), pw = __shfl_xor_sync(am, v.w, 4);
    if (((l >> 2) & 1) == 0) {  // even row g: the pair starts at an even column (4-B aligned: row0 % 16 == 0, N even)
      reda2(o, v.x, px); reda2(o + 8, v.z, pz);
      if (tok + 1 < p.m) { reda2(o + p.N, v.y, py); reda2(o + p.N + 8, v.w, pw); }
    }
  }
}
__device__ __forceinline__ void add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
__device__ __forceinline__ bool live(int idx, int MT, int m) { return ((idx >> 5) % MT) * 8 + 2 * (idx & 3) < m; }

struct CeS { uint32_t lo, hi, V, pb, iss, nw; };
__device__ __forceinline__ uint32_t ce_lds(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t ce_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ uint32_t ce_sel(uint32_t e, uint32_t d) {  // (e & ~0x807f807f) | (d & 0x807f807f): immLut b ? c : a = 0xb8
  uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xb8;" : "=r"(r) : "r"(e), "r"(0x807f807fu), "r"(d)); return r;
}
// smem address of the word at substream byte pb: rl = ring_s + lane * 4, ring_s aligned to ES * 128
template <int ES>
__device__ __forceinline__ uint32_t ce_addr(uint32_t rl, uint32_t pb) { return rl | ((pb << 5) & ((ES - 1) << 7)); }
// NI predicated 4 B cp.async; gsrc = stream + (soff[gw] + s * P * 32 + lane) * 4 (word 0 of this lane and substream)
template <int ES, int NI>
__device__ __forceinline__ void ce_issue(CeS& b, uint32_t P, const uint8_t* gsrc, uint32_t rl) {
#pragma unroll
  for (int k = 0; k < NI; k++) {
    const bool go = b.iss < P && b.iss < (b.pb >> 2) + ES;
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;\n @p cp.async.ca.shared.global [%0], [%1], 4; }\n"
                 :: "r"(rl | ((b.iss % ES) << 7)), "l"(gsrc + (size_t)b.iss * 128), "r"((uint32_t)go));
    b.iss += go;
  }
}
__device__ __forceinline__ void ce_refill(CeS& b, uint32_t w) {
  const bool need = (b.V & 0x60u) == 0u;
  w = need ? w : 0u;
  const uint32_t at = b.V & 31u;
  b.lo |= w << at; b.hi |= __funnelshift_l(w, 0u, at);
  b.V += need ? 32u : 0u; b.pb += need ? 4u : 0u;
}
// one unit: the K substreams' chains interleaved (independent), 16 A registers in r6 order A[4 j + i];
// rl = ring of substream 0 + lane * 4 (substream s at + s * ES * 128); LS = LUT index shift (2 single, 7 lane-replicated)
template <int ES, int LS, int K>
__device__ __forceinline__ void ce_unit(CeS* b, uint32_t lut, uint32_t rl, uint32_t t0, uint32_t t1, uint4 q0, uint4 q1, uint32_t* A) {
  constexpr int NS = 4 / K;
  const uint32_t L[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int t = 0; t < NS; t++) {
#pragma unroll
    for (int s = 0; s < K; s++) {
      const uint32_t rs = rl + s * ES * 128;
      ce_refill(b[s], t == 0 ? ce_lds(ce_addr<ES>(rs, b[s].pb)) : b[s].nw);
      if (t + 1 < NS) b[s].nw = ce_lds(ce_addr<ES>(rs, b[s].pb));
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int s = 0; s < K; s++) {
        const int j = s * NS + t;
        const uint32_t len = ce_prmt(t0, t1, b[s].lo & 7u);
        const uint32_t e = ce_lds(lut + ((b[s].lo & 0xffu) << LS));
        b[s].lo = __funnelshift_r(b[s].lo, b[s].hi, len); b[s].hi = __funnelshift_r(b[s].hi, 0u, len); b[s].V -= len;
        A[4 * j + i] = ce_sel(e, ce_prmt(L[2 * j + (i >> 1)], 0u, (i & 1) ? 0x3322u : 0x1100u));
      }
  }
}
// escapes (r6 walk, unchanged except the XOR field): A[r] ^= (nx & 0xff) << (7 + 16 * ((nx >> 10) & 1))
// one step t (4 A registers) of ce_unit<ES, LS, 1>, identical operations in identical order (refill, prefetch for
// step t + 1, 4 pairs), so a caller can decode and consume a unit step by step with 4 A registers live instead of 16
template <int ES, int LS>
__device__ __forceinline__ void ce_step1(CeS& b, uint32_t lut, uint32_t rl, uint32_t t0, uint32_t t1, uint32_t Lw0, uint32_t Lw1, int t, uint32_t* A) {
  ce_refill(b, t == 0 ? ce_lds(ce_addr<ES>(rl, b.pb)) : b.nw);
  if (t < 3) b.nw = ce_lds(ce_addr<ES>(rl, b.pb));
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint32_t len = ce_prmt(t0, t1, b.lo & 7u);
    const uint32_t e = ce_lds(lut + ((b.lo & 0xffu) << LS));
    b.lo = __funnelshift_r(b.lo, b.hi, len); b.hi = __funnelshift_r(b.hi, 0u, len); b.V -= len;
    A[i] = ce_sel(e, ce_prmt((i >> 1) ? Lw1 : Lw0, 0u, (i & 1) ? 0x3322u : 0x1100u));
  }
}

// per-warp private bulk-copy ring for W (V bit 3). V bits 4-5: slots NS = 2 << code; bits 6-7: chunks per slot
// CU = 1 << code; bit 8: refill without fence.proxy.async (the slot's values are consumed by the decode, in program
// order, before the __syncwarp and the refill; CUTLASS-style release). Lane 0 of each warp fills its own slots with cp.async.bulk (one copy per row tile and matrix, CU
// contiguous 64-k chunks) on a per-slot mbarrier; the warp waits on it, reads its fragments with ld.shared.v4 at the
// offsets r6's ld.global.v4 used, and lane 0 refills the slot after the warp has decoded its last chunk.
template <int FMT, int MT, int WR, int WK, int KU, int S, int RT, int XS, int V>
struct GL {
  static constexpr int NL = FMT == 4 ? 2 : ((FMT & 1) ? 3 : 4), NA = (FMT == 2 || FMT == 3) ? 2 : 1, NW = WR * WK, NI = WR * RT * MT * 32;
  static constexpr int CK = 1 << ((V >> 22) & 3), CED = ((V >> 19) & 1) ? 32 : 16, CES = CED / CK;
  static constexpr int CLUT = FMT == 4 ? (((V >> 20) & 1) ? 32768 : 1024) : 0;
  static constexpr int CAL = CES * 128 > 2048 ? CES * 128 : 2048;  // ring alignment (ring_s | slot offset)
  static constexpr int CRING = FMT == 4 ? NW * 32 * CED * 4 + CAL : 0;  // + CAL: alignment slack
  static constexpr bool BK = (V & 8) != 0;
  static constexpr int CU = BK ? 1 << ((V >> 6) & 3) : 1, G = KU / CU, NS0 = BK ? 2 << ((V >> 4) & 3) : 1;
  static constexpr int NS = NS0 < G ? NS0 : G, SB = RT * NA * CU * NL * 512;
  static constexpr int SLAB = XS == 1 ? WK * KU * MT * 8 * 128 : 0, REDB = (WK > 1 || S > 1) ? (WK + 1) * NI * 16 : 0;
  static constexpr int BAR = BK ? ((NW * NS * 8 + 127) / 128) * 128 : 0, RING = BK ? NW * NS * SB : 0;
  static constexpr int SMEM = SLAB + REDB + BAR + RING + CLUT + CRING;
};
__device__ __forceinline__ void gxwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nGXW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra GXW;\n}\n" :: "r"(b), "r"(ph) : "memory");
}

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, ((V >> 12) & 7) ? ((V >> 12) & 7) : (XS == 2 && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1)) gv(G p) {
  constexpr int LAD = (V >> 9) & 7;
  constexpr bool LX = LAD != 1, LDEC = LAD == 0 || LAD >= 3, LMMA = LAD == 0 || LAD == 4;
  uint32_t sk = 0;
  constexpr bool C12 = FMT & 1, HL = FMT == 2 || FMT == 3, CE = FMT == 4;
  constexpr int NL = CE ? 2 : (C12 ? 3 : 4), NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = (MT * RT * NA >= 2 || (CE && ((V >> 24) & 1))) ? 1 : 2, XB = XS ? 1 : ST;
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  constexpr bool XZ = ((V >> 29) & 1) != 0;  // grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow
  const int bz = blockIdx.x, bx = blockIdx.y, by = (int)blockIdx.z - (XZ ? 1 : 0), GX = gridDim.y;
  if constexpr (XZ) {
    if (by < 0) {  // dispatched first (lowest z): zero this tile of the output, publish, exit; no W, no x
      O* o = (O*)p.out + (size_t)bz * TK * p.N + bx * WR * RT * 16;
      const int R = WR * RT * 16, mm = min(TK, p.m - bz * TK);
      for (int i = threadIdx.x; i < mm * R; i += WR * WK * 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
      __threadfence(); __syncthreads();
      if (threadIdx.x == 0) atomicAdd(p.cnt + 2 * (bz * GX + bx) + 1, 1);
      return;
    }
  }
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  using L = GL<FMT, MT, WR, WK, KU, S, RT, XS, V>;
  static_assert(KU % L::CU == 0, "chunks per slot must divide KU");
  auto ld = [&](int s, int u) {
    if constexpr (!L::BK) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    }
    if (!XS && LX) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xb[s % XB][n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
  };
  uint32_t bbar = 0, bring = 0;
  auto bissue = [&](int slot, int grp) {  // lane 0 only: fill one slot with slot-group grp (CU chunks per row tile / matrix)
    const uint32_t bar = bbar + slot * 8;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(L::SB) : "memory");
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(bring + slot * L::SB + (j * NA + h) * L::CU * NL * 512),
                        "l"((h ? p.W2 : p.W) + (wo - lane) + j * js + (size_t)grp * L::CU * NL * 32), "n"(L::CU * NL * 512), "r"(bar) : "memory");
  };
  if constexpr (L::BK) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    bbar = s0 + w * L::NS * 8;
    bring = s0 + L::BAR + w * L::NS * L::SB;
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < L::NS; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bbar + k * 8));
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < L::NS; k++) bissue(k, k);
    }
    __syncwarp();
  }
  int tk = 0;
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // no memset node; the split-0 CTA of the tile zeroes it, then raises a flag
    if (w == 0) {
      const int tile = bz * GX + bx, R = WR * RT * 16;
      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);
      if (by == 0 && !XZ) {
        O* o = (O*)p.out + row0;
        for (int i = lane; i < p.m * R; i += 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(p.cnt + 2 * tile + 1, 1);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  CeS ceb[CE ? L::CK : 1];
  uint32_t cP = 0u, crl = 0u, clut = 0u, ct0 = 0u, ct1 = 0u;
  const uint8_t* cgs = nullptr;
  if constexpr (CE) {  // segment start, ring prologue (3 issue rounds, one commit group each), LUT + length table
#pragma unroll
    for (int s = 0; s < L::CK; s++) ceb[s] = CeS{0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t gw = (uint32_t)((by * GX + bx) * NW + w);
    const uint32_t* soff = (const uint32_t*)(((uint64_t)p.U1 << 32) | p.U0);
    const uint32_t s0 = __ldg(soff + gw);
    cP = (__ldg(soff + gw + 1) - s0) / (32u * L::CK);
    cgs = (const uint8_t*)p.W2 + ((size_t)s0 + lane) * 4;  // word 0 of this lane, substream 0 (substream s: + s * P * 128 B)
    const uint32_t cs00 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    const uint32_t cs0 = (cs00 + L::CAL - 1u) & ~(uint32_t)(L::CAL - 1);  // ring_s | slot offset needs ES * 128 alignment
    crl = cs0 + (w * 32 * L::CED + lane) * 4;
    clut = cs0 + NW * 32 * L::CED * 4 + (((V >> 20) & 1) ? lane * 4 : 0);
    const uint32_t* gl = (const uint32_t*)(((uint64_t)p.T1 << 32) | p.T0);
    ct0 = __ldg(gl + 256); ct1 = __ldg(gl + 257);
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    uint32_t* sl = (uint32_t*)(smem + L::SLAB + L::REDB + (cs0 - cs00) + NW * 32 * L::CED * 4);
    constexpr int NE = (256 + NT - 1) / NT;  // LUT entries per thread: all loads in flight at once, then the stores
    uint32_t lv[NE];
#pragma unroll
    for (int k = 0; k < NE; k++) lv[k] = threadIdx.x + k * NT < 256 ? __ldg(gl + threadIdx.x + k * NT) : 0u;
#pragma unroll
    for (int k = 0; k < NE; k++) {
      const int x = threadIdx.x + k * NT;
      if (x < 256) {
        if constexpr (((V >> 20) & 1) != 0) {  // entry x for lane l at word 32 x + l: 8 x 16 B per entry
#pragma unroll
          for (int q = 0; q < 8; q++) ((uint4*)(sl + x * 32))[q] = make_uint4(lv[k], lv[k], lv[k], lv[k]);
        } else sl[x] = lv[k];
      }
    }
    __syncthreads();
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  if ((C12 || CE) && LDEC) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }
  if (XS == 1) __syncthreads();
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](const int u, const int s) {
    if constexpr (CE && ((V >> 24) & 1)) {
      static_assert(RT == 1 && NA == 1 && XS != 1 && L::CK == 1, "lite path: RT 1, one substream, x not in a slab");
      ce_issue<L::CES, 4>(ceb[0], cP, cgs, crl);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      const uint4 q0 = wb[s][0][0][0], q1 = wb[s][0][0][1];
      uint32_t B[MT][8];
#pragma unroll
      for (int n = 0; n < MT; n++) {
        uint4 b0, b1;
        if (XS == 2) { b0 = gxp<(V >> 15) & 3>(xp[n] + u * 64); b1 = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
        else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
        B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
      }
      if (u + ST < KU) ld(s, u + ST);
      const uint32_t Lw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        uint32_t A4[4];
        ce_step1<L::CES, ((V >> 20) & 1) ? 7 : 2>(ceb[0], clut, crl, ct0, ct1, Lw[2 * jj], Lw[2 * jj + 1], jj, A4);
        uint32_t n3 = nx;  // this unit's escape run (warp-uniform), applied to step jj's registers
        int e3 = ep;
        while ((n3 >> 20) == (uint32_t)u) {
          if (lane == (int)((n3 >> 15) & 31) && (int)((n3 >> 13) & 3) == jj) {
            const uint32_t r = (n3 >> 11) & 3, xr = (n3 & 0xffu) << (7 + ((n3 >> 10) & 1) * 16);
#pragma unroll
            for (int i = 0; i < 4; i++) if (i == (int)r) A4[i] ^= xr;
          }
          n3 = ++e3 < 32 ? __shfl_sync(~0u, e, e3) : ~0u;
        }
        if (jj == 3) { nx = n3; ep = e3; }
#pragma unroll
        for (int n = 0; n < MT; n++) hmma(acc[0][n][jj % NC], A4, B[n][2 * jj], B[n][2 * jj + 1]);
      }
      return;
    }
    uint32_t A[RT][NA][16], B[MT][8];
    uint4 xd[XS == 2 ? MT : 1][2];
    if (XS == 2 && LX && !((V >> 25) & 1)) {  // the router token loop loads x at use
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) {
        uint4 qv[NL];
        if constexpr (L::BK) {
          const int grp = u / L::CU, ci = u % L::CU, slot = grp % L::NS;
          if (ci == 0 && j == 0 && h == 0) gxwait(bbar + slot * 8, (grp / L::NS) & 1);
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = lds4(bring + slot * L::SB + ((j * NA + h) * L::CU + ci) * NL * 512 + i * 512 + lane * 16);
        } else {
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = wb[s][j][h][i];
        }
        const uint4* q = qv;
        if constexpr (!LDEC) {  // ladder L0/L1: consume the W words with a cheap XOR, no decode
#pragma unroll
          for (int i = 0; i < NL; i++) sk ^= q[i].x ^ q[i].y ^ q[i].z ^ q[i].w;
        } else if constexpr (CE) {  // one issue round + commit per unit (u, j); the words this unit reads have landed
#pragma unroll
          for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
          asm volatile("cp.async.commit_group;\n" ::: "memory");
          asm volatile("cp.async.wait_group 2;\n" ::: "memory");
          ce_unit<L::CES, ((V >> 20) & 1) ? 7 : 2, L::CK>(ceb, clut, crl, ct0, ct1, q[0], q[1], A[j][h]);
        } else if (C12) {
          const uint32_t T0 = h ? p.U0 : p.T0, T1 = h ? p.U1 : p.T1;
          if (V & 1) {
            dec1(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec1(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec1(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec1(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          } else {
            dec0(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec0(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec0(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec0(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) { A[j][h][4 * i] = q[i].x; A[j][h][4 * i + 1] = q[i].y; A[j][h][4 * i + 2] = q[i].z; A[j][h][4 * i + 3] = q[i].w; }
        }
      }
    if constexpr (((V >> 25) & 1) != 0) {
      // multiplies every n8 token tile of the CTA in turn; tile n's x fragment is loaded at use (ld.global.nc, L1 shared
      // by the CTA's row-tile warps); one accumulator per (row tile, token tile)
      static_assert(XS == 2 && !L::BK, "token loop: x at use, register W ring");
      if (u + ST < KU) ld(s, u + ST);
      if ((C12 || CE) && LDEC) {
#pragma unroll
        for (int j = 0; j < RT; j++)
          while ((nx >> 20) == (uint32_t)(u * RT + j)) {
            if (lane == (int)((nx >> 15) & 31)) {
              const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
              for (int h = 0; h < NA; h++)
#pragma unroll
                for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
            }
            nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
          }
      }
#pragma unroll
      for (int n = 0; n < MT; n++) {
        const __nv_bfloat16* xq = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + (c0 + u) * 64 + t * 16;
        const uint4 b0 = gx(xq), b1 = gx(xq + 8);
        const uint32_t Bq[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
          for (int j = 0; j < RT; j++)
#pragma unroll
            for (int h = 0; h < NA; h++) hmma(acc[j][n][0], A[j][h] + 4 * q, Bq[2 * q], Bq[2 * q + 1]);
      }
      return;
    }
    if constexpr (L::BK) {  // the slot's last chunk is decoded into registers: refill it with the group NS ahead
      const int grp = u / L::CU;
      if (u % L::CU == L::CU - 1 && grp + L::NS < L::G) {
        __syncwarp();
        if (lane == 0) {
          if constexpr (!(V & 256)) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // V bit 8: no proxy fence
          bissue(grp % L::NS, grp + L::NS);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < MT; n++) {
      uint4 b0, b1;
      if (XS == 1) {
        const uint32_t a = sb + ((wk * KU + u) * TK + n * 8 + g) * 128;
        b0 = lds4(a + (((2 * t) ^ g) << 4)); b1 = lds4(a + (((2 * t + 1) ^ g) << 4));
      } else if (XS == 2) { b0 = xd[n % (XS == 2 ? MT : 1)][0]; b1 = xd[n % (XS == 2 ? MT : 1)][1]; }
      else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
      B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
    }
    if constexpr (LAD == 2 || LAD == 3) {  // ladder L1/L2: consume x (and the decoded A) instead of the mma
#pragma unroll
      for (int n = 0; n < MT; n++)
#pragma unroll
        for (int i = 0; i < 8; i++) sk ^= B[n][i];
      if constexpr (LAD == 3) {
#pragma unroll
        for (int j = 0; j < RT; j++)
#pragma unroll
          for (int h = 0; h < NA; h++)
#pragma unroll
            for (int i = 0; i < 16; i++) sk ^= A[j][h][i];
      }
    }
    if (u + ST < KU) ld(s, u + ST);
    if ((C12 || CE) && LDEC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
        while ((nx >> 20) == (uint32_t)(u * RT + j)) {
          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = CE ? (nx & 0xffu) << (7 + ((nx >> 10) & 1) * 16) : (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int h = 0; h < NA; h++)
#pragma unroll
              for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
          }
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
    }
    if constexpr (LMMA) {
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int h = 0; h < NA; h++) hmma(acc[j][n][(q * NA + h) % NC], A[j][h] + 4 * q, B[n][2 * q], B[n][2 * q + 1]);
    }
  };
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(u, u % ST);
  }
  if constexpr (LAD != 0) {  // ladder: keep the work alive with a store that never happens in practice, then exit
    if constexpr (LAD == 4) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int c = 0; c < NC; c++) sk ^= __float_as_uint(acc[j][n][c][0]) ^ __float_as_uint(acc[j][n][c][1]) ^ __float_as_uint(acc[j][n][c][2]) ^ __float_as_uint(acc[j][n][c][3]);
    }
    if (sk == (p.T0 ^ 0x9e3779b9u)) ((float*)p.ws)[threadIdx.x] = __uint_as_float(sk);
    return;
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // acquire-poll the tile's zero flag (H100 probe: 0.35 us under volatile + fence)
    if (threadIdx.x == 0) {
      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;
      unsigned spins_ = 0;  // bounded wait; a stuck zeroer aborts visibly instead of hanging to the timeout
      while (true) { int v_; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v_) : "l"(fl) : "memory"); if (v_ >= tgt) break; if (++spins_ > (1u << 24)) asm volatile("trap;"); }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (((V >> 28) & 1) != 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // PDL zero kernel done
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else if constexpr (((V >> 26) & 15) != 0) redemit<O>(p, idx, row0, v, MT);  // K split reduced by red.add, no cluster
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // partial to its ws slot
    }
  }
  if (S == 1) return;
  if constexpr (((V >> 26) & 15) != 0) return;
  if constexpr ((V & 4) != 0) {  // no cluster; the last-arriving CTA of the n-block sums the KS partials from L2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // release this thread's partial stores (lighter than __threadfence's fence.sc)
    __syncthreads();
    __shared__ int gxlast;
    if (threadIdx.x == 0) gxlast = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
    __syncthreads();
    if (!gxlast) return;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // acquire the other CTAs' partials
    const float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S; q++) if (q != by) add4(v, __ldcg(wsp + q * NI + idx));
      emit<O>(p, idx, row0, v, MT);
    }
    if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;  // self-cleaning for graph replay
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
  }
}

// Predicated format1 repair; source inputs are copied before any output write.
__device__ __forceinline__ void gv_fix_pred(uint32_t (&a)[16],uint32_t r,uint32_t xr,int lane,uint32_t target,uint32_t hm){
  asm volatile("{ .reg .b32 ri, mask; .reg .pred pl, pa, pr; mov.b32 ri, %16; mov.b32 mask, %17; setp.eq.u32 pl, %18, %19; setp.eq.and.u32 pa, %20, 0, pl; setp.eq.and.u32 pr, ri, 0, pa; @pr xor.b32 %0, %0, mask; setp.eq.and.u32 pr, ri, 1, pa; @pr xor.b32 %1, %1, mask; setp.eq.and.u32 pr, ri, 2, pa; @pr xor.b32 %2, %2, mask; setp.eq.and.u32 pr, ri, 3, pa; @pr xor.b32 %3, %3, mask; setp.eq.and.u32 pr, ri, 4, pa; @pr xor.b32 %4, %4, mask; setp.eq.and.u32 pr, ri, 5, pa; @pr xor.b32 %5, %5, mask; setp.eq.and.u32 pr, ri, 6, pa; @pr xor.b32 %6, %6, mask; setp.eq.and.u32 pr, ri, 7, pa; @pr xor.b32 %7, %7, mask; setp.eq.and.u32 pr, ri, 8, pa; @pr xor.b32 %8, %8, mask; setp.eq.and.u32 pr, ri, 9, pa; @pr xor.b32 %9, %9, mask; setp.eq.and.u32 pr, ri, 10, pa; @pr xor.b32 %10, %10, mask; setp.eq.and.u32 pr, ri, 11, pa; @pr xor.b32 %11, %11, mask; setp.eq.and.u32 pr, ri, 12, pa; @pr xor.b32 %12, %12, mask; setp.eq.and.u32 pr, ri, 13, pa; @pr xor.b32 %13, %13, mask; setp.eq.and.u32 pr, ri, 14, pa; @pr xor.b32 %14, %14, mask; setp.eq.and.u32 pr, ri, 15, pa; @pr xor.b32 %15, %15, mask; }" : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7]), "+r"(a[8]), "+r"(a[9]), "+r"(a[10]), "+r"(a[11]), "+r"(a[12]), "+r"(a[13]), "+r"(a[14]), "+r"(a[15]) : "r"(r),"r"(xr),"r"(lane),"r"(target),"r"(hm));
}

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, ((V >> 12) & 7) ? ((V >> 12) & 7) : (XS == 2 && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1)) gv_predxor(G p) {
  static_assert((FMT == 1 && MT == 1 && ST == 1 && WR == 2 && WK == 16 && KU == 8 && S == 1 && RT == 1 && XS == 0 && V == 3) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 1 && WK == 4 && KU == 16 && S == 1 && RT == 1 && XS == 2 && V == 1) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 1 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1), "GVpredxor only seven reviewed format1 templates");
  constexpr int LAD = (V >> 9) & 7;
  constexpr bool LX = LAD != 1, LDEC = LAD == 0 || LAD >= 3, LMMA = LAD == 0 || LAD == 4;
  uint32_t sk = 0;
  constexpr bool C12 = FMT & 1, HL = FMT == 2 || FMT == 3, CE = FMT == 4;
  constexpr int NL = CE ? 2 : (C12 ? 3 : 4), NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = (MT * RT * NA >= 2 || (CE && ((V >> 24) & 1))) ? 1 : 2, XB = XS ? 1 : ST;
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  constexpr bool XZ = ((V >> 29) & 1) != 0;  // grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow
  const int bz = blockIdx.x, bx = blockIdx.y, by = (int)blockIdx.z - (XZ ? 1 : 0), GX = gridDim.y;
  if constexpr (XZ) {
    if (by < 0) {  // dispatched first (lowest z): zero this tile of the output, publish, exit; no W, no x
      O* o = (O*)p.out + (size_t)bz * TK * p.N + bx * WR * RT * 16;
      const int R = WR * RT * 16, mm = min(TK, p.m - bz * TK);
      for (int i = threadIdx.x; i < mm * R; i += WR * WK * 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
      __threadfence(); __syncthreads();
      if (threadIdx.x == 0) atomicAdd(p.cnt + 2 * (bz * GX + bx) + 1, 1);
      return;
    }
  }
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  using L = GL<FMT, MT, WR, WK, KU, S, RT, XS, V>;
  static_assert(KU % L::CU == 0, "chunks per slot must divide KU");
  auto ld = [&](int s, int u) {
    if constexpr (!L::BK) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    }
    if (!XS && LX) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xb[s % XB][n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
  };
  uint32_t bbar = 0, bring = 0;
  auto bissue = [&](int slot, int grp) {  // lane 0 only: fill one slot with slot-group grp (CU chunks per row tile / matrix)
    const uint32_t bar = bbar + slot * 8;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(L::SB) : "memory");
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(bring + slot * L::SB + (j * NA + h) * L::CU * NL * 512),
                        "l"((h ? p.W2 : p.W) + (wo - lane) + j * js + (size_t)grp * L::CU * NL * 32), "n"(L::CU * NL * 512), "r"(bar) : "memory");
  };
  if constexpr (L::BK) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    bbar = s0 + w * L::NS * 8;
    bring = s0 + L::BAR + w * L::NS * L::SB;
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < L::NS; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bbar + k * 8));
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < L::NS; k++) bissue(k, k);
    }
    __syncwarp();
  }
  int tk = 0;
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // no memset node; the split-0 CTA of the tile zeroes it, then raises a flag
    if (w == 0) {
      const int tile = bz * GX + bx, R = WR * RT * 16;
      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);
      if (by == 0 && !XZ) {
        O* o = (O*)p.out + row0;
        for (int i = lane; i < p.m * R; i += 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(p.cnt + 2 * tile + 1, 1);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  CeS ceb[CE ? L::CK : 1];
  uint32_t cP = 0u, crl = 0u, clut = 0u, ct0 = 0u, ct1 = 0u;
  const uint8_t* cgs = nullptr;
  if constexpr (CE) {  // segment start, ring prologue (3 issue rounds, one commit group each), LUT + length table
#pragma unroll
    for (int s = 0; s < L::CK; s++) ceb[s] = CeS{0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t gw = (uint32_t)((by * GX + bx) * NW + w);
    const uint32_t* soff = (const uint32_t*)(((uint64_t)p.U1 << 32) | p.U0);
    const uint32_t s0 = __ldg(soff + gw);
    cP = (__ldg(soff + gw + 1) - s0) / (32u * L::CK);
    cgs = (const uint8_t*)p.W2 + ((size_t)s0 + lane) * 4;  // word 0 of this lane, substream 0 (substream s: + s * P * 128 B)
    const uint32_t cs00 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    const uint32_t cs0 = (cs00 + L::CAL - 1u) & ~(uint32_t)(L::CAL - 1);  // ring_s | slot offset needs ES * 128 alignment
    crl = cs0 + (w * 32 * L::CED + lane) * 4;
    clut = cs0 + NW * 32 * L::CED * 4 + (((V >> 20) & 1) ? lane * 4 : 0);
    const uint32_t* gl = (const uint32_t*)(((uint64_t)p.T1 << 32) | p.T0);
    ct0 = __ldg(gl + 256); ct1 = __ldg(gl + 257);
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    uint32_t* sl = (uint32_t*)(smem + L::SLAB + L::REDB + (cs0 - cs00) + NW * 32 * L::CED * 4);
    constexpr int NE = (256 + NT - 1) / NT;  // LUT entries per thread: all loads in flight at once, then the stores
    uint32_t lv[NE];
#pragma unroll
    for (int k = 0; k < NE; k++) lv[k] = threadIdx.x + k * NT < 256 ? __ldg(gl + threadIdx.x + k * NT) : 0u;
#pragma unroll
    for (int k = 0; k < NE; k++) {
      const int x = threadIdx.x + k * NT;
      if (x < 256) {
        if constexpr (((V >> 20) & 1) != 0) {  // entry x for lane l at word 32 x + l: 8 x 16 B per entry
#pragma unroll
          for (int q = 0; q < 8; q++) ((uint4*)(sl + x * 32))[q] = make_uint4(lv[k], lv[k], lv[k], lv[k]);
        } else sl[x] = lv[k];
      }
    }
    __syncthreads();
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  if ((C12 || CE) && LDEC) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }
  if (XS == 1) __syncthreads();
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](const int u, const int s) {
    if constexpr (CE && ((V >> 24) & 1)) {
      static_assert(RT == 1 && NA == 1 && XS != 1 && L::CK == 1, "lite path: RT 1, one substream, x not in a slab");
      ce_issue<L::CES, 4>(ceb[0], cP, cgs, crl);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      const uint4 q0 = wb[s][0][0][0], q1 = wb[s][0][0][1];
      uint32_t B[MT][8];
#pragma unroll
      for (int n = 0; n < MT; n++) {
        uint4 b0, b1;
        if (XS == 2) { b0 = gxp<(V >> 15) & 3>(xp[n] + u * 64); b1 = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
        else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
        B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
      }
      if (u + ST < KU) ld(s, u + ST);
      const uint32_t Lw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        uint32_t A4[4];
        ce_step1<L::CES, ((V >> 20) & 1) ? 7 : 2>(ceb[0], clut, crl, ct0, ct1, Lw[2 * jj], Lw[2 * jj + 1], jj, A4);
        uint32_t n3 = nx;  // this unit's escape run (warp-uniform), applied to step jj's registers
        int e3 = ep;
        while ((n3 >> 20) == (uint32_t)u) {
          if (lane == (int)((n3 >> 15) & 31) && (int)((n3 >> 13) & 3) == jj) {
            const uint32_t r = (n3 >> 11) & 3, xr = (n3 & 0xffu) << (7 + ((n3 >> 10) & 1) * 16);
#pragma unroll
            for (int i = 0; i < 4; i++) if (i == (int)r) A4[i] ^= xr;
          }
          n3 = ++e3 < 32 ? __shfl_sync(~0u, e, e3) : ~0u;
        }
        if (jj == 3) { nx = n3; ep = e3; }
#pragma unroll
        for (int n = 0; n < MT; n++) hmma(acc[0][n][jj % NC], A4, B[n][2 * jj], B[n][2 * jj + 1]);
      }
      return;
    }
    uint32_t A[RT][NA][16], B[MT][8];
    uint4 xd[XS == 2 ? MT : 1][2];
    if (XS == 2 && LX && !((V >> 25) & 1)) {  // the router token loop loads x at use
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) {
        uint4 qv[NL];
        if constexpr (L::BK) {
          const int grp = u / L::CU, ci = u % L::CU, slot = grp % L::NS;
          if (ci == 0 && j == 0 && h == 0) gxwait(bbar + slot * 8, (grp / L::NS) & 1);
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = lds4(bring + slot * L::SB + ((j * NA + h) * L::CU + ci) * NL * 512 + i * 512 + lane * 16);
        } else {
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = wb[s][j][h][i];
        }
        const uint4* q = qv;
        if constexpr (!LDEC) {  // ladder L0/L1: consume the W words with a cheap XOR, no decode
#pragma unroll
          for (int i = 0; i < NL; i++) sk ^= q[i].x ^ q[i].y ^ q[i].z ^ q[i].w;
        } else if constexpr (CE) {  // one issue round + commit per unit (u, j); the words this unit reads have landed
#pragma unroll
          for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
          asm volatile("cp.async.commit_group;\n" ::: "memory");
          asm volatile("cp.async.wait_group 2;\n" ::: "memory");
          ce_unit<L::CES, ((V >> 20) & 1) ? 7 : 2, L::CK>(ceb, clut, crl, ct0, ct1, q[0], q[1], A[j][h]);
        } else if (C12) {
          const uint32_t T0 = h ? p.U0 : p.T0, T1 = h ? p.U1 : p.T1;
          if (V & 1) {
            dec1(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec1(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec1(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec1(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          } else {
            dec0(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec0(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec0(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec0(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) { A[j][h][4 * i] = q[i].x; A[j][h][4 * i + 1] = q[i].y; A[j][h][4 * i + 2] = q[i].z; A[j][h][4 * i + 3] = q[i].w; }
        }
      }
    if constexpr (((V >> 25) & 1) != 0) {
      // multiplies every n8 token tile of the CTA in turn; tile n's x fragment is loaded at use (ld.global.nc, L1 shared
      // by the CTA's row-tile warps); one accumulator per (row tile, token tile)
      static_assert(XS == 2 && !L::BK, "token loop: x at use, register W ring");
      if (u + ST < KU) ld(s, u + ST);
      if ((C12 || CE) && LDEC) {
#pragma unroll
        for (int j = 0; j < RT; j++)
          while ((nx >> 20) == (uint32_t)(u * RT + j)) {
            if (lane == (int)((nx >> 15) & 31)) {
              const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
              for (int h = 0; h < NA; h++)
#pragma unroll
                for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
            }
            nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
          }
      }
#pragma unroll
      for (int n = 0; n < MT; n++) {
        const __nv_bfloat16* xq = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + (c0 + u) * 64 + t * 16;
        const uint4 b0 = gx(xq), b1 = gx(xq + 8);
        const uint32_t Bq[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
          for (int j = 0; j < RT; j++)
#pragma unroll
            for (int h = 0; h < NA; h++) hmma(acc[j][n][0], A[j][h] + 4 * q, Bq[2 * q], Bq[2 * q + 1]);
      }
      return;
    }
    if constexpr (L::BK) {  // the slot's last chunk is decoded into registers: refill it with the group NS ahead
      const int grp = u / L::CU;
      if (u % L::CU == L::CU - 1 && grp + L::NS < L::G) {
        __syncwarp();
        if (lane == 0) {
          if constexpr (!(V & 256)) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // V bit 8: no proxy fence
          bissue(grp % L::NS, grp + L::NS);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < MT; n++) {
      uint4 b0, b1;
      if (XS == 1) {
        const uint32_t a = sb + ((wk * KU + u) * TK + n * 8 + g) * 128;
        b0 = lds4(a + (((2 * t) ^ g) << 4)); b1 = lds4(a + (((2 * t + 1) ^ g) << 4));
      } else if (XS == 2) { b0 = xd[n % (XS == 2 ? MT : 1)][0]; b1 = xd[n % (XS == 2 ? MT : 1)][1]; }
      else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
      B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
    }
    if constexpr (LAD == 2 || LAD == 3) {  // ladder L1/L2: consume x (and the decoded A) instead of the mma
#pragma unroll
      for (int n = 0; n < MT; n++)
#pragma unroll
        for (int i = 0; i < 8; i++) sk ^= B[n][i];
      if constexpr (LAD == 3) {
#pragma unroll
        for (int j = 0; j < RT; j++)
#pragma unroll
          for (int h = 0; h < NA; h++)
#pragma unroll
            for (int i = 0; i < 16; i++) sk ^= A[j][h][i];
      }
    }
    if (u + ST < KU) ld(s, u + ST);
    if ((C12 || CE) && LDEC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
        while ((nx >> 20) == (uint32_t)(u * RT + j)) {
          const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
          gv_fix_pred(A[j][0],r,xr,lane,(nx>>15)&31,hm);
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
    }
    if constexpr (LMMA) {
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int h = 0; h < NA; h++) hmma(acc[j][n][(q * NA + h) % NC], A[j][h] + 4 * q, B[n][2 * q], B[n][2 * q + 1]);
    }
  };
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(u, u % ST);
  }
  if constexpr (LAD != 0) {  // ladder: keep the work alive with a store that never happens in practice, then exit
    if constexpr (LAD == 4) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int c = 0; c < NC; c++) sk ^= __float_as_uint(acc[j][n][c][0]) ^ __float_as_uint(acc[j][n][c][1]) ^ __float_as_uint(acc[j][n][c][2]) ^ __float_as_uint(acc[j][n][c][3]);
    }
    if (sk == (p.T0 ^ 0x9e3779b9u)) ((float*)p.ws)[threadIdx.x] = __uint_as_float(sk);
    return;
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // acquire-poll the tile's zero flag (H100 probe: 0.35 us under volatile + fence)
    if (threadIdx.x == 0) {
      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;
      unsigned spins_ = 0;  // bounded wait; a stuck zeroer aborts visibly instead of hanging to the timeout
      while (true) { int v_; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v_) : "l"(fl) : "memory"); if (v_ >= tgt) break; if (++spins_ > (1u << 24)) asm volatile("trap;"); }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (((V >> 28) & 1) != 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // PDL zero kernel done
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else if constexpr (((V >> 26) & 15) != 0) redemit<O>(p, idx, row0, v, MT);  // K split reduced by red.add, no cluster
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // partial to its ws slot
    }
  }
  if (S == 1) return;
  if constexpr (((V >> 26) & 15) != 0) return;
  if constexpr ((V & 4) != 0) {  // no cluster; the last-arriving CTA of the n-block sums the KS partials from L2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // release this thread's partial stores (lighter than __threadfence's fence.sc)
    __syncthreads();
    __shared__ int gxlast;
    if (threadIdx.x == 0) gxlast = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
    __syncthreads();
    if (!gxlast) return;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // acquire the other CTAs' partials
    const float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S; q++) if (q != by) add4(v, __ldcg(wsp + q * NI + idx));
      emit<O>(p, idx, row0, v, MT);
    }
    if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;  // self-cleaning for graph replay
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
  }
}


__global__ void p1zero(uint4* o, int n16) {  // the PDL primary, lets the gv grid launch at once
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n");
#endif
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) o[i] = make_uint4(0u, 0u, 0u, 0u);
}
template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
static int64_t gvl(const G& p, cudaStream_t st, int occ) {
  constexpr int NI = WR * RT * MT * 32;
  auto k = gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  if constexpr ((FMT == 1 && MT == 1 && ST == 1 && WR == 2 && WK == 16 && KU == 8 && S == 1 && RT == 1 && XS == 0 && V == 3) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 1 && WK == 4 && KU == 16 && S == 1 && RT == 1 && XS == 2 && V == 1) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 1 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1)) {
    const bool pooler = p.N == 4096 && p.K == 8192 && ((MT == 1 && (p.m == 1 || p.m == 8)) || (MT == 2 && p.m == 16));
    const bool unpooler = MT == 2 && p.N == 8192 && p.K == 4096 && p.m == 16;
    if (pooler || unpooler) k = gv_predxor<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  }
  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;
  const size_t sm = (XS == 1 ? (size_t)WK * KU * MT * 8 * 128 : 0) + ((WK > 1 || S > 1) ? (size_t)(WK + 1) * NI * 16 : 0)
                  + (size_t)(GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::BAR + GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::RING)
                  + (size_t)(GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::CLUT + GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::CRING);
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if ((V >> 17) & 1) { cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 0); cudaGetLastError(); }
    if (S > 8) { cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); cudaGetLastError(); }
    once = true;
  }
  static int cc = 0;
  if (!cc) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev); }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((p.m + 8 * MT - 1) / (8 * MT), p.N / (16 * WR * RT), S + (((V >> 29) & 1) ? 1 : 0)); cfg.blockDim = dim3(WR * WK * 32); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  const bool clu = S > 1 && cc >= 9 && !(V & 4) && !((V >> 26) & 15);  // V bit 2 = global last-arriver reduction, no cluster
  if (occ == 2) { cudaFuncAttributes fa; if (cudaFuncGetAttributes(&fa, k) != cudaSuccess) { cudaGetLastError(); return -1; } return fa.numRegs; }
  if (occ) {  // how many CTAs of this launch can be co-resident (clusters: cudaOccupancyMaxActiveClusters)
    int n = 0;
    if (clu) {
      at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1;
      if (cudaOccupancyMaxActiveClusters(&n, k, &cfg) != cudaSuccess) { cudaGetLastError(); return -1; }
      return (int64_t)n * S;
    }
    int dev, sms; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, WR * WK * 32, sm) != cudaSuccess) { cudaGetLastError(); return -1; }
    return (int64_t)n * sms;
  }
  if (clu) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1; }
  if constexpr (((V >> 28) & 1) != 0) {
    const int n16 = (int)(((size_t)p.m * p.N * sizeof(typename std::conditional<FMT == 2 || FMT == 3, float, __nv_bfloat16>::type)) / 16);
    p1zero<<<(n16 + 255) / 256 < 8 ? (n16 + 255) / 256 : 8, 256, 0, st>>>((uint4*)p.out, n16);
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1; cfg.attrs = at; cfg.numAttrs = 1;
  }
  const cudaError_t r = cudaLaunchKernelEx(&cfg, k, p);
  if (r != cudaSuccess) cudaGetLastError();
  return (int64_t)r;
}

// FP32 router, small m: CTA = R full rows (no cross-CTA reduction), 512 threads split K, fp32 weights read once
template <int M, int R, int NT = 512>
__global__ void __launch_bounds__(NT) rr(const float4* W, const __nv_bfloat16* x, float* out, int m, int N, int K) {
  __shared__ float red[NT / 32][R * M];
  const int t = threadIdx.x, lane = t & 31, w = t >> 5, K4 = K >> 2, row0 = blockIdx.x * R;
  constexpr int NI = 2048 / NT;  // float4 per thread per row (K = 8192)
  float4 wv[R][NI];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int i = 0; i < NI; i++) { const uint4 u = gnc(W + (size_t)(row0 + r) * K4 + t + i * NT); wv[r][i] = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) acc[r][q] = 0.f;
#pragma unroll
  for (int i = 0; i < NI; i++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      if (q < m) {
        const uint2 xv = __ldg((const uint2*)(x + (size_t)q * K) + t + i * NT);
        const float2 a = __bfloat1622float2(*(const __nv_bfloat162*)&xv.x), b = __bfloat1622float2(*(const __nv_bfloat162*)&xv.y);
#pragma unroll
        for (int r = 0; r < R; r++) acc[r][q] = fmaf(wv[r][i].w, b.y, fmaf(wv[r][i].z, b.x, fmaf(wv[r][i].y, a.y, fmaf(wv[r][i].x, a.x, acc[r][q]))));
      }
    }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      float v = acc[r][q];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0) red[w][r * M + q] = v;
    }
  __syncthreads();
  if (t < R * M && t % M < m) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 32; i++) v += red[i][t];
    out[(size_t)(t % M) * N + row0 + t / M] = v;
  }
}

__device__ __forceinline__ void rs_cpa(uint32_t dst, const void* src, uint32_t sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void rs_ldm4(uint32_t a, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void rs_red2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" :: "l"(a), "f"(x), "f"(y) : "memory");
}
template <int BN, int KR, int RED>
__global__ void __launch_bounds__(256, 1) rsk(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out,
                                            float4* ws, int* cnt, int m, int N, int K) {
  constexpr int TOK = 128, CH = KR / 8, WN = BN / 2, NT8 = WN / 8, RB = KR * 2;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sx = (uint32_t)__cvta_generic_to_shared(smem), sh = sx + TOK * RB, sl = sh + BN * RB;
  const int nt = blockIdx.x, ks = blockIdx.y, tb = blockIdx.z, NTL = gridDim.x, KS = gridDim.y;
  const int n0 = nt * BN, k0 = ks * KR, t0 = tb * TOK;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < TOK * CH; i += 256) {
    const int r = i / CH, c = i - r * CH, tok = t0 + r;
    rs_cpa(sx + r * RB + ((c ^ (r & 7)) << 4), x + (size_t)min(tok, m - 1) * K + k0 + c * 8, tok < m ? 16u : 0u);
  }
  for (int i = tid; i < BN * CH; i += 256) {
    const int r = i / CH, c = i - r * CH;
    const size_t o = (size_t)(n0 + r) * K + k0 + c * 8;
    const uint32_t d = r * RB + ((c ^ (r & 7)) << 4);
    rs_cpa(sh + d, hi + o, 16u); rs_cpa(sl + d, lo + o, 16u);
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  const int wt = w & 3, wn = w >> 2;
  float acc[2][NT8][4];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < NT8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const int ar = lane & 15, ac = lane >> 4;                 // A: row (lane & 15) of the m16 tile, k chunk + (lane >> 4)
  const int bj = lane >> 3, br = (bj >> 1) * 8 + (lane & 7), bc = bj & 1;  // B: n row of the n16 pair, k chunk + (j & 1)
#pragma unroll 4
  for (int q = 0; q < KR / 16; q++) {
    uint32_t a[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
      const int r = wt * 32 + mi * 16 + ar, c = 2 * q + ac;
      rs_ldm4(sx + r * RB + ((c ^ (r & 7)) << 4), a[mi]);
    }
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int nb = 0; nb < NT8 / 2; nb++) {
        const int r = wn * WN + nb * 16 + br, c = 2 * q + bc;
        uint32_t b[4];
        rs_ldm4((h ? sl : sh) + r * RB + ((c ^ (r & 7)) << 4), b);
#pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[mi][2 * nb][0]), "+f"(acc[mi][2 * nb][1]), "+f"(acc[mi][2 * nb][2]), "+f"(acc[mi][2 * nb][3])
            : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[0]), "r"(b[1]));
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[mi][2 * nb + 1][0]), "+f"(acc[mi][2 * nb + 1][1]), "+f"(acc[mi][2 * nb + 1][2]), "+f"(acc[mi][2 * nb + 1][3])
            : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[2]), "r"(b[3]));
        }
      }
  }
  if constexpr (RED == 0) {
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
      for (int j = 0; j < NT8; j++) {
        const int tok = t0 + wt * 32 + mi * 16 + g, n = n0 + wn * WN + j * 8 + 2 * t;
        if (tok < m) rs_red2(out + (size_t)tok * N + n, acc[mi][j][0], acc[mi][j][1]);
        if (tok + 8 < m) rs_red2(out + (size_t)(tok + 8) * N + n, acc[mi][j][2], acc[mi][j][3]);
      }
  } else {
    constexpr int NF = 2 * NT8 * 32 * 8;  // float4 fragments per CTA partial (8 warps)
    float4* slot = ws + ((size_t)(tb * NTL + nt) * KS + ks) * NF;
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
      for (int j = 0; j < NT8; j++) __stcg(slot + ((w * 2 + mi) * NT8 + j) * 32 + lane, make_float4(acc[mi][j][0], acc[mi][j][1], acc[mi][j][2], acc[mi][j][3]));
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (tid == 0) last = atomicAdd(cnt + tb * NTL + nt, 1) == KS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float4* base = ws + (size_t)(tb * NTL + nt) * KS * NF;
    for (int i = tid; i < NF; i += 256) {
      float4 v = __ldcg(base + i);
      for (int k2 = 1; k2 < KS; k2++) { const float4 u = __ldcg(base + (size_t)k2 * NF + i); v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
      const int l = i & 31, f = i >> 5, j = f % NT8, mi = (f / NT8) & 1, ww = f / NT8 / 2;
      const int tok = t0 + (ww & 3) * 32 + mi * 16 + (l >> 2), n = n0 + (ww >> 2) * WN + j * 8 + 2 * (l & 3);
      if (tok < m) { out[(size_t)tok * N + n] = v.x; out[(size_t)tok * N + n + 1] = v.y; }
      if (tok + 8 < m) { out[(size_t)(tok + 8) * N + n] = v.z; out[(size_t)(tok + 8) * N + n + 1] = v.w; }
    }
    if (tid == 0) cnt[tb * NTL + nt] = 0;  // self-cleaning for graph replay
  }
}
template <int BN, int KR, int NSTG, int TOKS, int RED, int MINB>
__global__ void __launch_bounds__(256, MINB) rsk2(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out,
                                                 float* ws, int* cnt, int m, int N, int K) {
  constexpr int CH = KR / 8, CS = CH / NSTG, WN = BN / 2, NT8 = WN / 8, RB = KR * 2, MI = TOKS / 64, QS = KR / 16 / NSTG;
  static_assert(CH % NSTG == 0 && NT8 % 2 == 0, "stages / n tiles");
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sx = (uint32_t)__cvta_generic_to_shared(smem), sh = sx + TOKS * RB, sl = sh + BN * RB;
  const int nt = blockIdx.x, ks = blockIdx.y, tb = blockIdx.z, NTL = gridDim.x, KS = gridDim.y;
  const int n0 = nt * BN, k0 = ks * KR, t0 = tb * TOKS;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int s = 0; s < NSTG; s++) {  // stage s = 16-B chunks [s CS, (s + 1) CS) of every row
    for (int i = tid; i < TOKS * CS; i += 256) {
      const int r = i / CS, c = s * CS + i - (i / CS) * CS, tok = t0 + r;
      rs_cpa(sx + r * RB + ((c ^ (r & 7)) << 4), x + (size_t)min(tok, m - 1) * K + k0 + c * 8, tok < m ? 16u : 0u);
    }
    for (int i = tid; i < BN * CS; i += 256) {
      const int r = i / CS, c = s * CS + i - (i / CS) * CS;
      const size_t o = (size_t)(n0 + r) * K + k0 + c * 8;
      const uint32_t d = r * RB + ((c ^ (r & 7)) << 4);
      rs_cpa(sh + d, hi + o, 16u); rs_cpa(sl + d, lo + o, 16u);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  const int wt = w & 3, wn = w >> 2;
  float acc[MI][NT8][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NT8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const int ar = lane & 15, ac = lane >> 4;
  const int bj = lane >> 3, br = (bj >> 1) * 8 + (lane & 7), bc = bj & 1;
#pragma unroll
  for (int s = 0; s < NSTG; s++) {
    if (s == 0) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG - 1) : "memory");
    else if (s == 1) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG > 1 ? NSTG - 2 : 0) : "memory");
    else if (s == 2) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG > 2 ? NSTG - 3 : 0) : "memory");
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
#pragma unroll 2
    for (int qq = 0; qq < QS; qq++) {
      const int q = s * QS + qq;
      uint32_t a[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {  // warp wt owns tokens 32 wt .. of each 128-token block: mi = (block, m16)
        const int r = (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + ar, c = 2 * q + ac;
        rs_ldm4(sx + r * RB + ((c ^ (r & 7)) << 4), a[mi]);
      }
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int nb = 0; nb < NT8 / 2; nb++) {
          const int r = wn * WN + nb * 16 + br, c = 2 * q + bc;
          uint32_t b[4];
          rs_ldm4((h ? sl : sh) + r * RB + ((c ^ (r & 7)) << 4), b);
#pragma unroll
          for (int mi = 0; mi < MI; mi++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[mi][2 * nb][0]), "+f"(acc[mi][2 * nb][1]), "+f"(acc[mi][2 * nb][2]), "+f"(acc[mi][2 * nb][3])
              : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[0]), "r"(b[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[mi][2 * nb + 1][0]), "+f"(acc[mi][2 * nb + 1][1]), "+f"(acc[mi][2 * nb + 1][2]), "+f"(acc[mi][2 * nb + 1][3])
              : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[2]), "r"(b[3]));
          }
        }
    }
  }
  float* dst = RED == 0 ? out : ws;
#pragma unroll
  for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int j = 0; j < NT8; j++) {
      const int tok = t0 + (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + g, n = n0 + wn * WN + j * 8 + 2 * t;
      if (tok < m) rs_red2(dst + (size_t)tok * N + n, acc[mi][j][0], acc[mi][j][1]);
      if (tok + 8 < m) rs_red2(dst + (size_t)(tok + 8) * N + n, acc[mi][j][2], acc[mi][j][3]);
    }
  if constexpr (RED == 2) {  // the CTA completing this (token block, n tile) copies it to out and re-zeroes it
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (tid == 0) last = atomicAdd(cnt + tb * NTL + nt, 1) == KS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int rows = min(TOKS, m - t0);
    for (int i = tid; i < rows * (BN / 4); i += 256) {
      const int r = i / (BN / 4), c4 = i - r * (BN / 4);
      float4* src = (float4*)(ws + (size_t)(t0 + r) * N + n0) + c4;
      const float4 v = __ldcg(src);
      __stcg(src, make_float4(0.f, 0.f, 0.f, 0.f));
      ((float4*)(out + (size_t)(t0 + r) * N + n0))[c4] = v;
    }
    if (tid == 0) cnt[tb * NTL + nt] = 0;
  }
}
template <int BN, int KR, int NSTG, int TOKS, int RED, int MINB>
static int rs2_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float* ws, long long wsn, int* cnt, int cntn,
                  int m, int N, int K, cudaStream_t st) {
  constexpr int SM = (TOKS + 2 * BN) * KR * 2;
  if (N % BN || K % KR || m < 1) return -4;
  const int TB = (m + TOKS - 1) / TOKS, KS = K / KR, NTL = N / BN;
  if (RED == 2 && ((long long)TB * TOKS * N > wsn || TB * NTL > cntn)) return -5;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(rsk2<BN, KR, NSTG, TOKS, RED, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  rsk2<BN, KR, NSTG, TOKS, RED, MINB><<<dim3(NTL, KS, TB), 256, SM, st>>>(x, hi, lo, out, ws, cnt, m, N, K);
  return (int)cudaGetLastError();
}
template <int BN, int KR, int RED>
static int rs_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float4* ws, long long wsn, int* cnt, int cntn,
                 int m, int N, int K, cudaStream_t st) {
  constexpr int SM = (128 + 2 * BN) * KR * 2;
  if (N % BN || K % KR || m < 1) return -4;
  const int TB = (m + 127) / 128, KS = K / KR, NTL = N / BN;
  if (RED == 1 && ((long long)TB * NTL * KS * (BN / 16) * 2 * 32 * 8 > wsn || TB * NTL > cntn)) return -5;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(rsk<BN, KR, RED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  rsk<BN, KR, RED><<<dim3(NTL, KS, TB), 256, SM, st>>>(x, hi, lo, out, ws, cnt, m, N, K);
  return (int)cudaGetLastError();
}
int rs_launch(int v, const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float4* ws, long long wsn, int* cnt, int cntn,
              int m, int N, int K, cudaStream_t st) {
  switch (v) {
    case 0: return rs_go<64, 256, 0>(x, hi, lo, out, ws, wsn, cnt, cntn, m, N, K, st);
    case 5: return rs2_go<64, 256, 4, 128, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 6: return rs2_go<64, 256, 2, 128, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 7: return rs2_go<64, 128, 2, 128, 0, 2>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 8: return rs2_go<64, 256, 4, 128, 2, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 9: return rs2_go<64, 256, 4, 256, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 10: return rs2_go<64, 128, 2, 256, 0, 2>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 11: return rs2_go<64, 256, 4, 256, 2, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
  }
  return -2;
}
template <int NL, int RED>
__global__ void __launch_bounds__(256) pr_ldg(const uint4* buf, long long n16, float* out, uint32_t* sink) {
  const long long T = (long long)gridDim.x * 256, t = (long long)blockIdx.x * 256 + threadIdx.x;
  uint4 v[NL];
#pragma unroll
  for (int i = 0; i < NL; i++) { const long long j = t + (long long)i * T; v[i] = j < n16 ? gnc(buf + j) : make_uint4(0u, 0u, 0u, 0u); }
  uint32_t a = 0;
#pragma unroll
  for (int i = 0; i < NL; i++) a ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
  if constexpr (RED == 1) {
    __shared__ uint32_t sa[256];
    sa[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x < 64) atomicAdd(out + (blockIdx.x * 64 + threadIdx.x) % 2048, (float)((sa[threadIdx.x] ^ sa[threadIdx.x + 64]) & 1u));
  } else if constexpr (RED == 2) {
    __shared__ float red[64];
    __shared__ uint32_t sa[256];
    sa[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x < 64) red[threadIdx.x] = (float)((sa[threadIdx.x] ^ sa[threadIdx.x + 64]) & 1u);
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
    if (rk == 0 && threadIdx.x < 64) {
      float s = 0.f;
      const uint32_t la = (uint32_t)__cvta_generic_to_shared(red + threadIdx.x);
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t ra; float f;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(la), "r"(q));
        asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(f) : "r"(ra) : "memory");
        s += f;
      }
      out[(blockIdx.x / 8 * 64 + threadIdx.x) % 2048] = s;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  }
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}
template <int CHK>
__global__ void __launch_bounds__(32) pr_blk(const uint8_t* buf, long long per, uint32_t* sink) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem), bar = sb + (uint32_t)per;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bar));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"((uint32_t)per) : "memory");
    for (long long o = 0; o < per; o += CHK)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   :: "r"(sb + (uint32_t)o), "l"(buf + blockIdx.x * per + o), "r"(CHK), "r"(bar) : "memory");
  }
  __syncwarp();
  asm volatile("{\n.reg .pred p;\nPRB_W:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra PRB_W;\n}\n" :: "r"(bar) : "memory");
  uint32_t a; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(a) : "r"(sb + threadIdx.x * 4));
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}
// kind: 0-3 ldg NL 16 / 8 / 4 / 2 (no reduction), 4 ldg NL 8 + red, 5 ldg NL 8 + cluster-8 DSMEM, 6 bulk 8 KB, 7 bulk 16 KB
int pr_launch(int kind, int G, const void* buf, long long bytes, float* out, uint32_t* sink, cudaStream_t st) {
  const long long n16 = bytes / 16;
  switch (kind) {
    case 0: pr_ldg<16, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 1: pr_ldg<8, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 2: pr_ldg<4, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 3: pr_ldg<2, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 4: pr_ldg<8, 1><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 5: {
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(G); cfg.blockDim = dim3(256); cfg.stream = st;
      cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
      at[0].val.clusterDim.x = 8; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
      if (cudaLaunchKernelEx(&cfg, pr_ldg<8, 2>, (const uint4*)buf, n16, out, sink) != cudaSuccess) { cudaGetLastError(); return -3; }
      break;
    }
    case 6: case 7: {
      const long long per = bytes / G;
      const int chk = kind == 6 ? 8192 : 16384;
      if (per % chk || per > 200 * 1024) return -4;
      auto k = kind == 6 ? pr_blk<8192> : pr_blk<16384>;
      if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)per + 64) != cudaSuccess) { cudaGetLastError(); return -6; }
      k<<<G, 32, (int)per + 64, st>>>((const uint8_t*)buf, per, sink);
      break;
    }
    default: return -2;
  }
  return (int)cudaGetLastError();
}
int64_t rrrun(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out) {
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  const float4* w = (const float4*)W.data_ptr(); const __nv_bfloat16* xp = (const __nv_bfloat16*)x.data_ptr(); float* o = out.data_ptr<float>();
  if (K != 8192) return -1;
  switch (inst) {
    case 0: if (m > 1) return -1; rr<1, 1><<<N, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 1: if (m > 1) return -1; rr<1, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 2: if (m > 8) return -1; rr<8, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 3: if (m > 8) return -1; rr<8, 4><<<N / 4, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 4: if (m > 1) return -1; rr<1, 1, 256><<<N, 256, 0, st>>>(w, xp, o, m, N, K); break;
    case 5: if (m > 1) return -1; rr<1, 2, 256><<<N / 2, 256, 0, st>>>(w, xp, o, m, N, K); break;
    case 6: if (m > 1) return -1; rr<1, 1, 1024><<<N, 1024, 0, st>>>(w, xp, o, m, N, K); break;
    case 7: if (m > 1) return -1; rr<1, 2, 1024><<<N / 2, 1024, 0, st>>>(w, xp, o, m, N, K); break;
    default: return -2;
  }
  const cudaError_t r = cudaGetLastError();
  return (int64_t)r;
}

static int64_t gvgo(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U, int O) {
  G p;
  p.W = (const uint4*)W.data_ptr(); p.W2 = (const uint4*)W2.data_ptr(); p.x = (const __nv_bfloat16*)x.data_ptr(); p.out = out.data_ptr();
  p.esc = (const uint32_t*)esc.data_ptr(); p.ws = (float4*)ws.data_ptr(); p.cnt = cnt.data_ptr<int>();
  p.m = x.size(0); p.K = x.size(1); p.N = out.size(1); p.KC = p.K / 64;
  p.T0 = (uint32_t)(T & 0xffffffff); p.T1 = (uint32_t)((uint64_t)T >> 32); p.U0 = (uint32_t)(U & 0xffffffff); p.U1 = (uint32_t)((uint64_t)U >> 32);
  auto st = at::cuda::getCurrentCUDAStream();
  const int occ = O;
  switch (inst) {
@@CASES@@
  }
  return -2;
}
int64_t gvrun(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 0);
}
int64_t gvocc(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 1);
}
int64_t prrun(int64_t kind, int64_t G, torch::Tensor buf, torch::Tensor out, torch::Tensor sink) {
  return (int64_t)pr_launch((int)kind, (int)G, buf.data_ptr(), (long long)buf.numel() * buf.element_size(), out.data_ptr<float>(),
                            (uint32_t*)sink.data_ptr(), at::cuda::getCurrentCUDAStream());
}
int64_t rsrun(int64_t v, torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt) {
  if (x.size(1) != hi.size(1) || out.size(0) != x.size(0) || out.size(1) != hi.size(0)) return -10;
  return (int64_t)rs_launch((int)v, (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)hi.data_ptr(), (const __nv_bfloat16*)lo.data_ptr(),
                            out.data_ptr<float>(), (float4*)ws.data_ptr(), (long long)(ws.numel() / 4), cnt.data_ptr<int>(), (int)cnt.numel(),
                            (int)x.size(0), (int)out.size(1), (int)x.size(1), at::cuda::getCurrentCUDAStream());
}
int64_t gvregs(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 2);
}
