
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <stdint.h>
@@WG@@
__device__ __forceinline__ void rw_cpa(uint32_t dst, const void* src, uint32_t sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void rw_cpwait(int n) {  // n is a constant after unrolling: the switch folds
  switch (n) {
    case 7: asm volatile("cp.async.wait_group 7;\n" ::: "memory"); break;
    case 6: asm volatile("cp.async.wait_group 6;\n" ::: "memory"); break;
    case 5: asm volatile("cp.async.wait_group 5;\n" ::: "memory"); break;
    case 4: asm volatile("cp.async.wait_group 4;\n" ::: "memory"); break;
    case 3: asm volatile("cp.async.wait_group 3;\n" ::: "memory"); break;
    case 2: asm volatile("cp.async.wait_group 2;\n" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 0;\n" ::: "memory"); break;
  }
}
// K-major, 128B swizzle: start >> 4, LBO 1 (unused), SBO 1024 B (8 rows x 128 B), base offset 0, mode 1 << 62
__device__ __forceinline__ uint64_t rw_desc(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void rw_red4(float* a, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
template <int NA> __device__ __forceinline__ void rw_keep(float* d) {
#pragma unroll
  for (int i = 0; i < NA; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <int WN> __device__ __forceinline__ void rw_mma(float* d, uint64_t a, uint64_t b) {
  if constexpr (WN == 128) rw_mma128(d, a, b); else rw_mma256(d, a, b);
}
// slot = one 64-k chunk: x rows [0, TOKS), hi rows [TOKS, TOKS + 64), lo rows [TOKS + 64, TOKS + 128), 128 B each;
// 16-B column c of row r at slot + r * 128 + ((c ^ (r & 7)) << 4) (TOKS % 8 == 0: the hi / lo tiles keep the phase).
// MODE bits: 1 loads only (ladder), 2 no reductions (ladder), 4 self-zeroing (no memset node: the first CTA of each
// output tile to arrive in a call zeroes the tile and releases a per-tile flag; the others acquire it before their
// reds; per-tile arrival counters are monotonic across calls, so nothing is reset). CL > 1: cluster of CL K-splits
// pre-reduces its partials through DSMEM, each CTA then reds 1/CL of the token rows.
__device__ __forceinline__ int rw_ldacq(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void rw_strel(int* p, int v) {
  asm volatile("st.release.gpu.global.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long rw_gt() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t) :: "memory"); return t;
}
__device__ __forceinline__ void rw_csync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
template <int KR, int NSTG, int NBUF, int TOKS, int WN, int MINB, int MODE, int CL>
__global__ void __launch_bounds__(128, MINB) rwk(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ hi,
                                                 const __nv_bfloat16* __restrict__ lo, float* out, int m, int N, int K,
                                                 int* cnt, int slot) {
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __trap();  // an unexpected plain sm_90 pass compiles to a stub instead of failing the extension
#else
  constexpr bool RING = NBUF < NSTG;
  constexpr int QT = KR / 16, QS = QT / NSTG, CPS = 2 * QS, NSLOT = RING ? NBUF : KR / 64;
  constexpr int ROWS = TOKS + 128, SB = ROWS * 128, NA = TOKS / 2, NH = TOKS / WN, SP = 68, LI = ROWS * CPS / 128;
  static_assert(KR % 64 == 0 && QT % NSTG == 0 && (ROWS * CPS) % 128 == 0, "stages");
  static_assert(RING ? QS == 4 : NBUF == NSTG, "a ring stage is one 64-k chunk");
  static_assert(TOKS % WN == 0 && TOKS * SP * 4 <= NSLOT * SB, "epilogue staging");
  extern __shared__ uint8_t rw_smem[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(rw_smem), base = (raw + 1023u) & ~1023u;
  float* sd = (float*)(rw_smem + (base - raw));
  const int n0 = blockIdx.x * 64, k0 = blockIdx.y * KR, t0 = blockIdx.z * TOKS, tid = threadIdx.x;
  constexpr bool ST = (MODE & 16) != 0;  // MODE 16: globaltimer stamps by thread 0 into cnt[8192..] (8 x u64 per CTA)
  unsigned long long ts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (ST && tid == 0) ts[0] = rw_gt();
  auto load = [&](int s) {
#pragma unroll
    for (int it = 0; it < LI; it++) {
      const int i = tid + it * 128, r = i / CPS, cc = s * CPS + i % CPS, c = cc & 7;
      const uint32_t dst = base + ((cc >> 3) % NSLOT) * SB + r * 128 + ((c ^ (r & 7)) << 4);
      if (r < TOKS) {
        const int tok = t0 + r;
        rw_cpa(dst, x + (size_t)min(tok, m - 1) * K + k0 + cc * 8, tok < m ? 16u : 0u);
      } else if (r < TOKS + 64) {
        rw_cpa(dst, hi + (size_t)(n0 + r - TOKS) * K + k0 + cc * 8, 16u);
      } else {
        rw_cpa(dst, lo + (size_t)(n0 + r - TOKS - 64) * K + k0 + cc * 8, 16u);
      }
    }
  };
  float d[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) d[i] = 0.f;
#pragma unroll
  for (int s = 0; s < (RING ? NBUF : NSTG); s++) {
    load(s);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  if (ST && tid == 0) ts[1] = rw_gt();
  const int rows = min(TOKS, m - t0);
  int* ctr = cnt + (slot * 64 + blockIdx.x + gridDim.x * blockIdx.z) * 2;  // per-tile arrival counter (monotonic)
  int arrived = 0;
  if constexpr ((MODE & 4) != 0) {  // self-zeroing: each of the G K-split CTAs zeroes 1/G of the tile, then arrives
    const int G = (int)gridDim.y, E = rows * 16, e0 = (int)blockIdx.y * E / G, e1 = ((int)blockIdx.y + 1) * E / G;
    for (int i = e0 + tid; i < e1; i += 128) {
      const int r = i >> 4, c4 = i & 15;
      *(float4*)(out + (size_t)(t0 + r) * N + n0 + 4 * c4) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    if (tid == 0) asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(arrived) : "l"(ctr) : "memory");
  }
  rw_keep<NA>(d);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
  for (int s = 0; s < NSTG; s++) {
    rw_cpwait(RING ? (s == 0 ? NBUF - 1 : NBUF - 2) : NSTG - 1 - s);  // stage s landed (this thread's copies)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");     // generic (cp.async) writes -> wgmma reads
    __syncthreads();
    if (ST && tid == 0 && s == 0) ts[2] = rw_gt();
    if (ST && tid == 0 && s == NSTG - 1) ts[3] = rw_gt();
#pragma unroll
    for (int qq = 0; qq < QS; qq++) {
      const int q = s * QS + qq;
      const uint32_t sb = base + ((q >> 2) % NSLOT) * SB + (q & 3) * 32;
      const uint64_t dh = rw_desc(sb + TOKS * 128), dl = rw_desc(sb + (TOKS + 64) * 128);
#pragma unroll
      for (int h = 0; h < NH; h++) {
        const uint64_t db = rw_desc(sb + h * WN * 128);
        if constexpr (!(MODE & 1)) {  // MODE 1 (ladder): loads only, no wgmma
          rw_mma<WN>(d + h * (WN / 2), dh, db);
          rw_mma<WN>(d + h * (WN / 2), dl, db);
        }
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if constexpr (RING) {
      if (s >= 1) {  // stage s - 1's wgmmas done in every warp -> refill its slot with stage s - 1 + NBUF
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        __syncthreads();
        if (s - 1 + NBUF < NSTG) load(s - 1 + NBUF);
        asm volatile("cp.async.commit_group;\n" ::: "memory");  // possibly empty: keeps the group count uniform
        rw_keep<NA>(d);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      }
    }
  }
  if constexpr ((MODE & 4) != 0) {  // every slice of this call's tile zeroed (arrivals of a call are contiguous); polled
    if (tid == 0) {                 // while the last wgmmas run, ordered before the reds by the staging barriers
      const int G = (int)gridDim.y, target = (arrived / G + 1) * G;
      while (rw_ldacq(ctr) < target) { }
    }
    __syncwarp();
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  rw_keep<NA>(d);
  __syncthreads();  // every warp's wgmmas are done before the staging overwrites the tiles
  if (ST && tid == 0) ts[4] = rw_gt();
  const int lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int h = 0; h < NH; h++)
#pragma unroll
    for (int i = 0; i < WN / 8; i++) {  // D: d[4i + e] = (row 16w + g (+8 for e >= 2), token 8i + 2t (+1 for odd e))
      const int tok = h * WN + 8 * i + 2 * t, n = 16 * w + g;
      const float* e = d + h * (WN / 2) + 4 * i;
      sd[tok * SP + n] = e[0]; sd[(tok + 1) * SP + n] = e[1];
      sd[tok * SP + n + 8] = e[2]; sd[(tok + 1) * SP + n + 8] = e[3];
    }
  if constexpr (CL > 1) rw_csync(); else __syncthreads();
  if (ST && tid == 0) ts[5] = rw_gt();
  const int nr = (MODE & 2) ? 0 : rows;  // MODE 2 (ladder): no reductions
  if constexpr (CL == 1) {
    // MODE 8: rotate the row order by K split, so the G CTAs of a tile (finishing together) hit different lines
    const int rot = (MODE & 8) ? (int)blockIdx.y * (nr / (int)gridDim.y + 1) : 0;
    for (int i = tid; i < nr * 16; i += 128) {
      int r = (i >> 4) + rot;
      r = (MODE & 8) ? r % nr : r;
      const int c4 = i & 15;
      rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, *(const float4*)(sd + r * SP + 4 * c4));
    }
  } else {
    uint32_t rk;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
    uint32_t rem[CL];
#pragma unroll
    for (int q = 0; q < CL; q++) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rem[q]) : "r"(base), "r"(q));
    const int mine = nr > (int)rk ? (nr - (int)rk + CL - 1) / CL : 0;  // token rows rk, rk + CL, ...
    for (int i = tid; i < mine * 16; i += 128) {
      const int r = (int)rk + CL * (i >> 4), c4 = i & 15;
      const uint32_t off = (uint32_t)(r * SP + 4 * c4) * 4u;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int q = 0; q < CL; q++) {
        float4 v;
        asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(rem[q] + off) : "memory");
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
      }
      rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, a);
    }
    rw_csync();  // no CTA leaves while a peer may still read its staging
  }
  if constexpr (ST) {
    __syncthreads();
    if (tid == 0) {
      ts[6] = rw_gt();
      unsigned long long* dbg = (unsigned long long*)(cnt + 16384) + 8 * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
#pragma unroll
      for (int j = 0; j < 7; j++) dbg[j] = ts[j];
      uint32_t sm; asm volatile("mov.u32 %0, %%smid;\n" : "=r"(sm));
      dbg[7] = sm;
    }
  }
#endif
}
template <int KR, int NSTG, int NBUF, int TOKS, int WN, int MINB, int MODE, int CL>
static int rw_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, int m, int N, int K,
                 int* cnt, int slot, cudaStream_t st) {
  constexpr int NSLOT = NBUF < NSTG ? NBUF : KR / 64, SM = NSLOT * (TOKS + 128) * 128 + 1024;
  if (N % 64 || K % KR || m < 1 || (K / KR) % CL || (N / 64) * ((m + TOKS - 1) / TOKS) > 64) return -4;
  auto kern = rwk<KR, NSTG, NBUF, TOKS, WN, MINB, MODE, CL>;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = CL; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = CL > 1 ? 1 : 0;
  if (cudaLaunchKernelEx(&cfg, kern, x, hi, lo, out, m, N, K, cnt, slot) != cudaSuccess) return (int)cudaGetLastError();
  return (int)cudaGetLastError();
}
template <int KR, int NSTG, int NBUF, int TOKS, int WN, int MINB, int MODE, int CL>
static int rw_occ(int m, int N, int K) {  // max co-resident clusters (CL > 1) or CTAs per SM (CL == 1)
  constexpr int NSLOT = NBUF < NSTG ? NBUF : KR / 64, SM = NSLOT * (TOKS + 128) * 128 + 1024;
  auto kern = rwk<KR, NSTG, NBUF, TOKS, WN, MINB, MODE, CL>;
  if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
  int n = -1;
  if (CL == 1) {
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, 128, SM) != cudaSuccess) { cudaGetLastError(); return -7; }
    return n;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = CL; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) != cudaSuccess) { cudaGetLastError(); return -8; }
  return n;
}
struct alignas(64) RtMap { unsigned long long v[16]; };
__device__ __forceinline__ void rt_mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void rt_mbwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRT_WAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RT_WAIT_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rt_mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rt_mbexpect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rt_tile(uint32_t dst, const RtMap* tm, int k, int row, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k), "r"(row), "r"(bar) : "memory");
}
__device__ __forceinline__ void rt_tile_mc(uint32_t dst, const RtMap* tm, int k, int row, uint32_t bar, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k), "r"(row), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void rt_rarrive(uint32_t bar, uint32_t rank) {  // arrive on the same mbarrier in CTA `rank`
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(bar), "r"(rank));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" :: "r"(r) : "memory");
}
__device__ __forceinline__ void rt_cbar() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }  // consumer warpgroup only
__device__ __forceinline__ void rt_kcsync() { asm volatile("barrier.cluster.arrive.relaxed;\nbarrier.cluster.wait.acquire;\n" ::: "memory"); }
template <int KR, int NBUF, int TOKS, int WN, int MINB, int MODE>
__global__ void __launch_bounds__((MODE & 4) ? 192 : 160, MINB) rtk(const __grid_constant__ RtMap tx, const __grid_constant__ RtMap th,
                                                 const __grid_constant__ RtMap tl, float* out, int m, int N, int K,
                                                 int* cnt, int slot) {
#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __trap();
#else
  constexpr int NSTG = KR / 64, ROWS = TOKS + 128, SB = ROWS * 128, NA = TOKS / 2, NH = TOKS / WN, SP = 68;
  constexpr int NSLOT = NBUF < NSTG ? NBUF : NSTG;
  static_assert(KR % 64 == 0 && TOKS % WN == 0 && TOKS * SP * 4 <= NSLOT * SB && SB % 1024 == 0, "rtk");
  constexpr bool ST = (MODE & 16) != 0;
  constexpr bool MC = (MODE & 32) != 0;  // cluster of CL n-tiles: each CTA multicasts 1/CL of the x box to all CL
  constexpr int CL = (MODE & 512) ? 2 : 4;  // MODE 512: cluster of 2 n-tiles (half the x traffic, 2x the clusters)
  constexpr bool KC = (MODE & 2048) != 0;   // MODE 2048: cluster of CLK K-splits, partials pre-reduced in DSMEM
  constexpr int CLK = (MODE & 4096) ? 4 : 2;  // MODE 4096: CLK 4 (else 2)
  static_assert(!MC || TOKS % 32 == 0, "mc");
  extern __shared__ uint8_t rw_smem[];
  const uint32_t raw = (uint32_t)__cvta_generic_to_shared(rw_smem), base = (raw + 1023u) & ~1023u;
  const uint32_t bars = base + NSLOT * SB;  // full[NSLOT], empty[NSLOT]
  float* sd = (float*)(rw_smem + (base - raw));
  const int n0 = blockIdx.x * 64, k0 = blockIdx.y * KR, t0 = blockIdx.z * TOKS, tid = threadIdx.x;
  unsigned long long ts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (ST && tid == 0) ts[0] = rw_gt();
  if ((MODE & 64) && tid == 128) {  // MODE 64: fetch the three tensor maps before the barrier setup
    asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
    asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&th) : "memory");
    asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tl) : "memory");
  }
  constexpr bool ET = (MODE & 128) != 0;  // MODE 128: the producer lane owns the full barriers and issues stage 0
  if (ET) {                               // before the CTA-wide sync, so the first TMA is in flight during it
    if (tid == 128) {
      for (int i = 0; i < NSLOT; i++) rt_mbinit(bars + 8 * i, 1);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
      const uint32_t fb = bars;
      rt_mbexpect(fb, (uint32_t)SB);
      rt_tile(base, &tx, k0, t0, fb);
      rt_tile(base + TOKS * 128, &th, k0, n0, fb);
      rt_tile(base + (TOKS + 64) * 128, &tl, k0, n0, fb);
    }
    if (tid == 0) {
      for (int i = 0; i < NSLOT; i++) rt_mbinit(bars + 8 * (NSLOT + i), MC ? 4 * CL : 4);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
  } else if (tid == 0) {
    for (int i = 0; i < NSLOT; i++) { rt_mbinit(bars + 8 * i, 1); rt_mbinit(bars + 8 * (NSLOT + i), MC ? 4 * CL : 4); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if constexpr (MC) rw_csync(); else __syncthreads();  // MC: every CTA's barriers exist before any multicast / remote arrive
  const int rows = min(TOKS, m - t0);
  if (tid >= 160) {  // MODE 4: the zeroing warp (off the consumers' critical path): zero 1/G of the tile, arrive with
    int* ctr = cnt + (slot * 64 + blockIdx.x + gridDim.x * blockIdx.z) * 2;  // release, poll the call's G arrivals,
    const int G = (int)gridDim.y, E = rows * 16, e0 = (int)blockIdx.y * E / G, e1 = ((int)blockIdx.y + 1) * E / G;
    const int ln = tid - 160;                                                // then meet the consumers before the reds
    for (int i = e0 + ln; i < e1; i += 32) {
      const int r = i >> 4, c4 = i & 15;
      *(float4*)(out + (size_t)(t0 + r) * N + n0 + 4 * c4) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncwarp();
    if (ln == 0) {
      int arrived;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(arrived) : "l"(ctr) : "memory");
      const int target = (arrived / G + 1) * G;
      while (rw_ldacq(ctr) < target) { }
    }
    __syncwarp();
    asm volatile("bar.sync 2, 160;\n" ::: "memory");
    if constexpr (MC) asm volatile("barrier.cluster.arrive.relaxed;\n" ::: "memory");
    if constexpr (KC) { rt_kcsync(); rt_kcsync(); }  // both cluster barriers: the peers read this CTA's staging
    return;
  }
  if (tid >= 128) {  // producer warp
    if (tid == 128) {
      for (int s = ET ? 1 : 0; s < NSTG; s++) {
        const int sl = s % NSLOT;
        if (s >= NSLOT) rt_mbwait(bars + 8 * (NSLOT + sl), (uint32_t)((s / NSLOT - 1) & 1));
        const uint32_t fb = bars + 8 * sl, dst = base + sl * SB;
        rt_mbexpect(fb, (uint32_t)SB);
        if constexpr (MC) {
          uint32_t rk;
          asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
          rt_tile_mc(dst + rk * (TOKS / CL) * 128, &tx, k0 + 64 * s, t0 + (int)rk * (TOKS / CL), fb, (uint16_t)((1u << CL) - 1));
        } else {
          rt_tile(dst, &tx, k0 + 64 * s, t0, fb);
        }
        rt_tile(dst + TOKS * 128, &th, k0 + 64 * s, n0, fb);
        rt_tile(dst + (TOKS + 64) * 128, &tl, k0 + 64 * s, n0, fb);
      }
    }
    if constexpr (MC) asm volatile("barrier.cluster.arrive.relaxed;\n" ::: "memory");
    if constexpr (KC) { rt_kcsync(); rt_kcsync(); }
    return;
  }
  if (ST && tid == 0) ts[1] = rw_gt();
  float d[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) d[i] = 0.f;
  rw_keep<NA>(d);
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll 1
  for (int s = 0; s < NSTG; s++) {
    const int sl = s % NSLOT;
    rt_mbwait(bars + 8 * sl, (uint32_t)((s / NSLOT) & 1));
    if (ST && tid == 0 && s == 0) ts[2] = rw_gt();
    if (ST && tid == 0 && s == NSTG - 1) ts[3] = rw_gt();
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint32_t sb = base + sl * SB + q * 32;
      const uint64_t dh = rw_desc(sb + TOKS * 128), dl = rw_desc(sb + (TOKS + 64) * 128);
#pragma unroll
      for (int h = 0; h < NH; h++) {
        const uint64_t db = rw_desc(sb + h * WN * 128);
        rw_mma<WN>(d + h * (WN / 2), dh, db);
        rw_mma<WN>(d + h * (WN / 2), dl, db);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if (s >= 1) {
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");  // stage s - 1 retired in this warp
      rw_keep<NA>(d);
      if ((tid & 31) == 0 && s - 1 + NSLOT < NSTG) {
        if constexpr (MC) {
#pragma unroll
          for (uint32_t q = 0; q < CL; q++) rt_rarrive(bars + 8 * (NSLOT + (s - 1) % NSLOT), q);
        } else {
          rt_mbarrive(bars + 8 * (NSLOT + (s - 1) % NSLOT));
        }
      }
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    }
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  rw_keep<NA>(d);
  rt_cbar();
  if (ST && tid == 0) ts[4] = rw_gt();
  const int lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int h = 0; h < NH; h++)
#pragma unroll
    for (int i = 0; i < WN / 8; i++) {
      const int tok = h * WN + 8 * i + 2 * t, n = 16 * w + g;
      const float* e = d + h * (WN / 2) + 4 * i;
      sd[tok * SP + n] = e[0]; sd[(tok + 1) * SP + n] = e[1];
      sd[tok * SP + n + 8] = e[2]; sd[(tok + 1) * SP + n + 8] = e[3];
    }
  if constexpr ((MODE & 4) != 0) asm volatile("bar.sync 2, 160;\n" ::: "memory");  // staged + this call's tile zeroed
  else rt_cbar();
  if constexpr (MC) asm volatile("barrier.cluster.arrive.relaxed;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
  if constexpr (KC) rt_kcsync();
  if (ST && tid == 0) ts[5] = rw_gt();
  if constexpr (KC) {  // MODE 2048: sum the CLK partials of this cluster in DSMEM, then one red per CLK rows
    uint32_t rk;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
    uint32_t rem[CLK];
#pragma unroll
    for (int q = 0; q < CLK; q++) asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rem[q]) : "r"(base), "r"(q));
    const int mine = rows > (int)rk ? (rows - (int)rk + CLK - 1) / CLK : 0;
    for (int i = tid; i < mine * 16; i += 128) {
      const int r = (int)rk + CLK * (i >> 4), c4 = i & 15;
      const uint32_t off = (uint32_t)(r * SP + 4 * c4) * 4u;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int q = 0; q < CLK; q++) {
        float4 v;
        asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(rem[q] + off) : "memory");
        a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
      }
      rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, a);
    }
    rt_kcsync();  // no CTA leaves while a peer may still read its staging
  } else {
  const int rot = (MODE & 1024) ? (int)blockIdx.y * (rows / (int)gridDim.y + 1) : 0;  // MODE 1024: rotate the row order
  for (int i = tid; i < rows * 16; i += 128) {
    int r = i >> 4;
    if (MODE & 1024) r = (r + rot) % rows;
    const int c4 = i & 15;
    rw_red4(out + (size_t)(t0 + r) * N + n0 + 4 * c4, *(const float4*)(sd + r * SP + 4 * c4));
  }
  }
  if constexpr (ST) {
    rt_cbar();
    if (tid == 0) {
      ts[6] = rw_gt();
      unsigned long long* dbg = (unsigned long long*)(cnt + 16384) + 8 * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
#pragma unroll
      for (int j = 0; j < 7; j++) dbg[j] = ts[j];
      uint32_t smid; asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
      dbg[7] = smid;
    }
  }
#endif
}
typedef CUresult (*RtEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                               const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                               CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RtMap), "CUtensorMap size");
static int rt_map(RtMap& tm, const void* p, int rows, int K, int box_rows) {
  static RtEncodeFn enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RtEncodeFn)fn;
  }
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)p, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int KR, int NBUF, int TOKS, int WN, int MINB, int MODE>
static int rt_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, int m, int N, int K,
                 int* cnt, int slot, cudaStream_t st) {
  constexpr int NSTG = KR / 64, NSLOT = NBUF < NSTG ? NBUF : NSTG, SM = NSLOT * (TOKS + 128) * 128 + 1024 + 16 * NSLOT;
  if (N % 64 || K % KR || m < 1 || (N / 64) * ((m + TOKS - 1) / TOKS) > 64) return -4;
  auto kern = rtk<KR, NBUF, TOKS, WN, MINB, MODE>;
  static bool once = false;
  static int slots = 0;  // co-resident CTAs (self-zeroing waits across the K splits of a tile: never launch a 2-wave grid)
  if (!once) {
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    int per = 0, dev = 0, sms = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kern, (MODE & 4) ? 192 : 160, SM) != cudaSuccess) { cudaGetLastError(); return -7; }
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    slots = per * sms;
    once = true;
  }
  constexpr int CL = (MODE & 512) ? 2 : 4;
  constexpr int CLK = (MODE & 4096) ? 4 : 2;
  if ((MODE & 2048) && (K / KR) % CLK) return -4;
  if ((MODE & 4) && ((MODE & 32) || (MODE & 2048))) {  // clustered + self-zeroing: every cluster must be co-resident
    cudaLaunchConfig_t oc = {};
    oc.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
    oc.blockDim = dim3(192, 1, 1);
    oc.dynamicSmemBytes = SM;
    cudaLaunchAttribute oa[1];
    oa[0].id = cudaLaunchAttributeClusterDimension;
    oa[0].val.clusterDim.x = (MODE & 32) ? CL : 1; oa[0].val.clusterDim.y = (MODE & 2048) ? CLK : 1; oa[0].val.clusterDim.z = 1;
    oc.attrs = oa; oc.numAttrs = 1;
    int maxcl = -1;
    if (cudaOccupancyMaxActiveClusters(&maxcl, kern, &oc) != cudaSuccess) { cudaGetLastError(); return -31; }
    const int need = (MODE & 32) ? (N / 64 / CL) * (K / KR) * ((m + TOKS - 1) / TOKS) : (N / 64) * (K / KR / CLK) * ((m + TOKS - 1) / TOKS);
    if (maxcl < need) return -31;
  }
  if ((MODE & 4) && (N / 64) * (K / KR) * ((m + TOKS - 1) / TOKS) > slots) return -30;
  constexpr bool MC = (MODE & 32) != 0;
  RtMap tx, th, tl;
  int r = rt_map(tx, x, m, K, MC ? TOKS / CL : TOKS);
  if (!r) r = rt_map(th, hi, N, K, 64);
  if (!r) r = rt_map(tl, lo, N, K, 64);
  if (r) return r;
  if (MC && (N / 64) % CL) return -4;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3((MODE & 4) ? 192 : 160, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = MC ? CL : 1; at[0].val.clusterDim.y = (MODE & 2048) ? CLK : 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = (MC || (MODE & 2048)) ? 1 : 0;
  if (cudaLaunchKernelEx(&cfg, kern, tx, th, tl, out, m, N, K, cnt, slot) != cudaSuccess) return (int)cudaGetLastError();
  return (int)cudaGetLastError();
}
template <int KR, int NBUF, int TOKS, int WN, int MINB, int MODE>
static int rt_occ(int m, int N, int K) {  // co-resident clusters of 4 (MODE 32) or CTAs per SM
  constexpr int NSTG = KR / 64, NSLOT = NBUF < NSTG ? NBUF : NSTG, SM = NSLOT * (TOKS + 128) * 128 + 1024 + 16 * NSLOT;
  auto kern = rtk<KR, NBUF, TOKS, WN, MINB, MODE>;
  if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
  const int nt = (MODE & 4) ? 192 : 160;
  int n = -1;
  if (!(MODE & 32) && !(MODE & 2048)) {
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kern, nt, SM) != cudaSuccess) { cudaGetLastError(); return -7; }
    return n;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / 64, K / KR, (m + TOKS - 1) / TOKS);
  cfg.blockDim = dim3(nt, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = (MODE & 32) ? ((MODE & 512) ? 2 : 4) : 1;
  at[0].val.clusterDim.y = (MODE & 2048) ? ((MODE & 4096) ? 4 : 2) : 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) != cudaSuccess) { cudaGetLastError(); return -8; }
  return n;
}
int64_t rwocc(int64_t v, int64_t m, int64_t N, int64_t K) {
  switch (v) {
@@RTOCC@@
@@RWOCC@@
  }
  return -2;
}
int64_t rwrun(int64_t v, torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out, torch::Tensor cnt) {
  if (x.size(1) != hi.size(1) || out.size(0) != x.size(0) || out.size(1) != hi.size(0) || !x.is_contiguous() || !out.is_contiguous()) return -10;
  if (((uintptr_t)out.data_ptr() & 15) || ((uintptr_t)x.data_ptr() & 15)) return -11;
  if (cnt.numel() < 128 * 64 * 2) return -12;
  if (v >= 20000 && cnt.numel() < 16384 + 2 * 8 * 1024) return -13;
  const __nv_bfloat16 *xp = (const __nv_bfloat16*)x.data_ptr(), *h = (const __nv_bfloat16*)hi.data_ptr(), *l = (const __nv_bfloat16*)lo.data_ptr();
  float* o = out.data_ptr<float>();
  int* c = cnt.data_ptr<int>();
  const int m = x.size(0), N = out.size(1), K = x.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  switch (v) {
@@RWCASES@@
@@RTCASES@@
  }
  return -2;
}
