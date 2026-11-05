
// H200 memory-pipeline microbenchmarks (diagnostics only).
#include <cuda_runtime.h>
#include <stdint.h>

namespace bw {


// TMA bulk (cp.async.bulk) ring: thread 0 issues 16 KB bulk copies into NS stages, full/empty mbarriers.
template <int NS>
__global__ void __launch_bounds__(256, 1) tma_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t full[NS];
  const int tid = threadIdx.x;
  constexpr int ST16 = 1024;
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  if (tid == 0) {
    for (int k = 0; k < NS; ++k) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[k]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(b));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  auto issue = [&](size_t s, int slot) {
    if (s < s1) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(16384));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(d), "l"(src + s * ST16), "r"(16384), "r"(b) : "memory");
    }
  };
  if (tid == 0)
    for (int k = 0; k < NS; ++k) issue(s0 + k, k);
  unsigned acc = 0;
  for (size_t s = s0; s < s1; ++s) {
    const int slot = (int)((s - s0) % NS);
    const uint32_t phase = (uint32_t)(((s - s0) / NS) & 1);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
        ::"r"(b), "r"(phase));
    const uint4* st = sm + slot * ST16;
#pragma unroll
    for (int k = 0; k < 4; ++k) { uint4 v = st[k * 256 + tid]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncthreads();
    if (tid == 0) issue(s + NS, slot);
  }
  if (acc == 0x12345678u) out[0] = acc;
}


template <int SB, int NS, int P, bool MIX>
__global__ void __launch_bounds__(256) tma2_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t full[NS];
  const int tid = threadIdx.x;
  constexpr int ST16 = SB / 16;
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  if (tid == 0) {
    for (int k = 0; k < NS; ++k) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[k]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(b), "r"(MIX ? 257 : 1));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  auto issue = [&](size_t s, int slot) {
    if (s < s1) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      constexpr int TB = MIX ? SB / 2 : SB;  // bytes via TMA
      if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(TB));
        for (int k = 0; k < P; ++k) {
          uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16 + k * (TB / 16 / P));
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                       ::"r"(d), "l"(src + s * ST16 + k * (TB / 16 / P)), "r"(TB / P), "r"(b) : "memory");
        }
      }
      if (MIX) {
        for (int i = tid; i < (SB - TB) / 16; i += 256) {
          uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16 + TB / 16 + i);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src + s * ST16 + TB / 16 + i));
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(b) : "memory");
      }
    } else if (MIX) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      (void)b;
    }
  };
  for (int k = 0; k < NS; ++k) issue(s0 + k, k);
  unsigned acc = 0;
  for (size_t s = s0; s < s1; ++s) {
    const int slot = (int)((s - s0) % NS);
    const uint32_t phase = (uint32_t)(((s - s0) / NS) & 1);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
        ::"r"(b), "r"(phase));
    const uint4* st = sm + slot * ST16;
    for (int k = tid; k < ST16; k += 256) { uint4 v = st[k]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncthreads();
    issue(s + NS, slot);
  }
  if (acc == 0x12345678u) out[0] = acc;
}


}  // namespace bw

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void bw_run(torch::Tensor src, torch::Tensor out, int64_t mode, int64_t grid) {
  const uint4* p = (const uint4*)src.data_ptr();
  size_t n16 = src.numel() * src.element_size() / 16;
  unsigned* o = (unsigned*)out.data_ptr();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (mode) {
    case 5: {
      size_t sm = 6 * 16384;
      cudaFuncSetAttribute(bw::tma_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma_kernel<6><<<(int)grid, 256, sm, st>>>(p, n16, o); break;
    }
    case 8: { constexpr int SB = 32768, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 12: { constexpr int SB = 16384, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid * 2, 256, sm, st>>>(p, n16, o); break; }
    case 16: { constexpr int SB = 36864, NS = 5; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 17: { constexpr int SB = 36864, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 18: { constexpr int SB = 36864, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 2, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 19: { constexpr int SB = 36864, NS = 3; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 20: { constexpr int SB = 32768, NS = 5; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 21: { constexpr int SB = 16384, NS = 8; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 22: { constexpr int SB = 16384, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 23: { constexpr int SB = 16384, NS = 6; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 24: { constexpr int SB = 36864, NS = 2; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid * 2, 256, sm, st>>>(p, n16, o); break; }
  }
}

