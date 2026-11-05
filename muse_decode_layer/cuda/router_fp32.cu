
#include <cuda_runtime.h>
#include <stdint.h>

__global__ __launch_bounds__(256) void router_fp32_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int m) {
  const int expert = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* row = w + (size_t)expert * 8192;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
  for (int k = tid * 4; k < 8192; k += 1024) {
    const float4 b = *(const float4*)(row + k);
#pragma unroll
    for (int t = 0; t < 8; ++t) if (t < m) {
      const float4 a = *(const float4*)(x + (size_t)t * 8192 + k);
      acc[t] = fmaf(a.x, b.x, acc[t]);
      acc[t] = fmaf(a.y, b.y, acc[t]);
      acc[t] = fmaf(a.z, b.z, acc[t]);
      acc[t] = fmaf(a.w, b.w, acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < 8; ++t) {
#pragma unroll
    for (int d = 16; d; d >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], d);
  }
  __shared__ float partial[8][8];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < 8; ++t) partial[warp][t] = acc[t];
  }
  __syncthreads();
  if (warp < m) {
    float v = lane < 8 ? partial[lane][warp] : 0.f;
#pragma unroll
    for (int d = 4; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    if (lane == 0) out[(size_t)warp * 256 + expert] = v;
  }
}

int router_fp32_launch(int64_t x, int64_t w, int64_t out, int m, int64_t stream) {
  if (m < 1 || m > 8) return -1;
  router_fp32_kernel<<<256, 256, 0, (cudaStream_t)stream>>>(
      (const float*)x, (const float*)w, (float*)out, m);
  return (int)cudaGetLastError();
}
