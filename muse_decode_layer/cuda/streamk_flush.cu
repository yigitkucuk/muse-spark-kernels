
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// READ flush: every 16-byte line is loaded (default .ca policy, so it allocates in L2 and evicts whatever was there).
// The XOR accumulator is kept live by a comparison that is never true in practice, so no store is needed.
__global__ void flush_read_kernel(const uint4* __restrict__ p, long n4, float* sink) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long stride = (long)gridDim.x * blockDim.x;
  unsigned a = 0u, b = 0u, c = 0u, d = 0u;
  for (; i < n4; i += stride) {
    uint4 v = p[i];
    a ^= v.x; b ^= v.y; c ^= v.z; d ^= v.w;
  }
  if ((a & b & c & d) == 0xFFFFFFFFu) sink[0] = 1.0f;
}

// probe: same access pattern, result written so the loads are live.
__global__ void probe_read_kernel(const uint4* __restrict__ p, long n4, float* out) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long stride = (long)gridDim.x * blockDim.x;
  unsigned a = 0u, b = 0u, c = 0u, d = 0u;
  for (; i < n4; i += stride) {
    uint4 v = p[i];
    a ^= v.x; b ^= v.y; c ^= v.z; d ^= v.w;
  }
  if ((a & b & c & d) == 0xFFFFFFFFu) out[0] = 1.0f;
}

// empty kernel for the graph fixed-cost measurement (132 CTAs, like the benchmark's probe graph).
__global__ void nop_kernel(float* p) {
  if (threadIdx.x == 1024) p[0] = 1.0f;
}

void flush_read(torch::Tensor buf, torch::Tensor sink, int64_t blocks, int64_t threads) {
  long n4 = (long)(buf.numel() * buf.element_size() / 16);
  flush_read_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)buf.data_ptr(), n4, sink.data_ptr<float>());
}
void probe_read(torch::Tensor buf, torch::Tensor out, int64_t blocks, int64_t threads) {
  long n4 = (long)(buf.numel() * buf.element_size() / 16);
  probe_read_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)buf.data_ptr(), n4, out.data_ptr<float>());
}
void nop(torch::Tensor out, int64_t blocks, int64_t threads) {
  nop_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>());
}
