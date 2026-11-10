// The only CUDA translation unit of the test suite: GoogleTest must not go
// through nvcc (see cuda_test_kernels.cuh).

#include "cuda_test_kernels.cuh"

namespace {

__global__ void dummy_kernel() {}

}  // namespace

namespace cuda_test {

void launch_dummy(dim3 grid, dim3 block, cudaStream_t stream) {
    dummy_kernel<<<grid, block, 0, stream>>>();
}

}  // namespace cuda_test
