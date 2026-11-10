#pragma once

// Device-side helpers for the CUDA test suites.
//
// GoogleTest is never included from a CUDA translation unit: nvcc's device
// pass cannot parse gtest's portability layer, which needs POSIX declarations
// (rmdir, isatty, struct stat) that are invisible there, and the build dies
// inside gtest-port.h. The suites are host .cpp files; only kernels live here,
// behind host-callable launchers.

#include <cuda_runtime.h>

namespace cuda_test {

// Launches an empty kernel with the given configuration. The Timer and
// Benchmark suites only need "some stream work" to bracket with events.
void launch_dummy(dim3 grid = dim3(1), dim3 block = dim3(1), cudaStream_t stream = 0);

}  // namespace cuda_test
