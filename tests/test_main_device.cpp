#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>

// Device suites create events, streams and launches, so they need a real GPU.
// Exit 77 (= ctest SKIP_RETURN_CODE) without one: an explicit "skipped" beats a
// mix of passing host-only checks and failing device ones.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cout << "[ SKIPPED ] no CUDA device available (cudaGetDeviceCount -> "
                  << static_cast<int>(status) << ")\n";
        return 77;
    }

    return RUN_ALL_TESTS();
}
