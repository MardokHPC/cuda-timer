#include <cuda_timer/timer.cuh>
#include <cuda_timer/csv_logger.cuh>
#include <cuda_timer/nvtx.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

// One thread per output element, no tiling: this is a demo workload.
__global__ void matrix_mul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_on_device(int device_id, cuda_timer::CsvLogger& logger) {
    cudaSetDevice(device_id);

    const int N = 512;  // 512x512 matrices
    size_t bytes = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Byte fill, not float fill: memset writes 0x01 per byte, i.e. about
    // 2.4e-38 as a float rather than 1.0f. Good enough for a throughput demo.
    cudaMemset(d_A, 1, bytes);
    cudaMemset(d_B, 2, bytes);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // The timer captures whichever device this thread selected above.
    cuda_timer::Timer timer(0, cudaEventDefault);

    cuda_timer::nvtx::ScopedRange range(("device_" + std::to_string(device_id)).c_str());

    for (int i = 0; i < 5; ++i) {
        timer.start();
        matrix_mul<<<grid, block>>>(d_A, d_B, d_C, N);
        float ms = timer.stop();

        cuda_timer::TimingRecord record;
        record.label = "matrix_mul";
        record.duration_ms = ms;
        record.device_id = device_id;
        record.stream_id = 0;
        record.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        record.grid_x = grid.x; record.grid_y = grid.y; record.grid_z = grid.z;
        record.block_x = block.x; record.block_y = block.y; record.block_z = block.z;
        record.kernel_name = "matrix_mul";

        logger.log(record);

        std::cout << "Device " << device_id << " iter " << i << ": " << ms << " ms\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::cout << "Found " << device_count << " CUDA device(s)\n";

    if (device_count < 2) {
        std::cout << "Need at least 2 GPUs for multi-device demo. Running on device 0 only.\n";
    }

    // One logger for every thread: CsvLogger serializes the writes.
    cuda_timer::CsvLogger logger("multi_device_timing.csv");

    std::vector<std::thread> threads;

    int devices_to_use = std::min(2, device_count);

    std::cout << "Running on " << devices_to_use << " device(s)...\n";

    for (int i = 0; i < devices_to_use; ++i) {
        threads.emplace_back(run_on_device, i, std::ref(logger));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Multi-device done. Results in multi_device_timing.csv\n";
    return 0;
}