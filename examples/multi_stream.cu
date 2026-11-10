#include <cuda_timer/timer.cuh>
#include <cuda_timer/csv_logger.cuh>
#include <cuda_timer/nvtx.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

// Burns ALU cycles on one element so each launch has a measurable duration.
__global__ void busy_kernel(float* data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < iterations; ++i) {
            val = val * 1.0001f + 0.0001f;
        }
        data[idx] = val;
    }
}

void run_stream(int stream_id, int device_id, int iterations, cuda_timer::CsvLogger& logger) {
    cudaSetDevice(device_id);

    // A non-default stream is what lets the two threads overlap.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int N = 1 << 18;  // 256K elements per stream
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemsetAsync(d_data, 0, N * sizeof(float), stream);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cuda_timer::Timer timer(stream);

    // Names per stream, otherwise Nsight shows identical ranges.
    cuda_timer::nvtx::ScopedRange range(("stream_" + std::to_string(stream_id)).c_str());

    for (int i = 0; i < iterations; ++i) {
        timer.start();
        busy_kernel<<<grid, block, 0, stream>>>(d_data, N, 100);
        float ms = timer.stop();

        cuda_timer::TimingRecord record;
        record.label = "stream_" + std::to_string(stream_id);
        record.duration_ms = ms;
        record.device_id = device_id;
        record.stream_id = stream_id;  // Logical stream ID
        record.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        record.grid_x = grid.x; record.grid_y = grid.y; record.grid_z = grid.z;
        record.block_x = block.x; record.block_y = block.y; record.block_z = block.z;
        record.kernel_name = "busy_kernel";

        logger.log(record);
    }

    cudaStreamSynchronize(stream);

    cudaFree(d_data);
    cudaStreamDestroy(stream);
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    std::cout << "Found " << device_count << " CUDA device(s)\n";

    // One logger for every thread: CsvLogger serializes the writes.
    cuda_timer::CsvLogger logger("multi_stream_timing.csv");

    const int iterations = 10;
    std::vector<std::thread> threads;

    int streams_to_use = std::min(2, device_count);

    std::cout << "Launching " << streams_to_use << " concurrent stream(s), "
              << iterations << " iterations each...\n";

    for (int i = 0; i < streams_to_use; ++i) {
        threads.emplace_back(run_stream, i, i, iterations, std::ref(logger));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Multi-stream done. Results in multi_stream_timing.csv\n";
    return 0;
}