#include <cuda_timer/benchmark.cuh>
#include <cuda_timer/csv_logger.cuh>
#include <cuda_timer/json_logger.cuh>
#include <cuda_timer/nvtx.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

// y = a*x + y: memory-bound, the classic benchmark shape.
__global__ void saxpy_kernel(float a, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

void run_benchmark(const char* name, cuda_timer::SyncMode mode,
                   cuda_timer::CsvLogger& csv, cuda_timer::JsonLogger& json) {
    const int N = 1 << 22;  // 4M elements
    std::vector<float> h_x(N, 1.0f), h_y(N, 2.0f);

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    auto kernel = [&]() {
        saxpy_kernel<<<grid, block>>>(2.5f, d_x, d_y, N);
    };

    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 10;
    cfg.measure_runs = 100;
    cfg.sync_mode = mode;
    cfg.store_samples = true;  // Without samples, median() returns 0.0
    cfg.use_nvtx = true;

    std::cout << "\n=== " << name << " ("
              << (mode == cuda_timer::SyncMode::Blocking ? "Blocking" : "NonBlocking")
              << ") ===\n";

    cuda_timer::Stats stats = cuda_timer::benchmark(kernel, cfg);

    std::cout << "  Runs:       " << stats.count() << "\n";
    std::cout << "  Mean:       " << std::fixed << std::setprecision(4)
              << stats.mean() << " ms\n";
    std::cout << "  StdDev:     " << stats.stddev() << " ms\n";
    std::cout << "  Min:        " << stats.min() << " ms\n";
    std::cout << "  Max:        " << stats.max() << " ms\n";
    std::cout << "  Median:     " << stats.median() << " ms\n";

    cuda_timer::TimingRecord record;
    record.label = name;
    record.duration_ms = static_cast<float>(stats.mean());
    record.device_id = 0;
    record.stream_id = 0;
    record.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    record.grid_x = grid.x; record.grid_y = grid.y; record.grid_z = grid.z;
    record.block_x = block.x; record.block_y = block.y; record.block_z = block.z;
    record.kernel_name = "saxpy_kernel";

    csv.log(record);
    json.log(record);

    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    cuda_timer::CsvLogger csv("benchmark_results.csv");
    cuda_timer::JsonLogger json("benchmark_results.json", true);

    std::cout << "=== cuda-timer Benchmark Harness ===\n";
    std::cout << "Testing SAXPY kernel with Blocking vs NonBlocking sync modes\n";

    run_benchmark("saxpy_blocking", cuda_timer::SyncMode::Blocking, csv, json);
    run_benchmark("saxpy_nonblocking", cuda_timer::SyncMode::NonBlocking, csv, json);

    std::cout << "\n=== Done ===\n";
    std::cout << "CSV:  benchmark_results.csv\n";
    std::cout << "JSON: benchmark_results.json\n";

    return 0;
}