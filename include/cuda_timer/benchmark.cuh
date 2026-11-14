#pragma once

#include "types.cuh"
#include "timer.cuh"
#include "stats.cuh"
#include "nvtx.cuh"
#include <functional>
#include <chrono>

namespace cuda_timer {

/**
 * @brief Times measure_runs() calls of kernel, discarding warmup_runs first.
 *
 * The first launches pay for module load and for the GPU clock ramping up;
 * counting them would inflate mean and stddev on a kernel that is fine.
 */
template <typename Fn>
Stats benchmark(Fn&& kernel, const BenchmarkConfig& cfg = {}) {
    for (size_t i = 0; i < cfg.warmup_runs; ++i) {
        if (cfg.use_nvtx) nvtx::push_range("warmup");
        kernel();
        if (cfg.use_nvtx) nvtx::pop_range();
        if (cfg.sync_mode == SyncMode::Blocking) {
            CUDA_TIMER_CHECK(cudaDeviceSynchronize());
        }
    }
    if (cfg.sync_mode == SyncMode::Blocking) {
        CUDA_TIMER_CHECK(cudaDeviceSynchronize());
    }

    Stats stats;
    if (cfg.store_samples) stats.reserve(cfg.measure_runs);
    Timer timer(cfg.stream);

    for (size_t i = 0; i < cfg.measure_runs; ++i) {
        if (cfg.use_nvtx) nvtx::push_range("measure");

        timer.start();
        kernel();

        if (cfg.sync_mode == SyncMode::Blocking) {
            stats.add(timer.stop());
        } else {
            cudaSetDevice(timer.device_id());
            cudaEventRecord(timer.event_stop(), cfg.stream);
            cudaError_t err;
            do {
                err = cudaEventQuery(timer.event_stop());
            } while (err == cudaErrorNotReady);
            CUDA_TIMER_CHECK(err);
            float ms = 0.0f;
            CUDA_TIMER_CHECK(cudaEventElapsedTime(&ms, timer.event_start(), timer.event_stop()));
            stats.add(ms);
        }

        if (cfg.use_nvtx) nvtx::pop_range();
    }
    return stats;
}

/**
 * @brief Adapts a launch(grid, block) callable to benchmark().
 *
 * kernel_name is kept for source compatibility and currently ignored: the
 * returned Stats carries no kernel identity.
 */
template <typename Fn>
Stats benchmark_kernel(Fn&& launch, const BenchmarkConfig& cfg,
                       dim3 grid, dim3 block, const char* kernel_name = nullptr) {
    auto wrapped = [&]() { launch(grid, block); };
    (void)kernel_name;
    return benchmark(wrapped, cfg);
}

} // namespace cuda_timer