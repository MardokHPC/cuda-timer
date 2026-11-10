#pragma once

#include <cuda_runtime.h>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace cuda_timer {

/// One timed kernel execution, as written by the loggers.
struct TimingRecord {
    std::string label;
    float duration_ms = 0.0f;
    int device_id = 0;
    int stream_id = 0;
    uint64_t timestamp_ns = 0;   ///< steady_clock ticks when the measurement started
    int grid_x = 1, grid_y = 1, grid_z = 1;
    int block_x = 1, block_y = 1, block_z = 1;
    std::string kernel_name;
};

/**
 * @brief How benchmark() waits for each iteration to finish.
 *
 * Blocking syncs the whole device per launch; NonBlocking records the stop
 * event and polls it, so the host pays only for that event.
 */
enum class SyncMode { Blocking, NonBlocking };

/// Iteration counts, stream, sync mode and sample storage for benchmark().
struct BenchmarkConfig {
    size_t warmup_runs = 10;           ///< Discarded: module load, clock ramp-up
    size_t measure_runs = 100;
    cudaStream_t stream = 0;
    bool use_nvtx = false;             ///< One NVTX range per iteration
    SyncMode sync_mode = SyncMode::Blocking;
    bool store_samples = false;        ///< Needed by Stats::median()
};

/// CUDA API failure, keeping the raw cudaError_t.
class CudaError : public std::runtime_error {
    cudaError_t code_;
public:
    explicit CudaError(cudaError_t err, const char* file, int line)
        : std::runtime_error(std::string("CUDA error ") + std::to_string(static_cast<int>(err))
                           + " at " + file + ":" + std::to_string(line)
                           + ": " + cudaGetErrorString(err))
        , code_(err) {}

    cudaError_t code() const noexcept { return code_; }
};

/// Throws CudaError unless err is cudaSuccess; file and line come from
/// CUDA_TIMER_CHECK.
inline void check_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw CudaError(err, file, line);
    }
}

namespace detail {

/// RFC 4180 quoting: wraps a field that holds a comma, quote, CR or LF.
/// Shared so the loggers and write_samples() escape labels identically.
inline std::string escape_csv(std::string_view s) {
    if (s.find_first_of(",\"\r\n") == std::string_view::npos) {
        return std::string(s);
    }
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

} // namespace detail

// Captures __FILE__/__LINE__ at the call site.
#define CUDA_TIMER_CHECK(err) cuda_timer::check_cuda_error(err, __FILE__, __LINE__)

} // namespace cuda_timer