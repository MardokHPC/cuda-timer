#pragma once

#include "compare.cuh"
#include "types.cuh"

#include <fstream>
#include <locale>
#include <stdexcept>
#include <string>

namespace cuda_timer {
namespace detail {

/// 12010 -> "12.1".
inline std::string format_cuda_version(int version) {
    if (version <= 0) return "unknown";
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

} // namespace detail

/**
 * @brief What a measurement ran on: device, compute capability, CUDA versions.
 *
 * Runtime API only, no NVML: no extra dependency, and it works on any driver
 * new enough to run the binary.
 */
inline Environment capture_environment(int device_id = 0) {
    int device_count = 0;
    CUDA_TIMER_CHECK(cudaGetDeviceCount(&device_count));

    cudaDeviceProp properties{};
    CUDA_TIMER_CHECK(cudaGetDeviceProperties(&properties, device_id));

    int driver_version = 0;
    int runtime_version = 0;
    CUDA_TIMER_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_TIMER_CHECK(cudaRuntimeGetVersion(&runtime_version));

    Environment environment;
    environment.device_name = properties.name;
    environment.compute_capability = std::to_string(properties.major) + "." +
                                     std::to_string(properties.minor);
    environment.driver_cuda_version = detail::format_cuda_version(driver_version);
    environment.runtime_cuda_version = detail::format_cuda_version(runtime_version);
    environment.device_count = device_count;
    return environment;
}

/**
 * @brief Writes the .meta.json sidecar describing how a samples file was measured.
 *
 * perf_gate looks for it by swapping the .csv extension for .meta.json.
 *
 * @throws std::runtime_error if the file cannot be written, CudaError on CUDA
 *         failure
 */
inline void write_environment(const std::string& path, int device_id = 0) {
    const Environment environment = capture_environment(device_id);
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open environment file: " + path);
    }
    file.imbue(std::locale::classic());
    file << environment_json(environment);
    if (!file) {
        throw std::runtime_error("Failed while writing environment file: " + path);
    }
}

} // namespace cuda_timer
