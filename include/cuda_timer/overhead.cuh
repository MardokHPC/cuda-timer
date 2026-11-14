#pragma once

#include "timer.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cuda_timer {

/**
 * @brief How small a duration this machine can measure at all.
 *
 * Timing back-to-back events with no work between them gives the cost of the
 * measurement itself: at or below median_ms a duration is indistinguishable
 * from overhead.
 */
struct OverheadFloor {
    std::size_t runs = 0;
    double min_ms = 0.0;
    double median_ms = 0.0;  ///< Upper middle element for even counts
    double mean_ms = 0.0;
};

/**
 * @brief Times empty start/stop event pairs, so no kernel cost enters the result.
 *
 * @throws std::invalid_argument if runs == 0
 * @throws CudaError on CUDA failure
 */
inline OverheadFloor measure_event_overhead(std::size_t runs = 100, cudaStream_t stream = 0) {
    if (runs == 0) {
        throw std::invalid_argument("measure_event_overhead: runs must be >= 1");
    }

    Timer timer(stream);
    std::vector<double> samples;
    samples.reserve(runs);
    for (std::size_t i = 0; i < runs; ++i) {
        timer.start();
        samples.push_back(timer.stop());
    }

    std::sort(samples.begin(), samples.end());

    OverheadFloor floor;
    floor.runs = runs;
    floor.min_ms = samples.front();
    floor.median_ms = samples[samples.size() / 2];
    double sum = 0.0;
    for (double value : samples) sum += value;
    floor.mean_ms = sum / static_cast<double>(samples.size());
    return floor;
}

} // namespace cuda_timer
