#pragma once

#include "types.cuh"
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

namespace cuda_timer {

/**
 * @brief Running mean/variance (Welford) plus min/max, O(1) per sample.
 *
 * Welford instead of sum-of-squares: timing samples sit far from zero compared
 * with their spread, and E[x^2] - E[x]^2 cancels badly there.
 */
struct Stats {
    size_t count_ = 0;
    double mean_ = 0.0;
    double m2_ = 0.0;                                       ///< Sum of squared deltas
    double min_ = std::numeric_limits<double>::infinity();
    double max_ = -std::numeric_limits<double>::infinity();
    std::vector<double> samples_;                           ///< Empty unless reserve() ran

    void add(double x) noexcept {
        ++count_;
        double delta = x - mean_;
        mean_ += delta / count_;
        double delta2 = x - mean_;
        m2_ += delta * delta2;
        if (x < min_) min_ = x;
        if (x > max_) max_ = x;
        if (!samples_.empty() || samples_.capacity() > 0) {
            samples_.push_back(x);
        }
    }

    /**
     * @brief Reserves room for n samples so add() starts storing them.
     *
     * Without a successful reserve() only the running aggregates are updated:
     * lower memory, but median() has nothing to sort.
     */
    void reserve(size_t n) { samples_.reserve(n); }

    size_t count() const noexcept { return count_; }

    /// Empty unless reserve() ran; feed it to write_samples() for histograms.
    const std::vector<double>& samples() const noexcept { return samples_; }

    double mean() const noexcept { return mean_; }

    /// Bessel-corrected sample variance; 0.0 below two samples.
    double variance() const noexcept { return count_ > 1 ? m2_ / (count_ - 1) : 0.0; }

    double stddev() const noexcept { return std::sqrt(variance()); }

    double min() const noexcept { return min_; }

    double max() const noexcept { return max_; }

    /**
     * @brief Median of the stored samples; 0.0 when none were kept.
     *
     * Sorts a copy, so the running state is untouched. Even counts average the
     * two middle elements.
     */
    double median() const {
        if (samples_.empty()) return 0.0;
        auto s = samples_;
        const size_t mid = s.size() / 2;
        std::nth_element(s.begin(), s.begin() + mid, s.end());
        if (s.size() % 2 == 0) {
            auto max_left = *std::max_element(s.begin(), s.begin() + mid);
            return (s[mid] + max_left) / 2.0;
        }
        return s[mid];
    }
};

} // namespace cuda_timer