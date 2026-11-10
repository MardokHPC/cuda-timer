#pragma once

#include "types.cuh"
#include "stats.cuh"

#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace cuda_timer {

/**
 * @brief Writes one CSV row per sample: label,index,value.
 *
 * Keeps every measurement at full double precision, so consumers can build
 * histograms that a mean would hide. Values use the classic locale, so a
 * decimal comma cannot split the columns. Truncated by default; with
 * std::ios::app the header is written only when the file is new or empty.
 *
 * @throws std::runtime_error if the file cannot be opened
 */
inline void write_samples(std::string_view path,
                          std::string_view label,
                          const std::vector<double>& samples,
                          std::ios::openmode mode = std::ios::trunc) {
    std::ofstream file{std::string(path), mode | std::ios::out};
    if (!file) {
        throw std::runtime_error("Cannot open samples file: " + std::string(path));
    }
    file.imbue(std::locale::classic());
    file.precision(std::numeric_limits<double>::max_digits10);

    bool write_header = true;
    if ((mode & std::ios::app) != 0) {
        std::error_code ec;
        const auto size = std::filesystem::file_size(std::string(path), ec);
        write_header = ec || size == 0;
    }

    const std::string escaped = detail::escape_csv(label);
    if (write_header) {
        file << "label,index,value\n";
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        file << escaped << ',' << i << ',' << samples[i] << '\n';
    }
}

/**
 * @brief Exports the samples held by a Stats object.
 *
 * Needs Stats::reserve() to have run before the samples were added (e.g.
 * BenchmarkConfig::store_samples); otherwise only the header is written.
 *
 * @throws std::runtime_error if the file cannot be opened
 */
inline void write_samples(std::string_view path,
                          std::string_view label,
                          const Stats& stats,
                          std::ios::openmode mode = std::ios::trunc) {
    write_samples(path, label, stats.samples(), mode);
}

} // namespace cuda_timer
