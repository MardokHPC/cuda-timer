#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_timer {

/**
 * @brief Outcome of a baseline-vs-current performance comparison.
 */
enum class Verdict {
    Pass,        ///< Difference within the configured threshold
    Regression,  ///< Current is slower than baseline beyond the threshold
    Improvement, ///< Current is faster than baseline beyond the threshold
    Suspect      ///< Not enough samples, non-positive baseline, noisy data,
                 ///< incompatible environments, or a samples file that mixes
                 ///< series / labels
};

/// Stable lowercase name: it is part of the JSON and CI contract.
inline const char* to_string(Verdict verdict) noexcept {
    switch (verdict) {
        case Verdict::Pass:        return "pass";
        case Verdict::Regression:  return "regression";
        case Verdict::Improvement: return "improvement";
        case Verdict::Suspect:     return "suspect";
    }
    return "unknown";
}

/**
 * @brief Summary statistics of one series of samples (milliseconds).
 */
struct SeriesSummary {
    std::size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

/**
 * @brief Hardware/runtime identity of a measurement, from the .meta.json
 * sidecar.
 *
 * Two series from different devices, compute capabilities or CUDA runtimes are
 * not comparable: the gate reports `suspect` rather than a verdict.
 */
struct Environment {
    std::string device_name;
    std::string compute_capability;
    std::string driver_cuda_version;
    std::string runtime_cuda_version;
    int device_count = 0;
};

/**
 * @brief Thresholds for compare() / compare_runs().
 */
struct ComparisonOptions {
    /// Max relative regression on the median that still counts as Pass. The
    /// median, not the mean, is the decision statistic: one slow outlier
    /// (throttling, scheduler hiccup) must not fail a gate on its own.
    double threshold = 0.05;
    /// Below this many samples the verdict is Suspect (noise, not a result).
    std::size_t min_samples = 20;
    /// If > 0, a noisier current series (coefficient of variation above this)
    /// is Suspect: the run says nothing about the kernel.
    double max_cv = 0.0;
    /// If > 0, a spread between run medians above this is Suspect: the machine
    /// was not stable between runs. compare_runs() only.
    double max_run_spread = 0.02;
};

/**
 * @brief Summaries, deltas, verdict and environment of a comparison.
 */
struct ComparisonResult {
    SeriesSummary baseline;
    SeriesSummary current;
    double median_delta = 0.0;  ///< Relative delta on the median (0.05 = +5%)
    double p90_delta = 0.0;     ///< Relative delta on the 90th percentile
    Verdict verdict = Verdict::Suspect;
    std::string reason;

    /// 1 when compare() is used directly.
    std::size_t baseline_runs = 1;
    std::size_t current_runs = 1;
    double baseline_run_spread = 0.0;
    double current_run_spread = 0.0;

    /// Filled by the CLI when both sidecars are present.
    bool has_environment = false;
    bool environment_match = false;
    Environment baseline_environment;
    Environment current_environment;
};

inline SeriesSummary summarize(const std::vector<double>& samples);

namespace detail {

/**
 * @brief Parses a double in the classic "C" locale.
 *
 * std::stod follows the global locale, so with it/fr/de installed it rejects
 * "0.5" -- the format write_samples() emits.
 */
inline double parse_double(const std::string& text) {
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    double value = 0.0;
    stream >> value;
    if (stream.fail()) {
        throw std::runtime_error("invalid number: '" + text + "'");
    }
    stream >> std::ws;
    if (!stream.eof()) {
        throw std::runtime_error("trailing characters after number: '" + text + "'");
    }
    return value;
}

inline double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted.front();
    const double position = p * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const double fraction = position - static_cast<double>(lower);
    const std::size_t upper = (lower + 1 < sorted.size()) ? lower + 1 : lower;
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

inline std::string json_escape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() + 4);
    for (unsigned char c : text) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b";  break;
            case '\f': escaped += "\\f";  break;
            case '\n': escaped += "\\n";  break;
            case '\r': escaped += "\\r";  break;
            case '\t': escaped += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                                  static_cast<unsigned int>(c));
                    escaped += buffer;
                } else {
                    escaped += static_cast<char>(c);
                }
        }
    }
    return escaped;
}

inline std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

/**
 * @brief Extracts a flat field from the .meta.json schema.
 *
 * Not a JSON parser: it only has to read the sidecar written by
 * environment.cuh, and throws if the field is missing or malformed.
 */
inline std::string json_field(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing field in environment file: " + key);
    }
    const auto colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("malformed field in environment file: " + key);
    }

    std::size_t pos = colon + 1;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;

    if (pos < text.size() && text[pos] == '"') {
        ++pos;
        std::string value;
        while (pos < text.size()) {
            const char c = text[pos++];
            if (c == '\\' && pos < text.size()) {
                value += text[pos++];
                continue;
            }
            if (c == '"') return value;
            value += c;
        }
        throw std::runtime_error("unterminated string field in environment file: " + key);
    }

    const auto end = text.find_first_of(",\n}", pos);
    const std::string raw = (end == std::string::npos) ? text.substr(pos)
                                                       : text.substr(pos, end - pos);
    return trim(raw);
}

inline std::string summary_json(const SeriesSummary& s) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(6);
    out << "{\"count\": " << s.count
        << ", \"mean_ms\": " << s.mean
        << ", \"median_ms\": " << s.median
        << ", \"p90_ms\": " << s.p90
        << ", \"p99_ms\": " << s.p99
        << ", \"min_ms\": " << s.min
        << ", \"max_ms\": " << s.max
        << ", \"stddev_ms\": " << s.stddev << "}";
    return out.str();
}

inline std::string environment_inline_json(const Environment& e) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"device_name\": \"" << json_escape(e.device_name) << "\""
        << ", \"compute_capability\": \"" << json_escape(e.compute_capability) << "\""
        << ", \"driver_cuda_version\": \"" << json_escape(e.driver_cuda_version) << "\""
        << ", \"runtime_cuda_version\": \"" << json_escape(e.runtime_cuda_version) << "\""
        << ", \"device_count\": " << e.device_count << "}";
    return out.str();
}

inline std::vector<double> run_medians(const std::vector<std::vector<double>>& runs) {
    std::vector<double> medians;
    medians.reserve(runs.size());
    for (const auto& run : runs) {
        medians.push_back(summarize(run).median);
    }
    return medians;
}

inline std::vector<double> pool_runs(const std::vector<std::vector<double>>& runs) {
    std::vector<double> pooled;
    for (const auto& run : runs) {
        pooled.insert(pooled.end(), run.begin(), run.end());
    }
    return pooled;
}

inline double run_spread(const std::vector<double>& medians) {
    if (medians.size() < 2) return 0.0;
    const auto bounds = std::minmax_element(medians.begin(), medians.end());
    const double center = summarize(medians).median;
    if (!(center > 0.0)) return 0.0;
    return (*bounds.second - *bounds.first) / center;
}

} // namespace detail

/**
 * @brief count/mean/stddev/min/max/median/p90/p99 of one series.
 *
 * Percentiles interpolate linearly between the two closest order statistics,
 * which behaves the same for odd and even counts.
 */
inline SeriesSummary summarize(const std::vector<double>& samples) {
    SeriesSummary summary;
    if (samples.empty()) return summary;

    summary.count = samples.size();

    double sum = 0.0;
    for (double value : samples) sum += value;
    summary.mean = sum / static_cast<double>(samples.size());

    double m2 = 0.0;
    for (double value : samples) {
        const double delta = value - summary.mean;
        m2 += delta * delta;
    }
    summary.stddev = (samples.size() > 1)
        ? std::sqrt(m2 / static_cast<double>(samples.size() - 1))
        : 0.0;

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    summary.min = sorted.front();
    summary.max = sorted.back();
    summary.median = detail::percentile(sorted, 0.50);
    summary.p90 = detail::percentile(sorted, 0.90);
    summary.p99 = detail::percentile(sorted, 0.99);
    return summary;
}

/**
 * @brief Classifies the difference between a baseline and a current series.
 *
 * Decides on the median; p90 is reported for diagnostics only.
 */
inline ComparisonResult compare(const std::vector<double>& baseline,
                                const std::vector<double>& current,
                                const ComparisonOptions& opts = {}) {
    ComparisonResult result;
    result.baseline = summarize(baseline);
    result.current = summarize(current);

    if (result.baseline.count < opts.min_samples || result.current.count < opts.min_samples) {
        result.verdict = Verdict::Suspect;
        result.reason = "insufficient samples (min_samples=" +
                        std::to_string(opts.min_samples) + ")";
        return result;
    }
    if (!(result.baseline.median > 0.0)) {
        result.verdict = Verdict::Suspect;
        result.reason = "baseline median is not positive";
        return result;
    }

    result.median_delta = (result.current.median - result.baseline.median) / result.baseline.median;
    result.p90_delta = (result.baseline.p90 > 0.0)
        ? (result.current.p90 - result.baseline.p90) / result.baseline.p90
        : 0.0;

    const double cv = (result.current.mean > 0.0)
        ? result.current.stddev / result.current.mean
        : 0.0;
    if (opts.max_cv > 0.0 && cv > opts.max_cv) {
        result.verdict = Verdict::Suspect;
        result.reason = "current coefficient of variation too high";
        return result;
    }

    if (result.median_delta > opts.threshold) {
        result.verdict = Verdict::Regression;
        result.reason = "median regression above threshold";
    } else if (result.median_delta < -opts.threshold) {
        result.verdict = Verdict::Improvement;
        result.reason = "median improvement above threshold";
    } else {
        result.verdict = Verdict::Pass;
        result.reason = "within threshold";
    }
    return result;
}

/**
 * @brief Compares several runs per side.
 *
 * Decides on the median of the per-run medians while the descriptive
 * statistics come from the pooled samples of all runs, which is what keeps
 * min_samples and max_cv meaningful here.
 */
inline ComparisonResult compare_runs(
    const std::vector<std::vector<double>>& baseline_runs,
    const std::vector<std::vector<double>>& current_runs,
    const ComparisonOptions& opts = {}) {
    ComparisonResult result;

    if (baseline_runs.empty() || current_runs.empty()) {
        result.verdict = Verdict::Suspect;
        result.reason = "no runs provided";
        return result;
    }
    for (const auto& run : baseline_runs) {
        if (run.empty()) {
            result.verdict = Verdict::Suspect;
            result.reason = "empty baseline run";
            return result;
        }
    }
    for (const auto& run : current_runs) {
        if (run.empty()) {
            result.verdict = Verdict::Suspect;
            result.reason = "empty current run";
            return result;
        }
    }

    const auto baseline_medians = detail::run_medians(baseline_runs);
    const auto current_medians = detail::run_medians(current_runs);

    // Decide on the per-run medians; min_samples/max_cv are re-applied below
    // on the pooled samples, which is what they are documented to describe.
    ComparisonOptions inner = opts;
    inner.min_samples = 1;
    inner.max_cv = 0.0;
    inner.max_run_spread = 0.0;
    result = compare(baseline_medians, current_medians, inner);

    result.baseline = summarize(detail::pool_runs(baseline_runs));
    result.current = summarize(detail::pool_runs(current_runs));
    result.baseline_runs = baseline_runs.size();
    result.current_runs = current_runs.size();
    result.baseline_run_spread = detail::run_spread(baseline_medians);
    result.current_run_spread = detail::run_spread(current_medians);
    result.p90_delta = (result.baseline.p90 > 0.0)
        ? (result.current.p90 - result.baseline.p90) / result.baseline.p90
        : 0.0;

    if (result.baseline.count < opts.min_samples || result.current.count < opts.min_samples) {
        result.verdict = Verdict::Suspect;
        result.reason = "insufficient samples (min_samples=" +
                        std::to_string(opts.min_samples) + ")";
        return result;
    }
    if (!(result.baseline.median > 0.0)) {
        result.verdict = Verdict::Suspect;
        result.reason = "baseline median is not positive";
        return result;
    }
    if (opts.max_cv > 0.0) {
        const double cv = (result.current.mean > 0.0)
            ? result.current.stddev / result.current.mean
            : 0.0;
        if (cv > opts.max_cv) {
            result.verdict = Verdict::Suspect;
            result.reason = "current coefficient of variation too high";
            return result;
        }
    }
    const double worst_spread = std::max(result.baseline_run_spread,
                                         result.current_run_spread);
    if (opts.max_run_spread > 0.0 && worst_spread > opts.max_run_spread) {
        result.verdict = Verdict::Suspect;
        result.reason = "run-to-run variance too high";
        return result;
    }
    return result;
}

namespace detail {

struct SampleRow {
    std::string label;
    double value = 0.0;
};

/**
 * @brief First CSV field of a row, honouring RFC 4180 quoting.
 *
 * Labels are escaped by write_samples(), so they may hold commas and doubled
 * quotes. One containing CR/LF spans several physical lines: the row reader
 * rejects those instead of mis-parsing them.
 *
 * @throws std::runtime_error if the field is unterminated or the row has no
 *         separator
 */
inline std::string parse_csv_label(const std::string& line) {
    if (line.empty()) {
        throw std::runtime_error("empty row");
    }
    if (line.front() != '"') {
        const auto comma = line.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("missing separator in '" + line + "'");
        }
        return line.substr(0, comma);
    }
    std::string label;
    for (std::size_t i = 1; i < line.size(); ++i) {
        if (line[i] == '"') {
            if (i + 1 < line.size() && line[i + 1] == '"') {
                label += '"';
                ++i;
                continue;
            }
            return label;
        }
        label += line[i];
    }
    throw std::runtime_error("unterminated quoted field in '" + line + "'");
}

inline std::string join_labels(const std::vector<std::string>& labels) {
    std::string joined;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (i != 0) joined += ", ";
        joined += labels[i];
    }
    return joined;
}

/**
 * @brief Every row of a samples CSV produced by write_samples().
 *
 * @throws std::runtime_error if the file is missing, empty or malformed
 */
inline std::vector<SampleRow> read_sample_rows(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open samples file: " + path);
    }

    std::vector<SampleRow> rows;
    std::string line;
    std::size_t line_number = 0;
    bool first_content_line = true;

    while (std::getline(file, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (first_content_line && line.rfind("label,", 0) == 0) {
            first_content_line = false;
            continue;
        }
        first_content_line = false;

        const auto comma = line.rfind(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("malformed row at line " + std::to_string(line_number) +
                                     " in " + path);
        }
        try {
            SampleRow row;
            row.label = parse_csv_label(line);
            row.value = parse_double(line.substr(comma + 1));
            rows.push_back(row);
        } catch (const std::exception& e) {
            throw std::runtime_error("line " + std::to_string(line_number) + " in " + path +
                                     ": " + e.what());
        }
    }

    if (rows.empty()) {
        throw std::runtime_error("no samples in " + path);
    }
    return rows;
}

/**
 * @brief One samples file loaded as a single series.
 *
 * `reason` stays empty when the file holds exactly one label; otherwise it
 * says why the file cannot be used as a series.
 */
struct SeriesFile {
    std::string label;
    std::vector<double> values;
    std::string reason;
};

inline SeriesFile load_series_file(const std::string& path, const char* side) {
    SeriesFile file;
    const auto rows = read_sample_rows(path);

    std::vector<std::string> labels;
    file.values.reserve(rows.size());
    for (const auto& row : rows) {
        file.values.push_back(row.value);
        if (std::find(labels.begin(), labels.end(), row.label) == labels.end()) {
            labels.push_back(row.label);
        }
    }

    if (labels.size() != 1) {
        file.reason = std::string(side) + " holds " + std::to_string(labels.size()) +
                      " labels (" + join_labels(labels) + "); one series per file";
        return file;
    }
    file.label = labels.front();
    return file;
}

} // namespace detail

/**
 * @brief All values of a samples CSV written by write_samples(), any label.
 *
 * Format: an optional `label,index,value` header, then one row per sample.
 * Parsing stays in the classic locale, so a decimal comma set globally by the
 * host program cannot break it.
 *
 * @throws std::runtime_error if the file is missing, empty or malformed
 */
inline std::vector<double> read_samples_csv(const std::string& path) {
    const auto rows = detail::read_sample_rows(path);
    std::vector<double> samples;
    samples.reserve(rows.size());
    for (const auto& row : rows) samples.push_back(row.value);
    return samples;
}

/**
 * @brief Values of one series only, for files that may hold several labels.
 *
 * The label match is case sensitive. Use this after
 * write_samples(..., std::ios::app) appended a second kernel to the file.
 *
 * @throws std::runtime_error if the file is malformed or no row has that label
 */
inline std::vector<double> read_samples_csv(const std::string& path,
                                            const std::string& label) {
    const auto rows = detail::read_sample_rows(path);
    std::vector<double> samples;
    for (const auto& row : rows) {
        if (row.label == label) samples.push_back(row.value);
    }
    if (samples.empty()) {
        throw std::runtime_error("label '" + label + "' not found in " + path);
    }
    return samples;
}

/**
 * @brief Distinct labels of a samples CSV, in order of first appearance.
 *
 * @throws std::runtime_error if the file is missing, empty or malformed
 */
inline std::vector<std::string> labels_in_csv(const std::string& path) {
    const auto rows = detail::read_sample_rows(path);
    std::vector<std::string> labels;
    for (const auto& row : rows) {
        if (std::find(labels.begin(), labels.end(), row.label) == labels.end()) {
            labels.push_back(row.label);
        }
    }
    return labels;
}

/**
 * @brief Compares two samples CSV files.
 *
 * Suspect, not a verdict, when a file holds several series or the labels
 * differ: write_samples(..., std::ios::app) can append a second kernel to one
 * file, and pooling those rows would grade neither kernel.
 */
inline ComparisonResult compare_files(const std::string& baseline_csv,
                                      const std::string& current_csv,
                                      const ComparisonOptions& opts = {}) {
    ComparisonResult result;

    const detail::SeriesFile baseline = detail::load_series_file(baseline_csv, "baseline");
    if (!baseline.reason.empty()) {
        result.verdict = Verdict::Suspect;
        result.reason = baseline.reason;
        return result;
    }

    const detail::SeriesFile current = detail::load_series_file(current_csv, "current");
    if (!current.reason.empty()) {
        result.verdict = Verdict::Suspect;
        result.reason = current.reason;
        return result;
    }

    if (baseline.label != current.label) {
        result.verdict = Verdict::Suspect;
        result.reason = "label mismatch: baseline '" + baseline.label + "' vs current '" +
                        current.label + "'";
        return result;
    }

    return compare(baseline.values, current.values, opts);
}

/**
 * @brief Reads the sidecar written by write_environment().
 *
 * @throws std::runtime_error if the file is missing or a required field is not
 *         present
 */
inline Environment read_environment(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open environment file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();

    Environment env;
    env.device_name = detail::json_field(text, "device_name");
    env.compute_capability = detail::json_field(text, "compute_capability");
    env.driver_cuda_version = detail::json_field(text, "driver_cuda_version");
    env.runtime_cuda_version = detail::json_field(text, "runtime_cuda_version");
    const double device_count = detail::parse_double(
        detail::json_field(text, "device_count"));
    if (device_count < 0.0 || device_count != std::floor(device_count)) {
        throw std::runtime_error("invalid device_count in environment file: " + path);
    }
    env.device_count = static_cast<int>(device_count);
    return env;
}

inline std::string environment_json(const Environment& env) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\n"
        << "  \"device_name\": \"" << detail::json_escape(env.device_name) << "\",\n"
        << "  \"compute_capability\": \"" << detail::json_escape(env.compute_capability) << "\",\n"
        << "  \"driver_cuda_version\": \"" << detail::json_escape(env.driver_cuda_version) << "\",\n"
        << "  \"runtime_cuda_version\": \"" << detail::json_escape(env.runtime_cuda_version) << "\",\n"
        << "  \"device_count\": " << env.device_count << "\n"
        << "}\n";
    return out.str();
}

/**
 * @brief Whether two environments are comparable.
 *
 * Name, compute capability and CUDA versions must match. Device count is
 * ignored on purpose: a machine can gain or lose a GPU without invalidating a
 * measurement of a specific one.
 *
 * @param reason If non-null, receives the first differing field name.
 */
inline bool environment_compatible(const Environment& a, const Environment& b,
                                   std::string* reason = nullptr) {
    auto mismatch = [reason](const char* field) {
        if (reason) *reason = std::string("environment mismatch on ") + field;
        return false;
    };
    if (a.device_name != b.device_name) return mismatch("device_name");
    if (a.compute_capability != b.compute_capability) return mismatch("compute_capability");
    if (a.driver_cuda_version != b.driver_cuda_version) return mismatch("driver_cuda_version");
    if (a.runtime_cuda_version != b.runtime_cuda_version) return mismatch("runtime_cuda_version");
    if (reason) reason->clear();
    return true;
}

/**
 * @brief Comparison result as stable, human-readable JSON.
 *
 * Numbers are formatted in the classic "C" locale, so the document is valid
 * JSON whatever the host program's global locale is.
 */
inline std::string to_json(const ComparisonResult& result) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"verdict\": \"" << to_string(result.verdict) << "\",\n"
        << "  \"reason\": \"" << detail::json_escape(result.reason) << "\",\n"
        << "  \"median_delta_pct\": " << result.median_delta * 100.0 << ",\n"
        << "  \"p90_delta_pct\": " << result.p90_delta * 100.0 << ",\n"
        << "  \"runs\": {\"baseline\": " << result.baseline_runs
        << ", \"current\": " << result.current_runs << "},\n"
        << "  \"run_spread_pct\": {\"baseline\": " << result.baseline_run_spread * 100.0
        << ", \"current\": " << result.current_run_spread * 100.0 << "},\n"
        << "  \"baseline\": " << detail::summary_json(result.baseline) << ",\n"
        << "  \"current\": " << detail::summary_json(result.current) << ",\n"
        << "  \"environment\": {"
        << "\"present\": " << (result.has_environment ? "true" : "false")
        << ", \"match\": " << (result.environment_match ? "true" : "false");
    if (result.has_environment) {
        out << ", \"baseline\": " << detail::environment_inline_json(result.baseline_environment)
            << ", \"current\": " << detail::environment_inline_json(result.current_environment);
    }
    out << "}\n"
        << "}\n";
    return out.str();
}

} // namespace cuda_timer
