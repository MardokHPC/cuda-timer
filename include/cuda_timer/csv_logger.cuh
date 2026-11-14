#pragma once

#include "types.cuh"
#include <filesystem>
#include <fstream>
#include <locale>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace cuda_timer {

/**
 * @brief Thread-safe RFC 4180 CSV writer for TimingRecord rows.
 *
 * The stream keeps the classic locale for its whole lifetime, so a decimal
 * comma in the program's global locale cannot split a value across columns.
 *
 * Opened with std::ios::trunc by default, which discards previous content.
 * With std::ios::app the header is written only when the file is new or empty.
 */
class CsvLogger {
    std::ofstream file_;
    std::mutex mutex_;
    bool header_written_ = false;

public:
    /// @throws std::runtime_error if the file cannot be opened
    explicit CsvLogger(std::string_view filepath,
                       std::ios::openmode mode = std::ios::trunc)
        : file_(std::string(filepath), mode | std::ios::out) {
        if (!file_) {
            throw std::runtime_error("Cannot open CSV file: " + std::string(filepath));
        }
        file_.imbue(std::locale::classic());
        if ((mode & std::ios::app) != 0) {
            std::error_code ec;
            const auto size = std::filesystem::file_size(std::string(filepath), ec);
            header_written_ = !ec && size > 0;
        }
    }

    /// Flushes on destruction; never throws, since a throw here would terminate
    /// the process and lose the measurements.
    ~CsvLogger() noexcept {
        try {
            flush();
        } catch (...) {
        }
    }

    // Non-copyable, non-movable: owns a stream and a mutex.
    CsvLogger(const CsvLogger&) = delete;
    CsvLogger& operator=(const CsvLogger&) = delete;
    CsvLogger(CsvLogger&&) = delete;
    CsvLogger& operator=(CsvLogger&&) = delete;

    /// Writes the header on the first call unless appending to a non-empty file.
    void log(const TimingRecord& record) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!header_written_) {
            file_ << "label,duration_ms,device_id,stream_id,timestamp_ns,"
                  << "grid_x,grid_y,grid_z,block_x,block_y,block_z,kernel_name\n";
            header_written_ = true;
        }
        file_ << detail::escape_csv(record.label) << ','
              << record.duration_ms << ','
              << record.device_id << ','
              << record.stream_id << ','
              << record.timestamp_ns << ','
              << record.grid_x << ',' << record.grid_y << ',' << record.grid_z << ','
              << record.block_x << ',' << record.block_y << ',' << record.block_z << ','
              << detail::escape_csv(record.kernel_name) << '\n';
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) file_.flush();
    }

    /// Flushes and closes. On Windows the file cannot be read or removed before
    /// this; safe to call more than once.
    void close() {
        flush();
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) file_.close();
    }
};

} // namespace cuda_timer
