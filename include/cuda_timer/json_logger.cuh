#pragma once

#include "types.cuh"
#include <cstdio>
#include <fstream>
#include <locale>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cuda_timer {

/**
 * @brief Thread-safe streaming JSON array logger.
 *
 * '[' is written at construction, each log() appends one object, close() writes
 * ']'. The file is a valid array only after close(): before that it holds a
 * valid prefix, which survives a crash and can be salvaged by appending ']'.
 * flush() only pushes buffered bytes to disk, so it never has to finalize.
 *
 * Always truncated on open and never appended to: two arrays cannot be
 * concatenated into one document. Numbers use the classic locale, so the
 * output stays valid JSON whatever the program's global locale is.
 */
class JsonLogger {
    std::ofstream file_;
    std::mutex mutex_;
    bool pretty_ = true;
    bool first_record_ = true;
    bool closed_ = false;

    static std::string escape_json(const std::string& s) {
        std::string escaped;
        escaped.reserve(s.size() + 4);
        for (char c : s) {
            switch (c) {
                case '"':  escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\b': escaped += "\\b";  break;
                case '\f': escaped += "\\f";  break;
                case '\n': escaped += "\\n";  break;
                case '\r': escaped += "\\r";  break;
                case '\t': escaped += "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[7];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        escaped += buf;
                    } else {
                        escaped += c;
                    }
            }
        }
        return escaped;
    }

public:
    /// @throws std::runtime_error if the file cannot be opened
    explicit JsonLogger(std::string_view filepath, bool pretty = true)
        : file_(std::string(filepath), std::ios::trunc | std::ios::out),
          pretty_(pretty) {
        if (!file_) {
            throw std::runtime_error("Cannot open JSON file: " + std::string(filepath));
        }
        file_.imbue(std::locale::classic());
        file_ << '[';
    }

    /// Closes the array and flushes on destruction; never throws, since a throw
    /// here would terminate the process and lose the whole run.
    ~JsonLogger() noexcept {
        try {
            close();
        } catch (...) {
        }
    }

    // Non-copyable, non-movable: owns a stream and a mutex.
    JsonLogger(const JsonLogger&) = delete;
    JsonLogger& operator=(const JsonLogger&) = delete;
    JsonLogger(JsonLogger&&) = delete;
    JsonLogger& operator=(JsonLogger&&) = delete;

    /// @throws std::runtime_error when called after close()
    void log(const TimingRecord& record) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            throw std::runtime_error("JsonLogger::log() called after close()");
        }
        if (!first_record_) file_ << ',';
        if (pretty_) file_ << "\n  ";
        file_ << "{"
              << "\"label\":\"" << escape_json(record.label) << "\","
              << "\"duration_ms\":" << record.duration_ms << ","
              << "\"device_id\":" << record.device_id << ","
              << "\"stream_id\":" << record.stream_id << ","
              << "\"timestamp_ns\":" << record.timestamp_ns << ","
              << "\"grid\":[" << record.grid_x << "," << record.grid_y << ","
              << record.grid_z << "],"
              << "\"block\":[" << record.block_x << "," << record.block_y << ","
              << record.block_z << "],"
              << "\"kernel_name\":\"" << escape_json(record.kernel_name) << "\""
              << "}";
        first_record_ = false;
    }

    /// Pushes buffered bytes to disk; never writes the closing ']'.
    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) file_.flush();
    }

    /// Writes ']', flushes and closes. Required on Windows before the file can be
    /// read or removed; safe to call more than once.
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        if (file_.is_open()) {
            if (pretty_ && !first_record_) file_ << '\n';
            file_ << ']';
            file_.flush();
            file_.close();
        }
        closed_ = true;
    }
};

} // namespace cuda_timer
