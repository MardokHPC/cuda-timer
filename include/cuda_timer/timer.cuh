#pragma once

#include "types.cuh"
#include <memory>

namespace cuda_timer {
namespace detail {

// cudaEvent_t is a handle, not a heap pointer: unique_ptr needs a deleter.
struct EventDeleter {
    void operator()(cudaEvent_t* ev) const noexcept {
        if (ev && *ev) {
            cudaEventDestroy(*ev);
        }
    }
};

using EventPtr = std::unique_ptr<cudaEvent_t, EventDeleter>;

/// @throws CudaError if event creation fails
inline EventPtr make_event(unsigned int flags = cudaEventDefault) {
    cudaEvent_t ev;
    CUDA_TIMER_CHECK(cudaEventCreateWithFlags(&ev, flags));
    return EventPtr(new cudaEvent_t(ev), EventDeleter{});
}

} // namespace detail

/**
 * @brief RAII start/stop event pair for one stream.
 *
 * The device is captured at construction and re-selected before every
 * operation, so a Timer stays valid on a thread that later switches device.
 */
class Timer {
public:
    /// @throws CudaError if event creation or cudaGetDevice fails
    explicit Timer(cudaStream_t stream = 0, unsigned int flags = cudaEventDefault)
        : start_(detail::make_event(flags))
        , stop_(detail::make_event(flags))
        , stream_(stream) {
        CUDA_TIMER_CHECK(cudaGetDevice(&device_id_));
    }

    ~Timer() noexcept = default;
    Timer(Timer&&) noexcept = default;
    Timer& operator=(Timer&&) noexcept = default;
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    void start() {
        CUDA_TIMER_CHECK(cudaSetDevice(device_id_));
        CUDA_TIMER_CHECK(cudaEventRecord(*start_, stream_));
    }

    /// Records the stop event and blocks on it: this is SyncMode::Blocking.
    float stop() {
        CUDA_TIMER_CHECK(cudaSetDevice(device_id_));
        CUDA_TIMER_CHECK(cudaEventRecord(*stop_, stream_));
        CUDA_TIMER_CHECK(cudaEventSynchronize(*stop_));
        float ms = 0;
        CUDA_TIMER_CHECK(cudaEventElapsedTime(&ms, *start_, *stop_));
        return ms;
    }

    /// Non-blocking read: the stop event must already be recorded and complete.
    float elapsed() const {
        CUDA_TIMER_CHECK(cudaSetDevice(device_id_));
        float ms = 0;
        CUDA_TIMER_CHECK(cudaEventElapsedTime(&ms, *start_, *stop_));
        return ms;
    }

    /// nullptr on a moved-from Timer.
    cudaEvent_t event_start() const noexcept { return start_ ? *start_ : nullptr; }

    /// nullptr on a moved-from Timer.
    cudaEvent_t event_stop() const noexcept { return stop_ ? *stop_ : nullptr; }

    int device_id() const noexcept { return device_id_; }

    cudaStream_t stream() const noexcept { return stream_; }

private:
    detail::EventPtr start_;
    detail::EventPtr stop_;
    cudaStream_t stream_ = 0;
    int device_id_ = 0;
};

} // namespace cuda_timer