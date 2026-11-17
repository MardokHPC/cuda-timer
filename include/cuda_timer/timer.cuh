#pragma once

#include "types.cuh"
#include <memory>

namespace cuda_timer {
namespace detail {

// cudaEvent_t is a handle, not a heap cell: the unique_ptr stores the handle
// itself and the deleter only has to destroy it. Wrapping the handle in `new`
// would leak one allocation per event.
struct EventDeleter {
    using pointer = cudaEvent_t;

    void operator()(pointer ev) const noexcept {
        if (ev) {
            cudaEventDestroy(ev);
        }
    }
};

using EventPtr = std::unique_ptr<cudaEvent_t, EventDeleter>;

/// @throws CudaError if event creation fails
inline EventPtr make_event(unsigned int flags = cudaEventDefault) {
    cudaEvent_t ev;
    CUDA_TIMER_CHECK(cudaEventCreateWithFlags(&ev, flags));
    return EventPtr(ev);
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
        CUDA_TIMER_CHECK(cudaEventRecord(start_.get(), stream_));
    }

    /// Records the stop event and blocks on it: this is SyncMode::Blocking.
    float stop() {
        CUDA_TIMER_CHECK(cudaSetDevice(device_id_));
        CUDA_TIMER_CHECK(cudaEventRecord(stop_.get(), stream_));
        CUDA_TIMER_CHECK(cudaEventSynchronize(stop_.get()));
        float ms = 0;
        CUDA_TIMER_CHECK(cudaEventElapsedTime(&ms, start_.get(), stop_.get()));
        return ms;
    }

    /// Non-blocking read: the stop event must already be recorded and complete.
    float elapsed() const {
        CUDA_TIMER_CHECK(cudaSetDevice(device_id_));
        float ms = 0;
        CUDA_TIMER_CHECK(cudaEventElapsedTime(&ms, start_.get(), stop_.get()));
        return ms;
    }

    /// nullptr on a moved-from Timer.
    cudaEvent_t event_start() const noexcept { return start_.get(); }

    /// nullptr on a moved-from Timer.
    cudaEvent_t event_stop() const noexcept { return stop_.get(); }

    int device_id() const noexcept { return device_id_; }

    cudaStream_t stream() const noexcept { return stream_; }

private:
    detail::EventPtr start_;
    detail::EventPtr stop_;
    cudaStream_t stream_ = 0;
    int device_id_ = 0;
};

} // namespace cuda_timer