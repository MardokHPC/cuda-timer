#pragma once

#include "types.cuh"

#include <cstdint>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef NVTX_ENABLED
// MUST stay at global scope. A system or toolkit header pulled inside a
// namespace takes its declarations (and glibc's, transitively) with it while
// the include guards still report the headers as done, so every later
// <unistd.h>/<sys/stat.h> consumer fails on "identifier rmdir is undefined" /
// "incomplete class type stat". GoogleTest is the usual victim.
#include <nvtx3/nvToolsExt.h>
#endif

namespace cuda_timer::nvtx {

#ifdef NVTX_ENABLED

// NVTX is loaded at runtime on purpose: the toolkit ships the headers but no
// import library, so linking the NVTX C API would fail. LoadLibrary/dlopen the
// injection library instead; without it (no profiler, no driver support) every
// call below is a no-op.

namespace detail {

using push_ex_fn = int (*)(const nvtxEventAttributes_t*);
using pop_fn = int (*)();

struct NvtxLibrary {
#ifdef _WIN32
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
    push_ex_fn push_ex = nullptr;
    pop_fn pop = nullptr;

    NvtxLibrary() {
#ifdef _WIN32
        handle = ::LoadLibraryA("nvToolsExt64_1.dll");
        if (handle) {
            push_ex = reinterpret_cast<push_ex_fn>(
                ::GetProcAddress(handle, "nvtxRangePushEx"));
            pop = reinterpret_cast<pop_fn>(::GetProcAddress(handle, "nvtxRangePop"));
        }
#else
        handle = ::dlopen("libnvToolsExt.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            handle = ::dlopen("libnvToolsExt.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (handle) {
            push_ex = reinterpret_cast<push_ex_fn>(::dlsym(handle, "nvtxRangePushEx"));
            pop = reinterpret_cast<pop_fn>(::dlsym(handle, "nvtxRangePop"));
        }
#endif
    }

    // Intentionally leaked (Meyers singleton): ranges may still pop during
    // static destruction.
    static const NvtxLibrary& instance() {
        static const NvtxLibrary lib;
        return lib;
    }

    bool available() const noexcept { return push_ex != nullptr && pop != nullptr; }
};

} // namespace detail

/**
 * @brief Pushes an NVTX range; no-op when the NVTX library is absent.
 *
 * @param name ASCII, must outlive the matching pop_range()
 */
inline void push_range(const char* name, uint32_t color = 0xFF0000FF) noexcept {
    const auto& lib = detail::NvtxLibrary::instance();
    if (!lib.available() || name == nullptr) {
        return;
    }
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    lib.push_ex(&attr);
}

/// Pops the innermost range; no-op when NVTX is absent.
inline void pop_range() noexcept {
    const auto& lib = detail::NvtxLibrary::instance();
    if (lib.available()) {
        lib.pop();
    }
}

/**
 * @brief RAII range: pushed by the constructor, popped by the destructor.
 */
class ScopedRange {
public:
    explicit ScopedRange(const char* name, uint32_t color = 0xFF0000FF) noexcept {
        push_range(name, color);
    }

    ~ScopedRange() noexcept { pop_range(); }
    ScopedRange(ScopedRange&&) noexcept = default;
    ScopedRange& operator=(ScopedRange&&) noexcept = default;
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
};

#else // NVTX_ENABLED

// Same API with empty bodies, so callers compile unchanged without the toolkit.
inline void push_range(const char*, uint32_t = 0) noexcept {}

inline void pop_range() noexcept {}

class ScopedRange {
public:
    explicit ScopedRange(const char*, uint32_t = 0) noexcept {}
    ~ScopedRange() noexcept = default;
    ScopedRange(ScopedRange&&) noexcept = default;
    ScopedRange& operator=(ScopedRange&&) noexcept = default;
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
};

#endif

} // namespace cuda_timer::nvtx
