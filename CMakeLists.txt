cmake_minimum_required(VERSION 3.18)

project(cuda_timer
    LANGUAGES CUDA CXX
    VERSION 1.1.0
    DESCRIPTION "Header-only CUDA timing/profiling library for GPU kernels"
    HOMEPAGE_URL "https://github.com/MardokHPC/cuda-timer"
)

option(CUDA_TIMER_NVTX "Enable NVTX (NVIDIA Tools Extension) support" ON)
option(CUDA_TIMER_BUILD_EXAMPLES "Build example executables" ON)
option(CUDA_TIMER_BUILD_TESTS "Build unit tests (requires GoogleTest)" OFF)
option(CUDA_TIMER_BUILD_TOOLS "Build command-line tools (perf_gate)" ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Provides CUDA::cudart and the NVTX headers.
find_package(CUDAToolkit REQUIRED)

function(cuda_timer_apply_warnings target)
    # Host C++: direct flags, CXX only, so they never reach nvcc raw.
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:
            -Wall -Wextra -Wpedantic -Werror
            -Wno-unused-parameter
            -Wno-missing-field-initializers
        >
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:
            /W4 /WX
            /wd4100   # unreferenced formal parameter
            /wd4244   # conversion from 'double' to 'float', possible loss of data
            /wd4267   # conversion from 'size_t' to 'int', possible loss of data
        >
    )
    # CUDA: host flags MUST go through -Xcompiler. nvcc has its own -Werror and
    # swallows the next flag as its value ("Value '-Wno-unused-parameter' is not
    # defined for option 'Werror'"); "/W4" is read as an input file instead.
    # -Wpedantic stays out: nvcc's generated cudafe1.cpp emits a GCC line-directive
    # extension, so -Wpedantic -Werror fails on every .cu.
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:
            SHELL:-Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Werror
            SHELL:-Xcompiler=-Wno-unused-parameter -Xcompiler=-Wno-missing-field-initializers
        >
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:
            SHELL:-Xcompiler=/W4 -Xcompiler=/WX -Xcompiler=/wd4100 -Xcompiler=/wd4244 -Xcompiler=/wd4267 -Xcompiler=/wd4505
        >
    )
endfunction()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Header-only INTERFACE target: same name whether consumed via
# add_subdirectory/FetchContent or via find_package().
add_library(cuda_timer INTERFACE)
add_library(cuda_timer::cuda_timer ALIAS cuda_timer)

target_include_directories(cuda_timer INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cuda_timer INTERFACE cxx_std_17)

# NVTX ranges are compiled in by default; the runtime is dlopen'ed (nvtx.cuh),
# so there is nothing to link.
target_compile_definitions(cuda_timer INTERFACE
    $<$<BOOL:${CUDA_TIMER_NVTX}>:NVTX_ENABLED>
)

install(DIRECTORY include/cuda_timer
    DESTINATION include
    FILES_MATCHING PATTERN "*.cuh"
)

install(TARGETS cuda_timer
    EXPORT cuda_timer-targets
)

install(EXPORT cuda_timer-targets
    FILE cuda_timer-targets.cmake
    NAMESPACE cuda_timer::
    DESTINATION lib/cmake/cuda_timer
)

if(CUDA_TIMER_NVTX)
    add_compile_definitions(NVTX_ENABLED)
endif()

function(cuda_timer_add_example name source)
    if(NOT CUDA_TIMER_BUILD_EXAMPLES)
        return()
    endif()
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE
        cuda_timer::cuda_timer
        CUDA::cudart
    )
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    cuda_timer_apply_warnings(${name})
    if(CUDA_TIMER_NVTX)
        target_compile_definitions(${name} PRIVATE NVTX_ENABLED)
    endif()
    install(TARGETS ${name} DESTINATION bin)
endfunction()

if(CUDA_TIMER_BUILD_EXAMPLES)
    cuda_timer_add_example(basic_usage        examples/basic_usage.cu)
    cuda_timer_add_example(multi_stream       examples/multi_stream.cu)
    cuda_timer_add_example(multi_device       examples/multi_device.cu)
    cuda_timer_add_example(benchmark_harness  examples/benchmark_harness.cu)
    cuda_timer_add_example(overhead_floor     examples/overhead_floor.cu)
    cuda_timer_add_example(environment_info   examples/environment_info.cu)
endif()

# Host-only tool: perf_gate links the headers but calls no CUDA API.
if(CUDA_TIMER_BUILD_TOOLS)
    add_executable(perf_gate tools/perf_gate.cpp)
    target_link_libraries(perf_gate PRIVATE cuda_timer::cuda_timer)
    set_target_properties(perf_gate PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    cuda_timer_apply_warnings(perf_gate)
    install(TARGETS perf_gate DESTINATION bin)
endif()

if(CUDA_TIMER_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        # No system GTest: fetch it (needs network at configure time).
        if(POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/v1.14.0.zip
        )
        # Keep GoogleTest on the parent project's CRT and out of our installs.
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        # FetchContent yields gtest/gtest_main: normalize to the FindGTest names.
        if(NOT TARGET GTest::GTest)
            add_library(GTest::GTest ALIAS gtest)
        endif()
        if(NOT TARGET GTest::Main)
            add_library(GTest::Main ALIAS gtest_main)
        endif()
    endif()

    function(cuda_timer_configure_test name source main_file)
        add_executable(${name} ${source} ${main_file})
        target_link_libraries(${name} PRIVATE
            GTest::GTest
            cuda_timer::cuda_timer
            CUDA::cudart
        )
        set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        cuda_timer_apply_warnings(${name})
        if(CUDA_TIMER_NVTX)
            target_compile_definitions(${name} PRIVATE NVTX_ENABLED)
        endif()
        add_test(NAME ${name} COMMAND ${name})
        # 77 from tests/test_main_device.cpp means "no CUDA device": skipped,
        # not failed and not a silent pass.
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    # Host-only suite: never checks for a device, runs on GPU-less runners.
    function(cuda_timer_add_test name source)
        cuda_timer_configure_test(${name} ${source} ${PROJECT_SOURCE_DIR}/tests/test_main.cpp)
    endfunction()

    # Needs a real GPU: main exits 77 without one, ctest reports SKIPPED.
    # GoogleTest cannot survive nvcc's device pass, so kernels live in
    # tests/cuda_test_kernels.cu and the suites stay host C++.
    function(cuda_timer_add_device_test name source)
        cuda_timer_configure_test(${name} ${source} ${PROJECT_SOURCE_DIR}/tests/test_main_device.cpp)
        target_sources(${name} PRIVATE ${PROJECT_SOURCE_DIR}/tests/cuda_test_kernels.cu)
    endfunction()

    cuda_timer_add_test(test_logger           tests/test_logger.cpp)
    cuda_timer_add_test(test_compare          tests/test_compare.cpp)
    cuda_timer_add_test(test_stats            tests/test_stats.cpp)
    cuda_timer_add_test(test_types            tests/test_types.cpp)
    cuda_timer_add_device_test(test_timer      tests/test_timer.cpp)
    cuda_timer_add_device_test(test_overhead   tests/test_overhead.cpp)
    cuda_timer_add_device_test(test_environment tests/test_environment.cpp)
    cuda_timer_add_device_test(test_benchmark  tests/test_benchmark.cpp)
else()
    # ctest still works when suites are added later.
    enable_testing()
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda_timer-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_timer-config.cmake
    INSTALL_DESTINATION lib/cmake/cuda_timer
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_timer-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_timer-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_timer-config-version.cmake
    DESTINATION lib/cmake/cuda_timer
)

include(CPack)
set(CPACK_PACKAGE_NAME "cuda-timer")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Header-only CUDA timing/profiling library")
set(CPACK_PACKAGE_VENDOR "MardokHPC")
set(CPACK_PACKAGE_CONTACT "MardokHPC")
set(CPACK_GENERATOR "ZIP;TGZ")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")