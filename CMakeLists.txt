cmake_minimum_required(VERSION 3.18)

project(flagfft LANGUAGES CXX)

option(FLAGFFT_BUILD_TESTS "Build C++ tests" OFF)
option(FLAGFFT_BUILD_BENCHMARKS "Build C++ benchmark executables" OFF)
set(FLAGFFT_LIBTRITON_JIT_BACKEND "CUDA" CACHE STRING "libtriton_jit backend used by FlagFFT")
set_property(CACHE FLAGFFT_LIBTRITON_JIT_BACKEND PROPERTY STRINGS "CUDA")

if(NOT FLAGFFT_LIBTRITON_JIT_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "FlagFFT currently supports only FLAGFFT_LIBTRITON_JIT_BACKEND=CUDA")
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)
find_package(SQLite3 REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/libtriton_jit/CMakeLists.txt")
    message(FATAL_ERROR "deps/libtriton_jit is required")
endif()
set(BACKEND "${FLAGFFT_LIBTRITON_JIT_BACKEND}" CACHE STRING "Target backend for libtriton_jit" FORCE)
set(TRITON_JIT_BUILD_EXAMPLES OFF CACHE BOOL "Build libtriton_jit examples" FORCE)
set(TRITON_JIT_INSTALL OFF CACHE BOOL "Install libtriton_jit" FORCE)
add_subdirectory(deps/libtriton_jit)

set(FLAGFFT_CORE_SOURCES
    src/utils/base.cpp
    src/utils/problem_key.cpp
    src/utils/plan_key.cpp
    src/utils/kernel_key.cpp
    src/utils/sqlite_wrapper.cpp
    src/utils/plan_json.cpp
    src/plan/nodes.cpp
    src/plan/builder_context.cpp
    src/plan/plan_deserialize.cpp
    src/plan/factorization.cpp
    src/plan/cost_model.cpp
    src/plan/auto_candidates.cpp
    src/plan/tune_candidates.cpp
    src/runtime/cuda_runtime.cpp
    src/runtime/cuda/memory.cpp
    src/runtime/cuda/check.cpp
    src/runtime/cuda/device.cpp
    src/codegen/jit_cache.cpp
    src/codegen/plan_kernel_keys.cpp
    src/codegen/tables.cpp
    src/codegen/compiler.cpp
    src/exec/tuned_plan.cpp
    src/exec/raw_nodes.cpp
    src/exec/c_api_internal.cpp
    src/exec/c_api_plan.cpp
    src/exec/c_api_exec.cpp
)

add_library(flagfft SHARED ${FLAGFFT_CORE_SOURCES})
target_compile_features(flagfft PUBLIC cxx_std_20)
target_include_directories(flagfft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src/utils/include
        src
)
target_link_libraries(flagfft PRIVATE CUDA::cuda_driver TritonJIT::triton_jit SQLite3::SQLite3 nlohmann_json::nlohmann_json)
set_target_properties(flagfft PROPERTIES OUTPUT_NAME flagfft)

install(TARGETS flagfft
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
        ARCHIVE DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include")

if(TARGET CUDA::cufft)
    add_library(flagfft_cli_common OBJECT
        src/cli_tools/common/runtime_raii.cpp
        src/cli_tools/common/plan_handles.cpp
        src/cli_tools/common/cli_utils.cpp
    )
    target_compile_features(flagfft_cli_common PUBLIC cxx_std_20)
    target_include_directories(flagfft_cli_common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/utils/include
        src
    )
    target_link_libraries(flagfft_cli_common PUBLIC
        flagfft
        CUDA::cudart
        CUDA::cufft
        nlohmann_json::nlohmann_json
    )
endif()

if(FLAGFFT_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        find_package(GTest QUIET)
        if(NOT GTest_FOUND)
            include(FetchContent)
            FetchContent_Declare(
                googletest
                URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            )
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(googletest)
        endif()

        add_executable(flagfft_plan_tests tests/ctest/flagfft_plan_test.cpp)
        target_compile_features(flagfft_plan_tests PRIVATE cxx_std_20)
        target_include_directories(flagfft_plan_tests PRIVATE src/utils/include src)
        target_link_libraries(flagfft_plan_tests
            PRIVATE
                flagfft
                CUDA::cuda_driver
                GTest::gtest_main
        )
        add_test(NAME flagfft_plan_tests COMMAND flagfft_plan_tests)

        if(TARGET CUDA::cufft)
            add_executable(flagfft_c_api_tests tests/ctest/flagfft_c_api_test.cpp)
            target_compile_features(flagfft_c_api_tests PRIVATE cxx_std_20)
            target_include_directories(flagfft_c_api_tests PRIVATE src/utils/include src)
            target_link_libraries(flagfft_c_api_tests
                PRIVATE
                    flagfft
                    CUDA::cudart
                    CUDA::cufft
                    GTest::gtest_main
            )
            add_test(NAME flagfft_c_api_tests COMMAND flagfft_c_api_tests)
            set_tests_properties(flagfft_c_api_tests PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            )
        else()
            message(WARNING "CUDA::cufft target was not found; skipping cuFFT comparison tests")
        endif()
    endif()
endif()

if(FLAGFFT_BUILD_BENCHMARKS)
    if(TARGET CUDA::cufft)
        add_executable(bench_vs_cufft benchmark/bench_vs_cufft.cpp)
        target_compile_features(bench_vs_cufft PRIVATE cxx_std_20)
        target_include_directories(bench_vs_cufft PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            src
        )
        target_link_libraries(bench_vs_cufft
            PRIVATE
                flagfft
                CUDA::cudart
                CUDA::cufft
        )

        add_executable(flagfft-tuner
            src/cli_tools/flagfft-tuner/main.cpp
            src/cli_tools/flagfft-tuner/bench.cpp
            src/cli_tools/flagfft-tuner/sqlite.cpp
        )
        target_compile_features(flagfft-tuner PRIVATE cxx_std_20)
        target_include_directories(flagfft-tuner PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            src/utils/include
            src/utils
            src/exec
            src
        )
        target_link_libraries(flagfft-tuner
            PRIVATE
                flagfft
                flagfft_cli_common
                CUDA::cudart
                CUDA::cufft
                SQLite3::SQLite3
                nlohmann_json::nlohmann_json
        )
    else()
        message(WARNING "CUDA::cufft not found; bench_vs_cufft will not be built")
    endif()
endif()
