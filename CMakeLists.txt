# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)

project(flagfft LANGUAGES CXX)

option(FLAGFFT_BUILD_CLI "Build the unified native validation/benchmark/tuning CLI" OFF)
option(FLAGFFT_BUILD_TESTS "Build C++ tests (requires Google Test)" OFF)
set(BACKEND "CUDA" CACHE STRING "FlagFFT adaptor and libtriton_jit backend")
set_property(CACHE BACKEND PROPERTY STRINGS "CUDA" "MUSA")

if(NOT (BACKEND STREQUAL "CUDA" OR BACKEND STREQUAL "MUSA"))
    message(FATAL_ERROR "FlagFFT currently supports only BACKEND=CUDA or BACKEND=MUSA")
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
find_package(SQLite3 REQUIRED)

# CMake >= 3.31 FindSQLite3 creates SQLite::SQLite3 instead of SQLite3::SQLite3
if(TARGET SQLite::SQLite3 AND NOT TARGET SQLite3::SQLite3)
    add_library(SQLite3::SQLite3 ALIAS SQLite::SQLite3)
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)"
    OUTPUT_VARIABLE FLAGFFT_TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE FLAGFFT_TORCH_ABI_QUERY_RESULT
)
if(FLAGFFT_TORCH_ABI_QUERY_RESULT EQUAL 0 AND DEFINED FLAGFFT_TORCH_CXX11_ABI)
    message(STATUS "Matching PyTorch libstdc++ ABI: _GLIBCXX_USE_CXX11_ABI=${FLAGFFT_TORCH_CXX11_ABI}")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${FLAGFFT_TORCH_CXX11_ABI})
else()
    message(WARNING "Could not determine PyTorch C++ ABI; using toolchain default. "
                    "If C++ tests fail to link with [abi:cxx11] undefined references, "
                    "this is the likely cause.")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/libtriton_jit/CMakeLists.txt")
    message(FATAL_ERROR "deps/libtriton_jit is required")
endif()
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
    src/plan/rader_utils.cpp
    src/plan/cost_model.cpp
    src/plan/auto_candidates.cpp
    src/plan/tune_candidates.cpp
    src/codegen/jit_kernel.cpp
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
add_subdirectory(src/adaptor)
target_compile_features(flagfft PUBLIC cxx_std_20)
target_include_directories(flagfft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src/utils/include
        src
)
target_link_libraries(flagfft PRIVATE TritonJIT::triton_jit SQLite3::SQLite3 nlohmann_json::nlohmann_json)
set_target_properties(flagfft PROPERTIES OUTPUT_NAME flagfft)

install(TARGETS flagfft
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
        ARCHIVE DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include")

# Test adaptor — reference FFT oracle for correctness checks and benchmarking.
# Reused by flagfft-cli (bench/verify) and ctest (correctness tests).
# Only built when CLI or tests are enabled (both need a reference FFT library).
if(FLAGFFT_BUILD_CLI OR FLAGFFT_BUILD_TESTS)
    if(BACKEND STREQUAL "CUDA")
        find_package(CUDAToolkit REQUIRED)
        add_library(flagfft_test_adaptor OBJECT
            src/adaptor/backend/cuda/test_adaptor.cpp
        )
        target_link_libraries(flagfft_test_adaptor PUBLIC
            CUDA::cufft
            CUDA::cudart
            CUDA::cuda_driver
        )
    else()
        if(NOT DEFINED MUSA_HOME)
            set(MUSA_HOME "/usr/local/musa")
        endif()
        find_library(MUSA_FFT_LIB mufft
            PATHS "${MUSA_HOME}/lib64" "${MUSA_HOME}/lib"
            NO_DEFAULT_PATH
        )
        if(NOT MUSA_FFT_LIB)
            find_library(MUSA_FFT_LIB mufft)
        endif()
        if(NOT MUSA_FFT_LIB)
            message(FATAL_ERROR "muFFT library not found; FLAGFFT_BUILD_CLI/TESTS requires mufft on MUSA")
        endif()
        add_library(flagfft_test_adaptor OBJECT
            src/adaptor/backend/musa/test_adaptor.cpp
        )
        target_link_libraries(flagfft_test_adaptor PUBLIC
            MUSA::musa_runtime
            ${MUSA_FFT_LIB}
        )
    endif()
    target_compile_features(flagfft_test_adaptor PUBLIC cxx_std_20)
    target_include_directories(flagfft_test_adaptor PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/utils/include
        src
    )
endif()

if(FLAGFFT_BUILD_CLI)
    if(BACKEND STREQUAL "CUDA")
        # cuFFT is only needed by test_adaptor (not by CLI directly).
        find_package(CUDAToolkit REQUIRED)
        if(NOT TARGET CUDA::cufft)
            message(FATAL_ERROR "FLAGFFT_BUILD_CLI requires the CUDA::cufft target")
        endif()
    endif()
    add_library(flagfft_cli_common OBJECT
        src/cli_tools/common/plan_handles.cpp
        src/cli_tools/common/cli_utils.cpp
        src/cli_tools/common/case.cpp
    )
    target_compile_features(flagfft_cli_common PUBLIC cxx_std_20)
    target_include_directories(flagfft_cli_common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/utils/include
        src
    )
    target_link_libraries(flagfft_cli_common PUBLIC
        flagfft
        nlohmann_json::nlohmann_json
    )

    add_executable(flagfft-cli
        src/cli_tools/flagfft-cli/main.cpp
        src/cli_tools/bench/runner.cpp
        src/cli_tools/bench/report.cpp
        src/cli_tools/tune/tune.cpp
    )
    target_compile_features(flagfft-cli PRIVATE cxx_std_20)
    target_include_directories(flagfft-cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/utils/include
        src/utils
        src/exec
        src
    )
    target_link_libraries(flagfft-cli
        PRIVATE
            flagfft
            flagfft_cli_common
            flagfft_test_adaptor
            nlohmann_json::nlohmann_json
    )
    if(BACKEND STREQUAL "CUDA")
        target_link_libraries(flagfft-cli PRIVATE CUDA::cufft)
    endif()
    install(TARGETS flagfft-cli RUNTIME DESTINATION "bin")
endif()

if(FLAGFFT_BUILD_TESTS)
    enable_testing()
    if(NOT TARGET gtest)
        include(FetchContent)
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP ON
        )
        set(INSTALL_GTEST OFF)
        FetchContent_MakeAvailable(googletest)
    endif()
    add_subdirectory(ctest)
endif()
