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

project(flagfft VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(FLAGFFT_BUILD_CLI "Build the unified native validation/benchmark/tuning CLI" OFF)
option(FLAGFFT_BUILD_TESTS "Build C++ tests (requires Google Test)" OFF)
set(BACKEND "CUDA" CACHE STRING "FlagFFT adaptor and libtriton_jit backend")
set_property(CACHE BACKEND PROPERTY STRINGS "CUDA" "MUSA" "PPU" "IX" "MACA" "NPU" "HCU")
set(_FLAGFFT_NPU_SIP_EXECUTION_DEFAULT OFF)
if(BACKEND STREQUAL "NPU")
    set(_FLAGFFT_NPU_SIP_EXECUTION_DEFAULT ON)
endif()
option(FLAGFFT_NPU_ENABLE_SIP_EXECUTION
       "Enable the installed Ascend SiP FFT as an optional NPU execution backend"
       ${_FLAGFFT_NPU_SIP_EXECUTION_DEFAULT})

if(NOT (BACKEND STREQUAL "CUDA" OR BACKEND STREQUAL "MUSA" OR BACKEND STREQUAL "PPU" OR BACKEND STREQUAL "IX" OR BACKEND STREQUAL "MACA" OR BACKEND STREQUAL "NPU" OR BACKEND STREQUAL "HCU"))
    message(FATAL_ERROR "FlagFFT currently supports only BACKEND=CUDA, BACKEND=MUSA, BACKEND=PPU, BACKEND=IX, BACKEND=MACA, BACKEND=NPU, or BACKEND=HCU")
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

set(TRITON_JIT_BUILD_EXAMPLES OFF CACHE BOOL "Build libtriton_jit examples" FORCE)
set(TRITON_JIT_INSTALL OFF CACHE BOOL "Install libtriton_jit" FORCE)
set(TRITON_JIT_BUILD_OPERATORS OFF CACHE BOOL "Build libtriton_jit operators" FORCE)
set(FLAGFFT_TRITON_JIT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/libtriton_jit"
    CACHE PATH "libtriton_jit source directory")
if(NOT EXISTS "${FLAGFFT_TRITON_JIT_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "libtriton_jit source is required at ${FLAGFFT_TRITON_JIT_SOURCE_DIR}")
endif()
add_subdirectory("${FLAGFFT_TRITON_JIT_SOURCE_DIR}"
                 "${CMAKE_CURRENT_BINARY_DIR}/deps/libtriton_jit")

if(BACKEND STREQUAL "HCU")
    # The bundled HCU backend logs every kernel launch at INFO. That host-side
    # work sits between benchmark events and adds latency to short FFTs.
    # Compile out INFO while retaining warnings and errors.
    target_compile_definitions(triton_jit PRIVATE GOOGLE_STRIP_LOG=1)

    # The HCU Triton backend discovers HIP in the dependency subdirectory. Keep
    # the matching hipFFT package discoverable for FlagFFT's reference adaptor
    # as well; DTK images use either /opt/dtk or a versioned /opt/dtk-* root.
    list(APPEND CMAKE_PREFIX_PATH
        "$ENV{ROCM_PATH}"
        "$ENV{HIP_PATH}"
        "$ENV{ROCM_PATH}/hipfft"
        "$ENV{HIP_PATH}/hipfft"
        "/opt/dtk"
        "/opt/dtk/hip"
        "/opt/dtk/hipfft"
        "/opt/dtk-26.04"
        "/opt/dtk-26.04/hip"
        "/opt/dtk-26.04/hipfft")
    find_package(hipfft CONFIG REQUIRED)
endif()

# libtriton_jit is a private bundled dependency. Give it a FlagFFT-owned
# soname so an installed libflagfft can coexist with the separately packaged
# libtriton_jit project, and locate it next to libflagfft at runtime.
set_target_properties(triton_jit PROPERTIES
    OUTPUT_NAME flagfft_triton_jit
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH FALSE
)

# libtorch's ARM OpenBLAS build has a private libgfortran dependency in
# torch.libs.  libtriton_jit discovers this directory in its subdirectory,
# but that directory scope does not reach the standalone CLI/capture targets
# declared here.  Add it at the parent scope so their link step can resolve
# the same transitive dependency.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), '..', 'torch.libs'))"
    OUTPUT_VARIABLE FLAGFFT_TORCH_LIBS_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(FLAGFFT_TORCH_LIBS_PATH AND EXISTS "${FLAGFFT_TORCH_LIBS_PATH}")
    link_directories("${FLAGFFT_TORCH_LIBS_PATH}")
endif()

if(TRITON_JIT_USE_EXTERNAL_JSON)
    # With an external package, libtriton_jit's imported target is scoped to
    # its subdirectory. Re-find it here for FlagFFT targets. When JSON is
    # provided through FetchContent, the normal target is already global and
    # must not be looked up as an installed package.
    # Only basic JSON (de)serialization is used, so the floor can follow the
    # oldest target: Ubuntu 22.04 ships 3.10.5, which is also what the deb
    # build container provides and what libtriton_jit requires for the same
    # switch. openEuler 24.03 (3.11.2) and Fedora 43 (3.11.3) are above it.
    find_package(nlohmann_json 3.10.5 REQUIRED)
endif()

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
    src/codegen/request_policy.cpp
    src/codegen/compiler.cpp
    src/exec/tuned_plan.cpp
    src/exec/raw_nodes.cpp
    src/exec/c_api_internal.cpp
    src/exec/c_api_plan.cpp
    src/exec/c_api_exec.cpp
)

add_library(flagfft SHARED ${FLAGFFT_CORE_SOURCES})
add_subdirectory(src/adaptor)

# CANN 9 ships the Ascend C toolchain and runtime helpers in the toolkit.
# Keep the native AIV leaf isolated to the NPU backend so other builds remain
# independent of the vendor SDK.  The leaf is opt-in at runtime while its
# correctness and performance are being evaluated.
if(BACKEND STREQUAL "NPU")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(_FLAGFFT_ASCEND_ARCH_DIR "aarch64-linux")
    else()
        set(_FLAGFFT_ASCEND_ARCH_DIR "x86_64-linux")
    endif()
    set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_TOOLKIT_HOME}")
    set(ASCENDC_CMAKE_DIR
        "${ASCEND_TOOLKIT_HOME}/${_FLAGFFT_ASCEND_ARCH_DIR}/tikcpp/ascendc_kernel_cmake")
    if(NOT EXISTS "${ASCENDC_CMAKE_DIR}/ascendc.cmake")
        message(FATAL_ERROR "Ascend C CMake helpers not found at ${ASCENDC_CMAKE_DIR}")
    endif()
    set(SOC_VERSION "Ascend910B4")
    set(RUN_MODE "npu")
    set(ASCEND_PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
    include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")
    ascendc_library(flagfft_npu_aiv_fft64 STATIC
        src/adaptor/backend/npu/ascendc_fft64_kernel.cpp)
    ascendc_library(flagfft_npu_aiv_fft256 STATIC
        src/adaptor/backend/npu/ascendc_fft256_kernel.cpp)
    ascendc_library(flagfft_npu_aiv_fft256_group4 STATIC
        src/adaptor/backend/npu/ascendc_fft256_group4_kernel.cpp)
    ascendc_library(flagfft_npu_aiv_fft256_group8 STATIC
        src/adaptor/backend/npu/ascendc_fft256_group8_kernel.cpp)
    ascendc_library(flagfft_npu_aiv_fft256_pair_group8 STATIC
        src/adaptor/backend/npu/ascendc_fft256_pair_group8_kernel.cpp)
    ascendc_library(flagfft_npu_aiv_fft256_pair_store_group8 STATIC
        src/adaptor/backend/npu/ascendc_fft256_pair_store_group8_kernel.cpp)
    ascendc_library(flagfft_npu_aiv_fft256_pair_radix4_store_group8 STATIC
        src/adaptor/backend/npu/ascendc_fft256_pair_radix4_store_group8_kernel.cpp)
    ascendc_library(flagfft_npu_aiv_transpose3d STATIC
        src/adaptor/backend/npu/ascendc_transpose3d_kernel.cpp)
    target_sources(flagfft PRIVATE
        src/adaptor/backend/npu/ascendc_fft64_host.cpp
        src/adaptor/backend/npu/ascendc_fft256_host.cpp
        src/adaptor/backend/npu/ascendc_transpose3d_host.cpp)
    target_link_libraries(flagfft PRIVATE
        flagfft_npu_aiv_fft64
        flagfft_npu_aiv_fft256
        flagfft_npu_aiv_fft256_group4
        flagfft_npu_aiv_fft256_group8
        flagfft_npu_aiv_fft256_pair_group8
        flagfft_npu_aiv_fft256_pair_store_group8
        flagfft_npu_aiv_fft256_pair_radix4_store_group8
        flagfft_npu_aiv_transpose3d)
    target_compile_definitions(flagfft PRIVATE FLAGFFT_BACKEND_NPU=1)
endif()

target_compile_features(flagfft PUBLIC cxx_std_20)
target_include_directories(flagfft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src/utils/include
        src
)
target_link_libraries(flagfft PRIVATE
    TritonJIT::triton_jit
    SQLite3::SQLite3
    nlohmann_json::nlohmann_json
    ${CMAKE_DL_LIBS}
)
set_target_properties(flagfft PROPERTIES
    OUTPUT_NAME flagfft
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH FALSE
)

install(TARGETS flagfft triton_jit
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
# The bundled libtriton_jit imports its Python helpers (standalone_compile)
# from <prefix>/share/triton_jit/scripts relative to its own library path.
install(DIRECTORY deps/libtriton_jit/scripts/
        DESTINATION "${CMAKE_INSTALL_DATADIR}/triton_jit/scripts"
        FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)

# Test adaptor — reference FFT oracle for correctness checks and benchmarking.
# Reused by flagfft-cli (bench/verify) and the native capture target. NPU links
# the SiP FFT library supplied by the CANN NNAL installation.
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
    elseif(BACKEND STREQUAL "MUSA")
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
    elseif(BACKEND STREQUAL "MACA")
        find_path(MCFFT_INCLUDE_DIR mcfft.h
            PATHS "${MACA_PATH}/include/mcfft" NO_DEFAULT_PATH REQUIRED)
        find_library(MCFFT_LIB mcfft
            PATHS "${MACA_PATH}/lib" NO_DEFAULT_PATH REQUIRED)
        add_library(flagfft_test_adaptor OBJECT
            src/adaptor/backend/maca/test_adaptor.cpp)
        target_include_directories(flagfft_test_adaptor PUBLIC "${MCFFT_INCLUDE_DIR}")
        target_link_libraries(flagfft_test_adaptor PUBLIC MACA::mcruntime "${MCFFT_LIB}")
    elseif(BACKEND STREQUAL "PPU")
        if(NOT DEFINED PPU_HOME)
            set(PPU_HOME "/usr/local/PPU_SDK")
        endif()
        # The PPU SDK ships a CUDA-compatible wrapper (CUDA_SDK) whose libcufft
        # maps cuFFT calls onto the native acfft library.
        find_library(PPU_FFT_LIB cufft
            PATHS "${PPU_HOME}/CUDA_SDK/lib64" "${PPU_HOME}/lib64" "${PPU_HOME}/lib"
            NO_DEFAULT_PATH
        )
        if(NOT PPU_FFT_LIB)
            find_library(PPU_FFT_LIB cufft)
        endif()
        if(NOT PPU_FFT_LIB)
            message(FATAL_ERROR "cuFFT (CUDA SDK wrapper) not found; FLAGFFT_BUILD_CLI/TESTS requires libcufft under PPU_HOME")
        endif()
        set(PPU_FFT_LIB "${PPU_FFT_LIB}" CACHE INTERNAL "PPU cuFFT wrapper library")
        add_library(flagfft_test_adaptor OBJECT
            src/adaptor/backend/ppu/test_adaptor.cpp
        )
        target_include_directories(flagfft_test_adaptor PUBLIC "${PPU_HOME}/CUDA_SDK/include")
        target_link_libraries(flagfft_test_adaptor PUBLIC
            PPU::ppu_runtime
            ${PPU_FFT_LIB}
        )
    elseif(BACKEND STREQUAL "IX")
        find_package(CUDAToolkit REQUIRED)
        add_library(flagfft_test_adaptor OBJECT
            src/adaptor/backend/ix/test_adaptor.cpp
        )
        target_link_libraries(flagfft_test_adaptor PUBLIC
            CUDA::cufft
            CUDA::cudart
            CUDA::cuda_driver
        )
    elseif(BACKEND STREQUAL "NPU")
        set(_ASCEND_SIP_DEFAULT_ROOT "$ENV{ASCEND_SIP_ROOT}")
        if(NOT _ASCEND_SIP_DEFAULT_ROOT)
            set(_ASCEND_SIP_DEFAULT_ROOT "/usr/local/Ascend/nnal/asdsip/latest")
        endif()
        set(ASCEND_SIP_ROOT "${_ASCEND_SIP_DEFAULT_ROOT}" CACHE PATH
            "installed Ascend SiP root used as the NPU FFT reference")
        find_path(ASCEND_SIP_INCLUDE_DIR fft_api.h
            HINTS "${ASCEND_SIP_ROOT}/include" REQUIRED)
        find_library(ASCEND_SIP_LIBRARY NAMES asdsip
            HINTS "${ASCEND_SIP_ROOT}/lib" REQUIRED)
        find_library(ASCEND_OPAPI_LIBRARY NAMES opapi
            HINTS "${ASCEND_TOOLKIT_HOME}/lib64"
                  "${ASCEND_TOOLKIT_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64" REQUIRED)
        find_library(ASCEND_NNOPBASE_LIBRARY NAMES nnopbase
            HINTS "${ASCEND_TOOLKIT_HOME}/lib64"
                  "${ASCEND_TOOLKIT_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64" REQUIRED)
        add_library(flagfft_test_adaptor OBJECT
            src/adaptor/backend/npu/test_adaptor.cpp
        )
        target_include_directories(flagfft_test_adaptor PUBLIC
            "${ASCEND_SIP_INCLUDE_DIR}")
        target_link_libraries(flagfft_test_adaptor PUBLIC
            "${ASCEND_SIP_LIBRARY}"
            "${ASCEND_OPAPI_LIBRARY}"
            "${ASCEND_NNOPBASE_LIBRARY}"
            Ascend::ascendcl
            Ascend::runtime)
    elseif(BACKEND STREQUAL "HCU")
        add_library(flagfft_test_adaptor OBJECT
            src/adaptor/backend/hcu/test_adaptor.cpp)
        target_link_libraries(flagfft_test_adaptor PUBLIC
            hip::hipfft
            hip::host)
    endif()
    target_compile_features(flagfft_test_adaptor PUBLIC cxx_std_20)
    target_include_directories(flagfft_test_adaptor PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/utils/include
        src
    )
endif()

if(FLAGFFT_BUILD_CLI)
    if(BACKEND STREQUAL "CUDA" OR BACKEND STREQUAL "IX")
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
            SQLite3::SQLite3
    )
    if(BACKEND STREQUAL "CUDA" OR BACKEND STREQUAL "IX")
        target_link_libraries(flagfft-cli PRIVATE CUDA::cufft)
    elseif(BACKEND STREQUAL "PPU")
        target_link_libraries(flagfft-cli PRIVATE ${PPU_FFT_LIB})
    endif()
    install(TARGETS flagfft-cli RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

if(FLAGFFT_BUILD_TESTS)
    enable_testing()
    if(NOT BACKEND STREQUAL "NPU" AND NOT TARGET gtest)
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
