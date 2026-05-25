cmake_minimum_required(VERSION 3.18)

project(flagfft LANGUAGES CXX)

option(FLAGFFT_BUILD_CLI "Build the unified native validation/benchmark/tuning CLI" OFF)
set(BACKEND "CUDA" CACHE STRING "FlagFFT adaptor and libtriton_jit backend")
set_property(CACHE BACKEND PROPERTY STRINGS "CUDA")

if(NOT BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "FlagFFT currently supports only BACKEND=CUDA")
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
find_package(SQLite3 REQUIRED)

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

if(FLAGFFT_BUILD_CLI)
    # cuFFT is used only by the CUDA validation/performance oracle in the CLI.
    find_package(CUDAToolkit REQUIRED)
    if(NOT TARGET CUDA::cufft)
        message(FATAL_ERROR "FLAGFFT_BUILD_CLI requires the CUDA::cufft target")
    endif()
    add_library(flagfft_cli_common OBJECT
        src/cli_tools/common/plan_handles.cpp
        src/cli_tools/common/cli_utils.cpp
        src/cli_tools/common/case.cpp
        src/cli_tools/common/runner.cpp
    )
    target_compile_features(flagfft_cli_common PUBLIC cxx_std_20)
    target_include_directories(flagfft_cli_common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        src/utils/include
        src
    )
    target_link_libraries(flagfft_cli_common PUBLIC
        flagfft
        CUDA::cufft
        nlohmann_json::nlohmann_json
    )

    add_executable(flagfft-cli
        src/cli_tools/flagfft-cli/main.cpp
        src/cli_tools/tune/driver.cpp
        src/cli_tools/tune/bench.cpp
        src/cli_tools/tune/sqlite.cpp
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
            CUDA::cufft
            SQLite3::SQLite3
            nlohmann_json::nlohmann_json
    )
    install(TARGETS flagfft-cli RUNTIME DESTINATION "bin")
endif()
