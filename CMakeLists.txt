cmake_minimum_required(VERSION 3.18)

project(flagfft LANGUAGES CXX)

option(FLAGFFT_BUILD_PYTHON "Build the legacy nanobind Python debug/tune module" OFF)
option(FLAGFFT_BUILD_TESTS "Build C++ tests" ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)
find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)
nanobind_build_library(nanobind-static)

set(FLAGFFT_CORE_SOURCES
    src/utils/base.cpp
    src/utils/problem_key.cpp
    src/utils/plan_key.cpp
    src/utils/kernel_key.cpp
    src/utils/serialization.cpp
    src/utils/request_builder.cpp
    src/plan/nodes.cpp
    src/plan/builder_context.cpp
    src/plan/plan_deserialize.cpp
    src/plan/factorization.cpp
    src/plan/cost_model.cpp
    src/plan/auto_candidates.cpp
    src/plan/tune_candidates.cpp
    src/runtime/cuda_runtime.cpp
    src/codegen/aot_cache.cpp
    src/codegen/kernel_keys.cpp
    src/codegen/tables.cpp
    src/codegen/compiler.cpp
    src/exec/helpers.cpp
    src/exec/raw_nodes.cpp
    src/exec/tensor_leaf.cpp
    src/exec/tensor_four_step.cpp
    src/exec/tensor_bluestein.cpp
    src/exec/executable_plan.cpp
    src/exec/plan_cache.cpp
    src/exec/debug_api.cpp
    src/exec/c_api_internal.cpp
    src/exec/c_api_plan.cpp
    src/exec/c_api_exec.cpp
)

add_library(flagfft SHARED ${FLAGFFT_CORE_SOURCES})
target_compile_features(flagfft PUBLIC cxx_std_17)
target_include_directories(flagfft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src/utils/include
)
target_link_libraries(flagfft PRIVATE nanobind-static Python::Python CUDA::cuda_driver)
set_target_properties(flagfft PROPERTIES OUTPUT_NAME flagfft)

if(FLAGFFT_BUILD_PYTHON)
    nanobind_add_module(_flagfft_core src/utils/module.cpp)
    target_compile_features(_flagfft_core PRIVATE cxx_std_17)
    target_include_directories(_flagfft_core PRIVATE include src/utils/include)
    target_link_libraries(_flagfft_core PRIVATE flagfft CUDA::cuda_driver)
    install(TARGETS _flagfft_core
            LIBRARY DESTINATION "."
            RUNTIME DESTINATION "."
            ARCHIVE DESTINATION ".")
endif()

install(TARGETS flagfft
        LIBRARY DESTINATION "lib"
        RUNTIME DESTINATION "bin"
        ARCHIVE DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include")
install(FILES src/__init__.py src/triton_aot.py DESTINATION "src")
install(DIRECTORY src/codegen src/tune DESTINATION "src" FILES_MATCHING PATTERN "*.py")

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
        target_compile_features(flagfft_plan_tests PRIVATE cxx_std_17)
        target_include_directories(flagfft_plan_tests PRIVATE src/utils/include)
        target_link_libraries(flagfft_plan_tests
            PRIVATE
                flagfft
                nanobind-static
                Python::Python
                CUDA::cuda_driver
                GTest::gtest_main
        )
        add_test(NAME flagfft_plan_tests COMMAND flagfft_plan_tests)

        if(TARGET CUDA::cufft)
            add_executable(flagfft_c_api_tests tests/ctest/flagfft_c_api_test.cpp)
            target_compile_features(flagfft_c_api_tests PRIVATE cxx_std_17)
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
