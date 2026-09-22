cmake_minimum_required(VERSION 3.18)
project(maca_tail_plans_cpu LANGUAGES CXX)
enable_testing()

# Standalone CPU build: no backend SDK, torch, GPU runtime, or JIT required.
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  set(FLAGFFT_GTEST_SOURCE_DIR /usr/src/googletest CACHE PATH "GoogleTest sources")
  add_subdirectory(${FLAGFFT_GTEST_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/googletest)
endif()
get_filename_component(FFT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(test_maca_tail_plans
  ../test_maca_tail_plans.cpp ../test_maca_prime_tune.cpp
  ../test_maca_tail_policy.cpp
  ${FFT_ROOT}/src/utils/base.cpp ${FFT_ROOT}/src/utils/plan_key.cpp
  ${FFT_ROOT}/src/plan/nodes.cpp ${FFT_ROOT}/src/plan/builder_context.cpp
  ${FFT_ROOT}/src/plan/factorization.cpp ${FFT_ROOT}/src/plan/rader_utils.cpp
  ${FFT_ROOT}/src/plan/cost_model.cpp ${FFT_ROOT}/src/plan/auto_candidates.cpp
  ${FFT_ROOT}/src/plan/tune_candidates.cpp ${FFT_ROOT}/src/exec/c_api_internal.cpp
  ${FFT_ROOT}/src/codegen/request_policy.cpp)
target_compile_features(test_maca_tail_plans PRIVATE cxx_std_20)
target_include_directories(test_maca_tail_plans PRIVATE
  ${FFT_ROOT}/include ${FFT_ROOT}/src ${FFT_ROOT}/src/utils/include)
target_link_libraries(test_maca_tail_plans PRIVATE GTest::gtest_main)
add_test(NAME maca_tail_plans_cpu COMMAND test_maca_tail_plans)
