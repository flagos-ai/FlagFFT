enable_testing()

# Reuse googletest already fetched by libtriton_jit via FetchContent
if(NOT TARGET gtest)
    find_package(GTest REQUIRED)
endif()
include(GoogleTest)

# Test adaptor library (backend-specific, mirrors src/adaptor/ pattern)
add_library(flagfft_test_adaptor STATIC)
target_compile_features(flagfft_test_adaptor PUBLIC cxx_std_20)
target_include_directories(flagfft_test_adaptor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(flagfft_test_adaptor PUBLIC flagfft gtest)

add_subdirectory(backend)

# Test executables
set(TEST_TARGETS
    test_plan
    test_exec_c2c
    test_exec_z2z
    test_exec_r2c_c2r
    test_exec_d2z_z2d
)

foreach(test_name ${TEST_TARGETS})
    add_executable(${test_name} ${test_name}.cpp main.cpp)
    target_link_libraries(${test_name} PRIVATE flagfft_test_adaptor gtest)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    gtest_discover_tests(${test_name})
endforeach()
