enable_testing()

if(NOT TARGET gtest)
    find_package(GTest REQUIRED)
endif()

set(TEST_TARGETS
    test_plan
    test_exec_c2c
    test_exec_z2z
    test_exec_r2c
    test_exec_c2r
    test_exec_d2z
    test_exec_z2d
    test_exec_r2c_c2r
    test_exec_d2z_z2d
)

foreach(test_name ${TEST_TARGETS})
    add_executable(${test_name} ${test_name}.cpp main.cpp)
    target_link_libraries(${test_name} PRIVATE flagfft flagfft_test_adaptor gtest)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    # A FlagFFT executable shares JIT initialization across its GTest cases.
    # Register it once rather than spawning one process per discovered case.
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
        SKIP_REGULAR_EXPRESSION "cuInit failed|Which is: 11"
    )
endforeach()
