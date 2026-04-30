cmake_minimum_required(VERSION 3.18)

project(flagfft_core LANGUAGES CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_flagfft_core src/cpp/flagfft_core.cpp)
target_compile_features(_flagfft_core PRIVATE cxx_std_17)
target_link_libraries(_flagfft_core PRIVATE CUDA::cuda_driver)

install(TARGETS _flagfft_core
        LIBRARY DESTINATION "."
        RUNTIME DESTINATION "."
        ARCHIVE DESTINATION ".")
install(FILES flagfft.py DESTINATION ".")
install(DIRECTORY src DESTINATION "." FILES_MATCHING PATTERN "*.py")
