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

set(FLAGFFT_CPP_SOURCES
    src/cpp/bindings/module.cpp
    src/cpp/common/request.cpp
    src/cpp/plan/planner.cpp
    src/cpp/runtime/cuda_runtime.cpp
    src/cpp/codegen/triton_compiler.cpp
    src/cpp/exec/executor.cpp
)

nanobind_add_module(_flagfft_core ${FLAGFFT_CPP_SOURCES})
target_compile_features(_flagfft_core PRIVATE cxx_std_17)
target_include_directories(_flagfft_core PRIVATE src/cpp/include)
target_link_libraries(_flagfft_core PRIVATE CUDA::cuda_driver)

install(TARGETS _flagfft_core
        LIBRARY DESTINATION "."
        RUNTIME DESTINATION "."
        ARCHIVE DESTINATION ".")
install(FILES flagfft.py DESTINATION ".")
install(FILES src/__init__.py src/api.py src/flagfft.py src/triton_aot.py DESTINATION "src")
install(DIRECTORY src/codegen src/codelet DESTINATION "src" FILES_MATCHING PATTERN "*.py")
