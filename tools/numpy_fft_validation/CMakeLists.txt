cmake_minimum_required(VERSION 3.18)

project(numpy_fft_validation_capture LANGUAGES CXX)

set(FLAGFFT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH
    "FlagFFT source tree containing include/ and src/")
set(FLAGFFT_BUILD_DIR "" CACHE PATH
    "Existing FlagFFT build directory containing libflagfft.so")
set(BACKEND "CUDA" CACHE STRING "FlagFFT backend: CUDA, MUSA, or PPU")
set_property(CACHE BACKEND PROPERTY STRINGS CUDA MUSA PPU)

if(FLAGFFT_BUILD_DIR STREQUAL "")
    message(FATAL_ERROR "-DFLAGFFT_BUILD_DIR=/path/to/FlagFFT/build is required")
endif()

get_filename_component(FLAGFFT_SOURCE_DIR "${FLAGFFT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(FLAGFFT_BUILD_DIR "${FLAGFFT_BUILD_DIR}" ABSOLUTE)
set(FLAGFFT_LIBRARY "${FLAGFFT_BUILD_DIR}/libflagfft.so")
if(NOT EXISTS "${FLAGFFT_LIBRARY}")
    message(FATAL_ERROR "FlagFFT shared library not found: ${FLAGFFT_LIBRARY}")
endif()

add_library(flagfft SHARED IMPORTED GLOBAL)
set_target_properties(flagfft PROPERTIES
    IMPORTED_LOCATION "${FLAGFFT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FLAGFFT_SOURCE_DIR}/include")

add_executable(numpy_fft_capture capture.cpp)
target_compile_features(numpy_fft_capture PRIVATE cxx_std_20)
target_include_directories(numpy_fft_capture PRIVATE
    "${FLAGFFT_SOURCE_DIR}/include"
    "${FLAGFFT_SOURCE_DIR}/src")
target_link_libraries(numpy_fft_capture PRIVATE flagfft)

# The existing test adaptor is compiled directly into this tool.  This keeps
# the tool independent of the FlagFFT build graph while reusing
# the exact platform-library calls already used by ctest and flagfft-cli.
if(BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    set(REFERENCE_SOURCE "${FLAGFFT_SOURCE_DIR}/src/adaptor/backend/cuda/test_adaptor.cpp")
    target_include_directories(numpy_fft_capture PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_sources(numpy_fft_capture PRIVATE "${REFERENCE_SOURCE}")
    target_link_libraries(numpy_fft_capture PRIVATE
        CUDA::cufft
        CUDA::cudart
        CUDA::cuda_driver)
elseif(BACKEND STREQUAL "MUSA")
    set(MUSA_HOME "/usr/local/musa" CACHE PATH "MUSA SDK root")
    find_path(MUSA_INCLUDE_DIR musa_runtime_api.h
        PATHS "${MUSA_HOME}/include" NO_DEFAULT_PATH REQUIRED)
    find_path(MUFFT_INCLUDE_DIR mufft.h
        PATHS "${MUSA_HOME}/include" NO_DEFAULT_PATH REQUIRED)
    find_library(MUSA_RUNTIME_LIB NAMES musa_runtime musa
        PATHS "${MUSA_HOME}/lib64" "${MUSA_HOME}/lib" NO_DEFAULT_PATH REQUIRED)
    find_library(MUFFT_LIB NAMES mufft
        PATHS "${MUSA_HOME}/lib64" "${MUSA_HOME}/lib" NO_DEFAULT_PATH REQUIRED)
    set(REFERENCE_SOURCE "${FLAGFFT_SOURCE_DIR}/src/adaptor/backend/musa/test_adaptor.cpp")
    target_include_directories(numpy_fft_capture PRIVATE
        "${MUSA_INCLUDE_DIR}"
        "${MUFFT_INCLUDE_DIR}")
    target_sources(numpy_fft_capture PRIVATE "${REFERENCE_SOURCE}")
    target_link_libraries(numpy_fft_capture PRIVATE "${MUFFT_LIB}" "${MUSA_RUNTIME_LIB}")
elseif(BACKEND STREQUAL "PPU")
    set(PPU_HOME "/usr/local/PPU_SDK" CACHE PATH "PPU SDK root")
    find_path(PPU_INCLUDE_DIR cuda_runtime_api.h
        PATHS "${PPU_HOME}/CUDA_SDK/include" "${PPU_HOME}/include" NO_DEFAULT_PATH REQUIRED)
    find_library(PPU_FFT_LIB NAMES cufft
        PATHS "${PPU_HOME}/CUDA_SDK/lib64" "${PPU_HOME}/lib64" "${PPU_HOME}/lib"
        NO_DEFAULT_PATH REQUIRED)
    set(REFERENCE_SOURCE "${FLAGFFT_SOURCE_DIR}/src/adaptor/backend/ppu/test_adaptor.cpp")
    target_include_directories(numpy_fft_capture PRIVATE "${PPU_INCLUDE_DIR}")
    target_sources(numpy_fft_capture PRIVATE "${REFERENCE_SOURCE}")
    target_link_libraries(numpy_fft_capture PRIVATE "${PPU_FFT_LIB}")
else()
    message(FATAL_ERROR "Unsupported BACKEND=${BACKEND}; expected CUDA, MUSA, or PPU")
endif()

# libflagfft.so's build-tree RPATH is not guaranteed to be present in a
# standalone executable.  The run script also adds vendor paths to
# LD_LIBRARY_PATH, but these entries make the common build-tree case work
# without extra linker flags.
set(_tool_rpath
    "${FLAGFFT_BUILD_DIR}"
    "${FLAGFFT_BUILD_DIR}/deps/libtriton_jit/src")
if(BACKEND STREQUAL "CUDA")
    list(APPEND _tool_rpath "${CUDAToolkit_LIBRARY_ROOT}")
elseif(BACKEND STREQUAL "MUSA")
    list(APPEND _tool_rpath "${MUSA_HOME}/lib64" "${MUSA_HOME}/lib")
elseif(BACKEND STREQUAL "PPU")
    list(APPEND _tool_rpath "${PPU_HOME}/CUDA_SDK/lib64" "${PPU_HOME}/lib64" "${PPU_HOME}/lib")
endif()
set_target_properties(numpy_fft_capture PROPERTIES
    BUILD_RPATH "${_tool_rpath}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
