if(NOT BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "FlagFFT adaptor currently supports only BACKEND=CUDA")
endif()

find_package(CUDAToolkit REQUIRED)

target_sources(flagfft PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/adaptor.cpp
)
target_link_libraries(flagfft PRIVATE CUDA::cuda_driver)
