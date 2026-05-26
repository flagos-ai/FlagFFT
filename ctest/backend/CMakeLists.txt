find_package(CUDAToolkit REQUIRED)
target_sources(flagfft_test_adaptor PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/cuda/adaptor.cpp
)
target_link_libraries(flagfft_test_adaptor PUBLIC
    CUDA::cufft
    CUDA::cudart
    CUDA::cuda_driver
)
