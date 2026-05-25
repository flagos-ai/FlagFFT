if(BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_sources(flagfft_test_adaptor PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/cuda/adaptor.cpp
    )
    target_link_libraries(flagfft_test_adaptor PUBLIC
        CUDA::cufft
        CUDA::cudart
        CUDA::cuda_driver
    )
elseif(BACKEND STREQUAL "NULL")
    target_sources(flagfft_test_adaptor PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/null/adaptor.cpp
    )
else()
    message(FATAL_ERROR "Unsupported test backend: ${BACKEND}")
endif()
