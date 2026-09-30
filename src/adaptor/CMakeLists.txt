# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(BACKEND STREQUAL "NPU")
    target_sources(flagfft PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/npu/adaptor.cpp
    )
    target_link_libraries(flagfft PRIVATE Ascend::ascendcl Ascend::runtime)
    if(FLAGFFT_NPU_ENABLE_SIP_EXECUTION)
        set(_ASCEND_SIP_DEFAULT_ROOT "$ENV{ASCEND_SIP_ROOT}")
        if(NOT _ASCEND_SIP_DEFAULT_ROOT)
            set(_ASCEND_SIP_DEFAULT_ROOT "/usr/local/Ascend/nnal/asdsip/latest")
        endif()
        set(ASCEND_SIP_ROOT "${_ASCEND_SIP_DEFAULT_ROOT}" CACHE PATH
            "installed Ascend SiP root used by the optional NPU execution backend")
        find_path(ASCEND_SIP_INCLUDE_DIR fft_api.h
            HINTS "${ASCEND_SIP_ROOT}/include" REQUIRED)
        find_library(ASCEND_SIP_LIBRARY NAMES asdsip
            HINTS "${ASCEND_SIP_ROOT}/lib" REQUIRED)
        find_library(ASCEND_OPAPI_LIBRARY NAMES opapi
            HINTS "${ASCEND_TOOLKIT_HOME}/lib64"
                  "${ASCEND_TOOLKIT_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64" REQUIRED)
        find_library(ASCEND_NNOPBASE_LIBRARY NAMES nnopbase
            HINTS "${ASCEND_TOOLKIT_HOME}/lib64"
                  "${ASCEND_TOOLKIT_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64" REQUIRED)
        target_sources(flagfft PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/backend/npu/sip_executor.cpp)
        target_include_directories(flagfft PRIVATE "${ASCEND_SIP_INCLUDE_DIR}")
        target_compile_definitions(flagfft PRIVATE FLAGFFT_NPU_ENABLE_SIP_EXECUTION=1)
        target_link_libraries(flagfft PRIVATE
            "${ASCEND_SIP_LIBRARY}"
            "${ASCEND_OPAPI_LIBRARY}"
            "${ASCEND_NNOPBASE_LIBRARY}")
    endif()
elseif(BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_sources(flagfft PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/adaptor.cpp
    )
    target_link_libraries(flagfft PRIVATE CUDA::cuda_driver)
elseif(BACKEND STREQUAL "MUSA")
    target_sources(flagfft PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/musa/adaptor.cpp
    )
    target_link_libraries(flagfft PRIVATE MUSA::musa_runtime)
elseif(BACKEND STREQUAL "PPU")
    target_sources(flagfft PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/ppu/adaptor.cpp
    )
    target_link_libraries(flagfft PRIVATE PPU::ppu_runtime)
elseif(BACKEND STREQUAL "IX")
    find_package(CUDAToolkit REQUIRED COMPONENTS cuda_driver)
    target_sources(flagfft PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/ix/adaptor.cpp
    )
    target_link_libraries(flagfft PRIVATE CUDA::cuda_driver)
elseif(BACKEND STREQUAL "MACA")
    target_sources(flagfft PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/maca/adaptor.cpp
    )
    target_link_libraries(flagfft PRIVATE MACA::mcruntime)
elseif(BACKEND STREQUAL "HCU")
    target_sources(flagfft PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/hcu/adaptor.cpp
    )
    target_link_libraries(flagfft PRIVATE hip::host)
else()
    message(FATAL_ERROR "FlagFFT adaptor currently supports only BACKEND=CUDA, BACKEND=MUSA, BACKEND=PPU, BACKEND=IX, BACKEND=MACA, BACKEND=NPU, or BACKEND=HCU")
endif()
