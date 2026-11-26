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
    find_program(FLAGFFT_MACA_MXCC mxcc PATHS "${MACA_PATH}/mxgpu_llvm/bin" NO_DEFAULT_PATH)
    find_library(FLAGFFT_MACA_COMPILER_RUNTIME mccompiler PATHS "${MACA_PATH}/lib" NO_DEFAULT_PATH)
    if(FLAGFFT_MACA_MXCC AND FLAGFFT_MACA_COMPILER_RUNTIME)
        set(maca_fft16_source "${CMAKE_CURRENT_LIST_DIR}/backend/maca/fft16.cpp")
        set(maca_fft16_object "${CMAKE_CURRENT_BINARY_DIR}/fft16_maca.o")
        add_custom_command(OUTPUT "${maca_fft16_object}"
            COMMAND "${FLAGFFT_MACA_MXCC}" -x maca "--maca-path=${MACA_PATH}"
                    -offload-arch=xcore1000 -std=c++17 -O3 -fPIC
                    -c "${maca_fft16_source}" -o "${maca_fft16_object}"
            DEPENDS "${maca_fft16_source}"
            VERBATIM)
        add_custom_target(flagfft_maca_fft16 DEPENDS "${maca_fft16_object}")
        add_dependencies(flagfft flagfft_maca_fft16)
        set_source_files_properties("${maca_fft16_object}" TARGET_DIRECTORY flagfft
            PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
        target_sources(flagfft PRIVATE "${maca_fft16_object}")
        target_link_libraries(flagfft PRIVATE "${FLAGFFT_MACA_COMPILER_RUNTIME}")
        target_compile_definitions(flagfft PRIVATE FLAGFFT_MACA_NATIVE_FFT16=1)
    endif()
elseif(BACKEND STREQUAL "HCU")
    target_sources(flagfft PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/hcu/adaptor.cpp
    )
    target_link_libraries(flagfft PRIVATE hip::host)
else()
    message(FATAL_ERROR "FlagFFT adaptor currently supports only BACKEND=CUDA, BACKEND=MUSA, BACKEND=PPU, BACKEND=IX, BACKEND=MACA, BACKEND=NPU, or BACKEND=HCU")
endif()
