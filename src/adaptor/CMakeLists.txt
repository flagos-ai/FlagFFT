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
        set(maca_native_objects)
        foreach(maca_kernel IN ITEMS fft16 fft23 r2c2048 long_axis cube_transpose)
            set(maca_source "${CMAKE_CURRENT_LIST_DIR}/backend/maca/${maca_kernel}.cpp")
            set(maca_object "${CMAKE_CURRENT_BINARY_DIR}/${maca_kernel}_maca.o")
            add_custom_command(OUTPUT "${maca_object}"
                COMMAND "${FLAGFFT_MACA_MXCC}" -x maca "--maca-path=${MACA_PATH}"
                        -offload-arch=xcore1000 -std=c++17 -O3 -fPIC
                        -c "${maca_source}" -o "${maca_object}"
                DEPENDS "${maca_source}"
                VERBATIM)
            list(APPEND maca_native_objects "${maca_object}")
        endforeach()
        add_custom_target(flagfft_maca_native_kernels DEPENDS ${maca_native_objects})
        add_dependencies(flagfft flagfft_maca_native_kernels)
        set_source_files_properties(${maca_native_objects} TARGET_DIRECTORY flagfft
            PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
        target_sources(flagfft PRIVATE ${maca_native_objects})
        target_link_libraries(flagfft PRIVATE "${FLAGFFT_MACA_COMPILER_RUNTIME}")
        target_compile_definitions(flagfft PRIVATE FLAGFFT_MACA_NATIVE_KERNELS=1)
    endif()
elseif(BACKEND STREQUAL "HCU")
    target_sources(flagfft PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend/hcu/adaptor.cpp
    )
    target_link_libraries(flagfft PRIVATE hip::host)
else()
    message(FATAL_ERROR "FlagFFT adaptor currently supports only BACKEND=CUDA, BACKEND=MUSA, BACKEND=PPU, BACKEND=IX, BACKEND=MACA, BACKEND=NPU, or BACKEND=HCU")
endif()
