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

if(NOT BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "FlagFFT adaptor currently supports only BACKEND=CUDA")
endif()

find_package(CUDAToolkit REQUIRED)

target_sources(flagfft PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/adaptor.cpp
)
target_link_libraries(flagfft PRIVATE CUDA::cuda_driver)
