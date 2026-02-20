# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Try to find pybind11 from Python environment if not found in system
if(NOT pybind11_DIR)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    RESULT_VARIABLE PYBIND11_FOUND_VIA_PYTHON)
  if(PYBIND11_FOUND_VIA_PYTHON EQUAL 0 AND pybind11_DIR)
    message(STATUS "Found pybind11 via Python: ${pybind11_DIR}")
  endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

# Create the Python module
pybind11_add_module(_edgellm_runtime MODULE edgellm_pybind.cpp)

# Set include directories
target_include_directories(
  _edgellm_runtime
  PRIVATE ${CMAKE_SOURCE_DIR}/cpp ${CMAKE_SOURCE_DIR}/3rdParty/nlohmannJson
          ${COMMON_INCLUDE_DIRS})

# Link against the core libraries
target_link_libraries(_edgellm_runtime PRIVATE edgellmCore edgellmTokenizer
                                               ${CUDART_LIB} ${NVINFER_LIB} dl)

# Add cross-build link options for CUDA library paths
add_cross_build_link_options(_edgellm_runtime)

# Set C++ standard
target_compile_features(_edgellm_runtime PRIVATE cxx_std_17)

# Set output directory
set_target_properties(
  _edgellm_runtime PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                              ${CMAKE_BINARY_DIR}/python/tensorrt_edgellm)

# Install target
install(TARGETS _edgellm_runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/python/tensorrt_edgellm)
