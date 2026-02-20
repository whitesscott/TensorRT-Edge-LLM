#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Setup script for building TensorRT Edge LLM Python bindings.

This script builds the C++ pybind11 extension module using CMake.

Usage:
    # Basic build (requires TRT_PACKAGE_DIR environment variable)
    python setup_pybind.py build_ext --inplace

    # With explicit TensorRT path
    TRT_PACKAGE_DIR=/path/to/tensorrt python setup_pybind.py build_ext --inplace

    # Install to site-packages
    pip install -e .
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension module.

    This class represents a CMake-based extension. It doesn't specify sources
    because CMake handles the compilation.
    """

    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext command that uses CMake to build the extension."""

    def build_extension(self, ext: CMakeExtension) -> None:
        # Get the directory where the extension will be placed
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Build configuration
        cfg = "Debug" if self.debug else "Release"

        # CMake arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir / 'tensorrt_edgellm'}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_PYTHON_BINDINGS=ON",
        ]

        # Get TensorRT path from environment or use default
        trt_package_dir = os.environ.get("TRT_PACKAGE_DIR")
        if trt_package_dir:
            cmake_args.append(f"-DTRT_PACKAGE_DIR={trt_package_dir}")
        else:
            # Try common TensorRT locations
            common_trt_paths = [
                "/usr/local/tensorrt",
                "/opt/nvidia/tensorrt",
                "/usr/local/TensorRT",
            ]
            for path in common_trt_paths:
                if os.path.exists(path):
                    cmake_args.append(f"-DTRT_PACKAGE_DIR={path}")
                    break
            else:
                raise RuntimeError(
                    "TensorRT not found. Please set TRT_PACKAGE_DIR environment variable "
                    "to the path of your TensorRT installation.")

        # CUDA path
        cuda_dir = os.environ.get("CUDA_DIR")
        if cuda_dir:
            cmake_args.append(f"-DCUDA_DIR={cuda_dir}")

        # CUDA version
        cuda_version = os.environ.get("CUDA_VERSION")
        if cuda_version:
            cmake_args.append(f"-DCUDA_VERSION={cuda_version}")

        # Enable NVTX profiling if requested
        if os.environ.get("ENABLE_NVTX_PROFILING",
                          "").lower() in ("1", "true", "on"):
            cmake_args.append("-DENABLE_NVTX_PROFILING=ON")

        # Build arguments
        build_args = ["--config", cfg]

        # Use parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # Use all available CPUs
            import multiprocessing
            build_args += ["-j", str(multiprocessing.cpu_count())]

        # Create build directory
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # Run CMake configure
        print(
            f"Running CMake configure: cmake {ext.sourcedir} {' '.join(cmake_args)}"
        )
        subprocess.run(["cmake", ext.sourcedir, *cmake_args],
                       cwd=build_temp,
                       check=True)

        # Run CMake build
        print(f"Running CMake build: cmake --build . {' '.join(build_args)}")
        subprocess.run(["cmake", "--build", ".", *build_args],
                       cwd=build_temp,
                       check=True)


# Read version from version.py
def get_version():
    version_file = Path(__file__).parent / "tensorrt_edgellm" / "version.py"
    if version_file.exists():
        with open(version_file) as f:
            content = f.read()
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']',
                              content)
            if match:
                return match.group(1)
    return "0.5.0"  # Default version


setup(
    name="tensorrt_edgellm",
    version=get_version(),
    author="NVIDIA Corporation",
    author_email="",
    description="TensorRT Edge LLM Python Runtime",
    long_description=open("README.md").read()
    if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("_edgellm_runtime")],
    cmdclass={"build_ext": CMakeBuild},
    packages=["tensorrt_edgellm"],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
    zip_safe=False,
)
