#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build script for TensorRT Edge LLM Python bindings (Jetson Thor)
#
# Usage:
#   ./build_pybind.sh [--clean] [--jobs N]

set -e

# Default values for Jetson Thor
BUILD_TYPE="Release"
TRT_DIR="/usr"
CLEAN_BUILD=false
NUM_JOBS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./build_pybind.sh [--clean] [--jobs N]"
            echo ""
            echo "Options:"
            echo "  --clean    Clean build directory before building"
            echo "  --jobs N   Number of parallel jobs (default: auto)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Determine number of jobs
if [ -z "$NUM_JOBS" ]; then
    NUM_JOBS=$(nproc 2>/dev/null || echo 4)
fi

echo "=============================================="
echo "Building TensorRT Edge LLM Python Bindings"
echo "=============================================="
echo "Target: Jetson Thor"
echo "TensorRT: $TRT_DIR"
echo "Build type: $BUILD_TYPE"
echo "Parallel jobs: $NUM_JOBS"
echo "=============================================="

# Run CMake configure
cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-thor \
    -DTRT_PACKAGE_DIR="$TRT_DIR" \
    -DBUILD_PYTHON_BINDINGS=ON

# Run CMake build
cmake --build . -j "$NUM_JOBS"

# Copy the built module to the Python package directory
MODULE_PATH=$(find "$BUILD_DIR" -name "_edgellm_runtime*.so" -type f | head -1)
if [ -n "$MODULE_PATH" ]; then
    cp "$MODULE_PATH" "${SCRIPT_DIR}/tensorrt_edgellm/"
    echo ""
    echo "=============================================="
    echo "Build successful!"
    echo "Module installed to: ${SCRIPT_DIR}/tensorrt_edgellm/"
    echo ""
    echo "To use the Python bindings:"
    echo "  export PYTHONPATH=${SCRIPT_DIR}:\$PYTHONPATH"
    echo "  python -c 'from tensorrt_edgellm.runtime import LLMRuntime; print(\"OK\")'"
    echo "=============================================="
else
    echo "ERROR: Built module not found"
    exit 1
fi
