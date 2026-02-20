#!/bin/bash

# Dynamic Eagle engine build script
# Supports w4a4 (nvfp4) precision only
# Supports Offline and SingleStream scenarios with different batch sizes
#
# Usage:
#   ./engine_build.sh Offline      # Build W4A4 (nvfp4) engines for Offline (bs=8)
#   ./engine_build.sh SingleStream # Build W4A4 (nvfp4) engines for SingleStream (bs=1)
#
# Environment Variables (optional):
#   WORKSPACE_DIR: Workspace directory containing ONNX models (default: $HOME/tensorrt-edgellm-workspace)
#   MODEL_NAME: Model name directory (default: Llama-3.1-8B-Instruct)
#   BASE_ONNX_DIR: Override base ONNX directory path
#   DRAFT_ONNX_DIR: Override draft ONNX directory path
#
# Example with custom paths:
#   export WORKSPACE_DIR=$HOME/my-workspace
#   export MODEL_NAME=Llama-3.1-8B-Instruct
#   ./engine_build.sh Offline

set -e

# Set precision to w4a4 (only supported precision)
PRECISION=w4a4

# Check scenario argument
SCENARIO=${1:-Offline}

if [ "$SCENARIO" != "Offline" ] && [ "$SCENARIO" != "SingleStream" ]; then
    echo "ERROR: Invalid scenario. Must be 'Offline' or 'SingleStream'"
    echo "Usage: $0 [Offline|SingleStream]"
    exit 1
fi

# Set batch size and Eagle parameters based on scenario
if [ "$SCENARIO" == "Offline" ]; then
    BATCH_SIZE=8
    EAGLE_DRAFT_TOP_K=4
    EAGLE_DRAFT_STEP=3
    EAGLE_VERIFY_TREE_SIZE=12
elif [ "$SCENARIO" == "SingleStream" ]; then
    BATCH_SIZE=1
    EAGLE_DRAFT_TOP_K=10
    EAGLE_DRAFT_STEP=6
    EAGLE_VERIFY_TREE_SIZE=60
fi

# Calculate maxDraftTreeSize: 1 + draftTopK + (draftStep - 1) * draftTopK * draftTopK
MAX_DRAFT_TREE_SIZE=$EAGLE_VERIFY_TREE_SIZE
MAX_VERIFY_TREE_SIZE=$EAGLE_VERIFY_TREE_SIZE

# Set workspace directory (can be overridden via environment variable)
WORKSPACE_DIR=${WORKSPACE_DIR:-$HOME/tensorrt-edgellm-workspace}
MODEL_NAME=${MODEL_NAME:-Llama-3.1-8B-Instruct}

# Set paths for w4a4 (nvfp4) precision
BASE_ONNX_DIR=${BASE_ONNX_DIR:-$WORKSPACE_DIR/$MODEL_NAME/llm-base-nvfp4-nvfp4}
DRAFT_ONNX_DIR=${DRAFT_ONNX_DIR:-$WORKSPACE_DIR/$MODEL_NAME/draft-eagle3-nvfp4-nvfp4}
ENGINE_DIR=$HOME/tensorrt-edge-llm/mlperf/engines_nvfp4_bs${BATCH_SIZE}
PRECISION_NAME="W4A4 (nvfp4)"

# Create engine directory if it doesn't exist
mkdir -p $ENGINE_DIR

echo "=========================================="
echo "Building $PRECISION_NAME Eagle Engines"
echo "=========================================="
echo "Precision: $PRECISION"
echo "Scenario: $SCENARIO"
echo "Batch Size: $BATCH_SIZE"
echo "Eagle Draft Top-K: $EAGLE_DRAFT_TOP_K"
echo "Eagle Draft Step: $EAGLE_DRAFT_STEP"
echo "Eagle Verify Tree Size: $EAGLE_VERIFY_TREE_SIZE"
echo "Max Draft Tree Size: $MAX_DRAFT_TREE_SIZE"
echo "Max Verify Tree Size: $MAX_VERIFY_TREE_SIZE"
echo "Base ONNX: $BASE_ONNX_DIR"
echo "Draft ONNX: $DRAFT_ONNX_DIR"
echo "Engine Dir: $ENGINE_DIR"
echo ""

# Verify ONNX directories exist
if [ ! -d "$BASE_ONNX_DIR" ]; then
    echo "ERROR: Base ONNX directory not found: $BASE_ONNX_DIR"
    exit 1
fi

if [ ! -d "$DRAFT_ONNX_DIR" ]; then
    echo "ERROR: Draft ONNX directory not found: $DRAFT_ONNX_DIR"
    exit 1
fi

# Build Eagle base engine
echo "Building Eagle base engine ($PRECISION_NAME)..."
export EDGELLM_PLUGIN_PATH=$HOME/tensorrt-edge-llm/build/libNvInfer_edgellm_plugin.so
$HOME/tensorrt-edge-llm/build/examples/llm/llm_build \
    --onnxDir $BASE_ONNX_DIR \
    --engineDir $ENGINE_DIR \
    --maxBatchSize=$BATCH_SIZE \
    --maxKVCacheCapacity=2700 \
    --maxInputLen=2560 \
    --maxVerifyTreeSize=$MAX_VERIFY_TREE_SIZE \
    --debug \
    --eagleBase

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build Eagle base engine"
    exit 1
fi

echo ""
echo "Base engine build complete!"
echo ""

# Build Eagle draft engine
echo "Building Eagle draft engine ($PRECISION_NAME)..."
$HOME/tensorrt-edge-llm/build/examples/llm/llm_build \
    --onnxDir $DRAFT_ONNX_DIR \
    --engineDir $ENGINE_DIR/ \
    --maxBatchSize=$BATCH_SIZE \
    --maxInputLen=2560 \
    --maxKVCacheCapacity=2700 \
    --maxDraftTreeSize=$MAX_DRAFT_TREE_SIZE \
    --eagleDraft \
    --debug

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build Eagle draft engine"
    exit 1
fi

echo ""
echo "=========================================="
echo "$PRECISION_NAME Engine build complete!"
echo "Engines are in: $ENGINE_DIR"
echo "=========================================="
