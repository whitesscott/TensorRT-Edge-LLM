#!/bin/bash

# Consolidated MLPerf run script for all scenarios
# Supports Offline and SingleStream scenarios with w4a4 (nvfp4) precision only
#
# Usage:
#   ./run_thor.sh Offline                           # Run Offline performance with w4a4
#   ./run_thor.sh SingleStream                      # Run SingleStream performance with w4a4
#   ./run_thor.sh Offline --accuracy                # Run Offline accuracy with w4a4
#   ./run_thor.sh SingleStream --accuracy           # Run SingleStream accuracy with w4a4
#   ./run_thor.sh Offline --total-sample-count 256  # Run with custom sample count
#   ./run_thor.sh Offline --accuracy --total-sample-count 10 # Accuracy with custom count

set -e

# Set precision to w4a4 (only supported precision)
PRECISION=w4a4

# Parse arguments
SCENARIO=""
ACCURACY_FLAG=""
TOTAL_SAMPLE_COUNT=""

# Parse positional and optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        Offline|SingleStream)
            SCENARIO=$1
            shift
            ;;
        --accuracy)
            ACCURACY_FLAG="--accuracy"
            shift
            ;;
        --total-sample-count)
            TOTAL_SAMPLE_COUNT=$2
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Usage: $0 [Offline|SingleStream] [--accuracy] [--total-sample-count N]"
            exit 1
            ;;
    esac
done

# Check scenario (required)
if [ -z "$SCENARIO" ]; then
    echo "ERROR: Scenario is required. Must be 'Offline' or 'SingleStream'"
    echo "Usage: $0 [Offline|SingleStream] [--accuracy] [--total-sample-count N]"
    exit 1
fi

if [ "$SCENARIO" != "Offline" ] && [ "$SCENARIO" != "SingleStream" ]; then
    echo "ERROR: Invalid scenario. Must be 'Offline' or 'SingleStream'"
    echo "Usage: $0 [Offline|SingleStream] [--accuracy] [--total-sample-count N]"
    exit 1
fi

# Set default sample count if not provided
if [ -z "$TOTAL_SAMPLE_COUNT" ]; then
    TOTAL_SAMPLE_COUNT=5000
fi

if [ -n "$ACCURACY_FLAG" ]; then
    echo "Running in ACCURACY mode"
else
    echo "Running in PERFORMANCE mode"
fi

# Set batch size based on scenario (used only for engine directory path)
# Note: batch_size is automatically determined by SUT class based on scenario
# (8 for Offline, 1 for SingleStream) - not passed as argument
if [ "$SCENARIO" == "Offline" ]; then
    BATCH_SIZE=8
elif [ "$SCENARIO" == "SingleStream" ]; then
    BATCH_SIZE=1
fi

# Set engine directory for w4a4 (nvfp4) precision
ENGINE_DIR=$HOME/tensorrt-edge-llm/mlperf/engines_nvfp4_bs${BATCH_SIZE}

# Always use the 5000 sample dataset file
DATASET_PATH=$HOME/tensorrt-edge-llm/mlperf/sample_cnn_eval_5000.json

# Set output directory based on scenario and accuracy mode
if [ -n "$ACCURACY_FLAG" ]; then
    if [ "$SCENARIO" == "Offline" ]; then
        OUTPUT_DIR=$HOME/tensorrt-edge-llm/mlperf/output-logs-accuracy-offline
    elif [ "$SCENARIO" == "SingleStream" ]; then
        OUTPUT_DIR=$HOME/tensorrt-edge-llm/mlperf/output-logs-accuracy-singlestream
    fi
else
    if [ "$SCENARIO" == "Offline" ]; then
        OUTPUT_DIR=$HOME/tensorrt-edge-llm/mlperf/output-logs-offline
    elif [ "$SCENARIO" == "SingleStream" ]; then
        OUTPUT_DIR=$HOME/tensorrt-edge-llm/mlperf/output-logs-singlestream
    fi
fi

# Set Edge LLM plugin path
export PYTHONPATH=$HOME/tensorrt-edge-llm:$PYTHONPATH
export EDGELLM_PLUGIN_PATH=$HOME/tensorrt-edge-llm/build/libNvInfer_edgellm_plugin.so

# Set model path (can be overridden via environment variable)
MODEL_PATH=${MODEL_PATH:-$HOME/llm-models/Llama-3.1-8B-Instruct}

echo "=========================================="
echo "MLPerf Run Configuration"
echo "=========================================="
echo "Scenario: $SCENARIO"
echo "Precision: $PRECISION"
echo "Batch Size: $BATCH_SIZE (auto-determined from scenario)"
echo "Engine Dir: $ENGINE_DIR"
echo "Model Path: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Total Sample Count: $TOTAL_SAMPLE_COUNT"
echo "Output Dir: $OUTPUT_DIR"
echo "Mode: $([ -n "$ACCURACY_FLAG" ] && echo "Accuracy" || echo "Performance")"
echo "=========================================="
echo ""

# Verify engine directory exists
if [ ! -d "$ENGINE_DIR" ]; then
    echo "ERROR: Engine directory not found: $ENGINE_DIR"
    echo "Please build engines first using: ./engine_build.sh $SCENARIO"
    exit 1
fi

# Verify dataset file exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Run the benchmark
python main.py --scenario $SCENARIO \
    --model-path=$MODEL_PATH \
    --dataset-path $DATASET_PATH \
    --user-conf=user.conf \
    --output-log-dir=$OUTPUT_DIR \
    --enable-log-trace \
    --total-sample-count $TOTAL_SAMPLE_COUNT \
    --token-output-file=$OUTPUT_DIR/tokens_output.json \
    --engine-dir=$ENGINE_DIR \
    $ACCURACY_FLAG
