# Thor Workflow for MLPerf Inference

Complete workflow for building and running MLPerf inference on the Thor platform with Edge-LLM integration.

## Prerequisites

- Jetpack 7.1
- Install required Python packages:

```bash
sudo apt-get install python3-venv python3-dev python3-pip
```

## Step 1: Export ONNX Models (x86 Host / Data Center GPU)

Export ONNX models on an x86 host or data center GPU (ideally ~80GB GPU memory) before building engines on Thor. This uses EAGLE speculative decoding.

> **Reference**: See the [LLM EAGLE Speculative Decoding example](../docs/source/developer_guide/getting-started/examples.md#example-2-llm-eagle-speculative-decoding) for details.

### Setup Export Environment

```bash
python3 -m venv edgellm-export-venv
source edgellm-export-venv/bin/activate
cd /path/to/tensorrt-edge-llm
pip install -e .
```

### Export Base and Draft Models

```bash
# Set workspace directory
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Llama-3.1-8B-Instruct
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Download base model from HuggingFace
# May need to download the model from https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# Accept the license agreement and authenticate with HuggingFace (huggingface-cli login)
# The model will be automatically downloaded when using the model_dir parameter below

# Download EAGLE draft model from HuggingFace
git clone https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
cd EAGLE3-LLaMA3.1-Instruct-8B && git lfs pull && cd ..

# Quantize base model with NVFP4 (both model and lm_head)
tensorrt-edgellm-quantize-llm \
  --model_dir meta-llama/Llama-3.1-8B-Instruct \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4 \
  --output_dir $MODEL_NAME/quantized-base

# Export base model with EAGLE flag
tensorrt-edgellm-export-llm \
  --model_dir $MODEL_NAME/quantized-base \
  --output_dir $MODEL_NAME/llm-base-nvfp4-nvfp4 \
  --is_eagle_base

# Quantize draft model with NVFP4 (both model and lm_head)
tensorrt-edgellm-quantize-draft \
  --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
  --draft_model_dir EAGLE3-LLaMA3.1-Instruct-8B \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4 \
  --output_dir $MODEL_NAME/quantized-draft

# Export draft model
tensorrt-edgellm-export-draft \
  --draft_model_dir $MODEL_NAME/quantized-draft \
  --base_model_dir meta-llama/Llama-3.1-8B-Instruct \
  --output_dir $MODEL_NAME/draft-eagle3-nvfp4-nvfp4
```

### Transfer ONNX Models to Thor Device

```bash
scp -r $WORKSPACE_DIR/$MODEL_NAME/llm-base-nvfp4-nvfp4 \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
scp -r $WORKSPACE_DIR/$MODEL_NAME/draft-eagle3-nvfp4-nvfp4 \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

**Note**: Ensure ONNX models are accessible on Thor before building engines. Configure paths via environment variables (see Step 11).

## Step 2: Install Jetpack

Install Jetpack 7.1 on your Thor platform following the [official NVIDIA Jetpack installation guide](https://developer.nvidia.com/embedded/jetpack).

## Step 3: Set Up Python Virtual Environment

```bash
python3 -m venv mlperf-venv
source mlperf-venv/bin/activate
```

## Step 4: Install MLPerf LoadGen

```bash
git clone https://github.com/mlcommons/inference
cd inference/loadgen
pip install .
cd ../..
```

## Step 5: Clone Edge-LLM Repository

```bash
cd ~
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git tensorrt-edge-llm
cd tensorrt-edge-llm
git submodule update --init --recursive
```

## Step 6: Install MLPerf Dependencies

```bash
cd ~/tensorrt-edge-llm/mlperf
pip install -r requirements.txt
```

## Step 7: Build Edge-LLM Project

```bash
cd ~/tensorrt-edge-llm
./build.sh
```

## Step 8: Set PYTHONPATH

```bash
export PYTHONPATH=~/tensorrt-edge-llm:$PYTHONPATH
```

**Note**: Add this to your `~/.bashrc` or `~/.profile` to make it persistent across sessions.

## Step 9: Download Dataset

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-sample-cnn-eval-5000.uri
```

**Note**: Download the model and tokenizer from [Hugging Face Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) if not already available. For more information, see the [MLPerf Inference guide](https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b).

## Step 10: Set Thor to Max Power Mode

Before building engines and running inference, configure Thor for maximum performance:

```bash
# Clear system caches
sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'

# Set to maximum performance mode
sudo nvpmodel -m 0

# Set maximum clock speeds
sudo jetson_clocks
```

**Note**: These settings persist until reboot. Re-run these commands after each system restart.

## Step 11: Build TensorRT Engine

```bash
cd ~/tensorrt-edge-llm

# Set workspace directory if different from default
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Llama-3.1-8B-Instruct

# Build engines for Offline Scenario
./mlperf/engine_build.sh Offline

# Or build for SingleStream Scenario
./mlperf/engine_build.sh SingleStream
```

**Configuration Options:**
- `WORKSPACE_DIR`: Workspace directory containing ONNX models (default: `$HOME/tensorrt-edgellm-workspace`)
- `MODEL_NAME`: Model name directory (default: `Llama-3.1-8B-Instruct`)
- `BASE_ONNX_DIR`: Override base ONNX directory path (default: `$WORKSPACE_DIR/$MODEL_NAME/llm-base-nvfp4-nvfp4`)
- `DRAFT_ONNX_DIR`: Override draft ONNX directory path (default: `$WORKSPACE_DIR/$MODEL_NAME/draft-eagle3-nvfp4-nvfp4`)
- `ENGINE_DIR`: Override engine output directory path (default: `$HOME/tensorrt-edge-llm/mlperf/engines_nvfp4_bs${BATCH_SIZE}`)

## Step 12: Run Inference Tests

Set model path (optional, defaults to `~/llm-models/Llama-3.1-8B-Instruct`):

```bash
export MODEL_PATH=$HOME/llm-models/Llama-3.1-8B-Instruct
```

### Performance Benchmarks

```bash
# Offline Performance (w4a4)
./run_thor.sh Offline

# SingleStream Performance (w4a4)
./run_thor.sh SingleStream
```

### Accuracy Evaluation

```bash
# Offline Accuracy (w4a4)
./run_thor.sh Offline --accuracy

# SingleStream Accuracy (w4a4)
./run_thor.sh SingleStream --accuracy
```

### Custom Sample Count

```bash
# Performance with custom sample count
./run_thor.sh Offline --total-sample-count 10

# Accuracy with custom sample count
./run_thor.sh Offline --accuracy --total-sample-count 10
```

**Note**: The script auto-activates `mlperf-venv` if it exists. Default sample counts: 5000 for performance and accuracy modes.

## Step 13: Evaluate Accuracy

MLPerf LoadGen generates accuracy logs:
- **Offline**: `output-logs-accuracy-offline/mlperf_log_accuracy.json`
- **SingleStream**: `output-logs-accuracy-singlestream/mlperf_log_accuracy.json`

Run evaluation:

```bash
# Set model path
export MODEL_PATH=$HOME/llm-models/Llama-3.1-8B-Instruct

# For Offline accuracy results
python evaluation.py \
    --mlperf-accuracy-file output-logs-accuracy-offline/mlperf_log_accuracy.json \
    --dataset-file=$HOME/tensorrt-edge-llm/mlperf/sample_cnn_eval_5000.json \
    --dtype int32 \
    --model-name=$MODEL_PATH

# For SingleStream accuracy results
python evaluation.py \
    --mlperf-accuracy-file output-logs-accuracy-singlestream/mlperf_log_accuracy.json \
    --dataset-file=$HOME/tensorrt-edge-llm/mlperf/sample_cnn_eval_5000.json \
    --dtype int32 \
    --model-name=$MODEL_PATH
```

The evaluation script calculates ROUGE scores by comparing model outputs against the ground truth dataset.

## Troubleshooting

- Verify all data files are downloaded and accessible
- Check CUDA 13.0 driver compatibility
- Ensure Thor hardware is properly configured
- Confirm virtual environment is activated and LoadGen is installed
- Verify Edge-LLM build completed successfully (`./build.sh`)
- Ensure engine build completed before running inference

