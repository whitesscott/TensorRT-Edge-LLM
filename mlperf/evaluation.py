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

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import evaluate
import nltk
import numpy as np
from dataset import Dataset
from transformers import AutoTokenizer


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file",
                        required=True,
                        help="path to mlperf_log_accuracy.json")
    parser.add_argument("--dataset-file",
                        required=True,
                        help="path to cnn_eval.json")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="verbose messages")
    parser.add_argument(
        "--dtype",
        default="int32",
        help="dtype of the accuracy log",
        choices=["int32", "int64"],
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get(
            "MODEL_PATH",
            os.path.expanduser("~/llm-models/Llama-3.1-8B-Instruct")),
        help="Model path (can also be set via MODEL_PATH environment variable)"
    )
    parser.add_argument(
        "--total-sample-count",
        type=int,
        default=None,
        help=
        "Total number of samples to use from the dataset. Useful when dataset file and output file lengths don't match."
    )
    args = parser.parse_args()
    return args


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def main():

    args = get_args()
    model_name = args.model_name
    dataset_path = args.dataset_file

    # Validate dataset file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' does not exist")
        return 1

    # Load dataset to get the actual sample count
    print(f"Loading dataset from {dataset_path}...")
    # Use provided total_sample_count if specified, otherwise load all samples
    dataset_sample_count = args.total_sample_count if args.total_sample_count is not None else 999999999
    data_object = Dataset(
        model_name=args.model_name,
        dataset_path=dataset_path,
        total_sample_count=dataset_sample_count,
    )
    total_sample_count = len(data_object.input_ids)
    if args.total_sample_count is not None:
        print(
            f"Dataset loaded with {total_sample_count} samples (limited to {args.total_sample_count})"
        )
    else:
        print(f"Dataset contains {total_sample_count} samples")

    metric = evaluate.load("rouge")
    nltk.download("punkt", quiet=True)
    nltk.download('punkt_tab', quiet=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    targets = data_object.targets

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    # Deduplicate the results loaded from the json
    dedup_results = []
    seen = set()
    for result in results:
        item = result["qsl_idx"]
        if item not in seen:
            seen.add(item)
            dedup_results.append(result)
    results = dedup_results

    target_required = []
    preds_token_ids = []

    eval_dtype = np.int32
    if args.dtype == "int64":
        eval_dtype = np.int64

    for pred in results:
        qsl_idx = pred["qsl_idx"]
        if qsl_idx >= len(targets):
            print(
                f"Warning: qsl_idx {qsl_idx} is out of range (dataset has {len(targets)} samples). Skipping."
            )
            continue
        target = targets[qsl_idx]
        target_required.append(target)

        # Use token_count to properly size the array
        hex_data = pred["data"]
        token_count = pred.get(
            "token_count",
            len(bytes.fromhex(hex_data)) // np.dtype(eval_dtype).itemsize)

        # Convert hex to bytes and ensure proper sizing
        byte_data = bytes.fromhex(hex_data)
        expected_bytes = token_count * np.dtype(eval_dtype).itemsize

        # Truncate or pad if necessary
        if len(byte_data) > expected_bytes:
            byte_data = byte_data[:expected_bytes]
        elif len(byte_data) < expected_bytes:
            # Pad with zeros if needed (shouldn't happen normally)
            byte_data = byte_data + b'\x00' * (expected_bytes - len(byte_data))

        preds_token_ids.append(np.frombuffer(byte_data, dtype=eval_dtype))

    preds_decoded_text = tokenizer.batch_decode(preds_token_ids,
                                                skip_special_tokens=True)

    preds, targets = postprocess_text(preds_decoded_text, target_required)

    result = metric.compute(predictions=preds,
                            references=targets,
                            use_stemmer=True,
                            use_aggregator=False)
    result = {k: f"{round(np.mean(v) * 100, 4)}" for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = int(np.sum(prediction_lens))
    result["gen_num"] = len(preds)
    print("\nResults\n")
    print(result)


if __name__ == "__main__":
    main()
