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

import json
import logging
import os

import numpy as np
from transformers import AutoTokenizer

log = logging.getLogger("Llama-8B-Dataset")


class Dataset:

    def __init__(
        self,
        model_name=None,
        total_sample_count=13368,
        perf_count_override=None,
        dataset_path=None,
    ):
        default_model_path = os.environ.get(
            "MODEL_PATH",
            os.path.expanduser("~/llm-models/Llama-3.1-8B-Instruct"))
        self.model_name = model_name or default_model_path
        self.dataset_path = dataset_path

        # self.total_sample_count = total_sample_count
        self.load_processed_dataset()

        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warn(
                "Processed pickle file {} not found. Please check that the path is correct"
                .format(self.dataset_path))

        log.info("Loading dataset...")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.processed_data = json.load(f)

        # Parse tokens directly from tok_input field (already in correct format)
        self.input = [item.get("input", "") for item in self.processed_data]
        self.input_ids = [
            item.get("tok_input", []) for item in self.processed_data
        ]
        self.input_lens = [len(x) for x in self.input_ids]
        self.targets = [item.get("output", "") for item in self.processed_data]
        del self.processed_data
        log.info("Finished loading dataset.")

    def postProcess(
        self,
        out_tokens,
        query_id_list=None,
        sample_index_list=None,
    ):
        """Postprocesses output prediction"""

        # TODO: Create response object in postProcess(?)
        """
        preds = []
        for i in range(out_tokens.shape[0]):
            #pred = out_tokens[i].reshape(-1).cpu().numpy() # Slice up to original input length as below?

            input_len = input_seq_lens[i] if input_seq_lens else 0
            pred = out_tokens[i, input_len:].reshape(-1).cpu().numpy()
            preds.append(pred)
        """
        # Everything is padded to max_len (1024), so prune the input and parse
        # to numpy
        output_seq = out_tokens
        assert len(query_id_list) == len(output_seq)

        return [np.asarray(out, dtype=np.int32) for out in output_seq]

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass
