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
import queue
import signal
import sys
import threading
from typing import Generator, List

import mlperf_loadgen as lg
import numpy as np
from dataset import Dataset

log = logging.getLogger("Llama-8B-SUT")

# Import TensorRT Edge LLM runtime (Eagle spec decode runtime)
try:
    from tensorrt_edgellm.runtime import EagleRuntime
except ImportError:
    # Fallback path if not installed as package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
    from tensorrt_edgellm.runtime import EagleRuntime


def complete_loadgen_request(request_id: int,
                             output_tokens: List[int],
                             is_first_token: bool = False):
    complete_fn = lg.FirstTokenComplete if is_first_token else lg.QuerySamplesComplete
    output_tokens_t = np.ascontiguousarray(output_tokens, dtype=np.uint32)
    output_seq_len = len(output_tokens_t)
    output_toks_ptr = output_tokens_t.ctypes.data
    output_toks_size = output_seq_len * output_tokens_t.itemsize
    complete_fn([
        lg.QuerySampleResponse(request_id, output_toks_ptr, output_toks_size,
                               output_seq_len)
    ])


class SUT:
    # Maximum number of output tokens to generate
    MAX_OUTPUT_TOKENS = 128

    def __init__(self,
                 model_path,
                 dataset_path,
                 scenario,
                 engine_dir,
                 total_sample_count=5000,
                 token_output_file="tokens_output.json",
                 eagle_draft_top_k=10,
                 eagle_draft_step=6,
                 eagle_verify_tree_size=60,
                 warmup_count=10):

        self.model_path = model_path
        self.engine_dir = engine_dir

        # Store Eagle parameters
        self.eagle_draft_top_k = eagle_draft_top_k
        self.eagle_draft_step = eagle_draft_step
        self.eagle_verify_tree_size = eagle_verify_tree_size

        # Store warmup count
        self.warmup_count = warmup_count

        # Set batch size based on scenario
        if scenario == "offline":
            batch_size = 8
        elif scenario == "singlestream":
            batch_size = 1
        else:
            batch_size = 1
        self.batch_size = batch_size

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )
        self.load_model()

        self.worker_thread = None
        self.query_queue = queue.Queue()
        self.scenario = scenario
        self.token_output_file = token_output_file

        # Synchronization for proper shutdown coordination
        self.queries_issued = 0
        self.issue_complete = False
        self.sync_lock = threading.Lock()

        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.count = 0

        # Create a fresh JSON file at the beginning
        self._initialize_token_file()

    def _initialize_token_file(self):
        """Initialize a fresh JSONL file for token logging"""
        try:
            with open(self.token_output_file, 'w', encoding='utf-8') as f:
                pass  # Create empty file
            log.info(
                f"Initialized fresh token output file: {self.token_output_file}"
            )
        except Exception as e:
            log.error(f"Failed to initialize token output file: {e}")

    def log_tokens_to_json(self, query_ids, input_ids, output_ids, input_text,
                           output_text):
        """
        Log tokens and text to JSONL file (one JSON object per line).
        
        Args:
            query_ids: Query sample library ID
            input_ids: List of input token IDs
            output_ids: List of output token IDs  
            input_text: Detokenized input text
            output_text: Detokenized output text
        """
        try:
            # Create the data entry
            entry = {
                "qsl_idx": query_ids,
                "input_ids": input_ids,
                "output_ids": output_ids,
                "input_text": input_text,
                "output_text": output_text
            }

            # Append as a single line (JSONL format) - much more efficient!
            with open(self.token_output_file, 'a', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')  # Add newline for JSONL format

            log.debug(
                f"Logged tokens to JSONL - Query ID: {query_ids}, Output tokens: {len(output_ids)}"
            )

        except Exception as e:
            log.error(f"Failed to log tokens to JSONL: {e}")

    def _sort_jsonl_by_qsl_index(self):
        """Sort the JSONL file by qsl_index after all processing is complete"""
        try:
            # Read all entries from the JSONL file
            entries = []
            if os.path.exists(self.token_output_file):
                with open(self.token_output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                entry = json.loads(line)
                                entries.append(entry)
                            except json.JSONDecodeError as e:
                                log.warning(
                                    f"Skipping invalid JSON line: {line[:100]}... Error: {e}"
                                )

                # Sort entries by qsl_index
                entries.sort(key=lambda x: x.get('qsl_idx', 0))

                # Write sorted entries back to the file
                with open(self.token_output_file, 'w', encoding='utf-8') as f:
                    for entry in entries:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write('\n')

                log.info(
                    f"Successfully sorted {len(entries)} entries in {self.token_output_file} by qsl_idx"
                )
            else:
                log.warning(
                    f"Token output file {self.token_output_file} does not exist, skipping sort"
                )

        except Exception as e:
            log.error(f"Failed to sort JSONL file by qsl_idx: {e}")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals"""
        log.info(f"Received signal {signum}. Initiating graceful shutdown...")
        # Raise KeyboardInterrupt to be caught by main thread
        raise KeyboardInterrupt("Signal received")

    def warmup(self):
        """Perform warmup runs to initialize CUDA kernels and runtime components"""
        if self.warmup_count <= 0:
            return

        log.info(f"Starting warmup with {self.warmup_count} runs...")

        # Get a sample input for warmup (use first sample from dataset)
        if len(self.data_object.input_ids) == 0:
            log.warning("No samples available for warmup, skipping...")
            return

        warmup_input_ids = self.data_object.input_ids[0]
        if not isinstance(warmup_input_ids, list):
            warmup_input_ids = list(warmup_input_ids)

        # Perform warmup runs
        for warmup_run in range(self.warmup_count):
            try:
                if self.scenario == "offline":
                    # For offline, use batch processing
                    batch_input_ids = [warmup_input_ids] * min(
                        self.batch_size, len(self.data_object.input_ids))
                    response = self.runtime._runtime.handle_request_with_tokens(
                        batch_input_ids,
                        temperature=0,
                        top_p=1.0,
                        top_k=1,
                        max_generate_length=self.MAX_OUTPUT_TOKENS,
                        enable_spec_decode=True)
                else:
                    # For singlestream, use single request
                    batched_input_token_ids = [warmup_input_ids]
                    response = self.runtime._runtime.handle_request_with_tokens(
                        batched_input_token_ids,
                        temperature=0,
                        top_p=1.0,
                        top_k=1,
                        max_generate_length=self.MAX_OUTPUT_TOKENS,
                        enable_spec_decode=True)

                log.debug(
                    f"Warmup run {warmup_run + 1}/{self.warmup_count} completed"
                )
            except Exception as e:
                log.error(
                    f"Warmup run {warmup_run + 1}/{self.warmup_count} failed: {e}"
                )
                # Continue with other warmup runs even if one fails

        log.info(f"Warmup of {self.warmup_count} runs completed")

    def start(self):
        # Perform warmup before starting worker threads
        self.warmup()

        # Create worker threads
        if self.scenario == "offline":
            self.worker_thread = threading.Thread(
                target=self.process_queries_offline)
        elif self.scenario == "singlestream":
            self.worker_thread = threading.Thread(
                target=self.process_queries_singlestream)
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}")
        self.worker_thread.start()

    def stop(self):
        log.info("Stopping SUT...")

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                self.query_queue.put(None)
                log.warning("Worker thread did not stop within timeout")
        self._sort_jsonl_by_qsl_index()

    def _handle_request(
            self, input_token_ids: List[int],
            max_generate_length: int) -> Generator[List[int], None, None]:
        """
        Async token-based request handling that yields tokens as they're generated.
        
        Uses real streaming with C++ runtime callbacks and threading for true async token-by-token streaming.
        The generation runs in a background thread while tokens are yielded immediately as they arrive.
        
        Args:
            input_token_ids: List of input token IDs
            max_generate_length: Maximum number of tokens to generate
            
        Yields:
            Lists of newly generated token IDs (yielded as soon as they're generated)
        """
        # Use token-based API directly with Eagle spec decode and streaming callback
        batched_input_token_ids = [input_token_ids]

        # Queue to pass tokens from callback (background thread) to generator (main thread)
        token_queue = queue.Queue()
        generation_done = threading.Event()
        generation_error = [
            None
        ]  # Use list to allow modification in nested function

        def token_callback(batch_index: int, token_id: int,
                           is_first_token: bool) -> bool:
            """Callback function called for each generated token from C++ runtime"""
            # Put token in queue for immediate yielding (non-blocking with timeout to avoid deadlock)
            try:
                token_queue.put((token_id, is_first_token), timeout=0.1)
            except queue.Full:
                # Queue full - shouldn't happen with reasonable generation speed, but handle gracefully
                log.warning("Token queue full, skipping token")
            return True  # Continue generation

        def run_generation():
            """Run generation in background thread"""
            try:
                response = self.runtime._runtime.handle_request_with_tokens(
                    batched_input_token_ids,
                    temperature=0,
                    top_p=1.0,
                    top_k=1,
                    max_generate_length=max_generate_length,
                    enable_spec_decode=True,
                    token_callback=token_callback)
            except Exception as e:
                generation_error[0] = e
            finally:
                # Signal that generation is complete
                generation_done.set()
                # Put sentinel value to wake up generator if it's waiting
                try:
                    token_queue.put(None, timeout=0.1)
                except queue.Full:
                    pass

        # Start generation in background thread
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()

        # Yield tokens as they arrive from the callback
        first_token_yielded = False
        while True:
            try:
                # Wait for token with timeout to periodically check if generation is done
                item = token_queue.get(timeout=0.1)

                # Sentinel value indicates generation complete
                if item is None:
                    break

                token_id, is_first_token = item

                # Yield first token immediately when it arrives
                if is_first_token and not first_token_yielded:
                    yield [token_id]
                    first_token_yielded = True
                elif not is_first_token:
                    # Yield subsequent tokens
                    yield [token_id]

            except queue.Empty:
                # Check if generation is done (might have finished while queue was empty)
                if generation_done.is_set():
                    # Drain any remaining tokens
                    while True:
                        try:
                            item = token_queue.get_nowait()
                            if item is None:
                                break
                            token_id, is_first_token = item
                            if is_first_token and not first_token_yielded:
                                yield [token_id]
                                first_token_yielded = True
                            elif not is_first_token:
                                yield [token_id]
                        except queue.Empty:
                            break
                    break
                # Continue waiting for tokens
                continue

        # Wait for thread to complete and check for errors
        gen_thread.join(timeout=5.0)
        if generation_error[0]:
            raise generation_error[0]

    def process_queries_offline(self):
        """Process queries in offline mode with batching"""
        while True:
            try:
                qitem = self.query_queue.get(timeout=0.05)
                if qitem is None:
                    log.info(
                        "Received shutdown signal (None), breaking immediately"
                    )
                    break
            except queue.Empty:
                continue

            # Process batch
            batch_input_ids = []
            batch_query_ids = []
            batch_request_ids = []

            for query_sample in qitem:
                query_idx = query_sample.index
                # tok_input is already a list, no conversion needed
                input_token_ids = self.data_object.input_ids[query_idx]
                if not isinstance(input_token_ids, list):
                    input_token_ids = list(input_token_ids)
                batch_input_ids.append(input_token_ids)
                batch_query_ids.append(query_idx)
                batch_request_ids.append(query_sample.id)

            # Process batch with token-based API (token-in, token-out) with Eagle spec decode
            try:
                # Use token-based API directly with Eagle spec decode - no tokenization/untokenization
                response = self.runtime._runtime.handle_request_with_tokens(
                    batch_input_ids,
                    temperature=0,
                    top_p=1.0,
                    top_k=1,
                    max_generate_length=self.MAX_OUTPUT_TOKENS,
                    enable_spec_decode=True)

                # Complete each request
                for i, (query_id, request_id) in enumerate(
                        zip(batch_query_ids, batch_request_ids)):
                    if i < len(response.output_ids):
                        output_tokens = response.output_ids[i]
                        output_tokens_list = output_tokens.tolist() if hasattr(
                            output_tokens, 'tolist') else list(output_tokens)

                        # Log tokens (decode only for logging, not for API)
                        input_text = self.data_object.tokenizer.decode(
                            batch_input_ids[i]) if hasattr(
                                self.data_object, 'tokenizer') else ""
                        output_text = self.data_object.tokenizer.decode(
                            output_tokens_list
                        ) if output_tokens_list and hasattr(
                            self.data_object, 'tokenizer') else ""
                        self.log_tokens_to_json(query_ids=query_id,
                                                input_ids=batch_input_ids[i],
                                                output_ids=output_tokens_list,
                                                input_text=input_text,
                                                output_text=output_text)

                        # Complete loadgen request
                        complete_loadgen_request(request_id,
                                                 output_tokens_list)
                    else:
                        log.warning(f"No output tokens for query {query_id}")

            except Exception as e:
                log.error(f"Error processing batch: {e}")
                # Complete requests with empty tokens on error
                for request_id in batch_request_ids:
                    complete_loadgen_request(request_id, [])

    def process_queries_singlestream(self):
        """Process queries in singlestream mode (one at a time)"""
        while True:
            try:
                qitem = self.query_queue.get(timeout=0.05)
                if qitem is None:
                    log.info(
                        "Received shutdown signal (None), breaking immediately"
                    )
                    break
            except queue.Empty:
                continue
            assert len(qitem) == 1
            query_ids = qitem[0].index
            request_id = qitem[0].id

            # tok_input is already a list, no conversion needed
            input_token_ids_list = self.data_object.input_ids[query_ids]
            if not isinstance(input_token_ids_list, list):
                input_token_ids_list = list(input_token_ids_list)

            # Stream tokens as they're generated (always async)
            try:
                first_token_sent = False
                all_tokens = []

                for token_chunk in self._handle_request(
                        input_token_ids_list, self.MAX_OUTPUT_TOKENS):
                    if token_chunk:
                        all_tokens.extend(token_chunk)

                        # Send first token immediately
                        if not first_token_sent and len(token_chunk) > 0:
                            complete_loadgen_request(request_id,
                                                     [token_chunk[0]],
                                                     is_first_token=True)
                            first_token_sent = True

                # Send final complete response with all tokens
                if all_tokens:
                    # Log tokens (decode only for logging, not for API)
                    input_text = self.data_object.tokenizer.decode(
                        input_token_ids_list) if hasattr(
                            self.data_object, 'tokenizer') else ""
                    output_text = self.data_object.tokenizer.decode(
                        all_tokens) if all_tokens and hasattr(
                            self.data_object, 'tokenizer') else ""
                    self.log_tokens_to_json(query_ids=query_ids,
                                            input_ids=input_token_ids_list,
                                            output_ids=all_tokens,
                                            input_text=input_text,
                                            output_text=output_text)
                    complete_loadgen_request(request_id,
                                             all_tokens,
                                             is_first_token=False)
                else:
                    log.warning(f"No tokens generated for query {query_ids}")
                    complete_loadgen_request(request_id, [],
                                             is_first_token=False)
            except Exception as e:
                log.error(f"Error processing query {query_ids}: {e}")
                complete_loadgen_request(request_id, [], is_first_token=False)

    def load_model(self):
        log.info("Loading model...")

        # Initialize TensorRT Edge LLM Eagle spec decode runtime
        self.runtime = EagleRuntime(
            engine_dir=self.engine_dir,
            multimodal_engine_dir="",
            draft_top_k=self.eagle_draft_top_k,
            draft_step=self.eagle_draft_step,
            verify_tree_size=self.eagle_verify_tree_size,
            capture_cuda_graph=True)

        log.info(
            f"Loaded model with Eagle speculative decoding: draft_top_k={self.eagle_draft_top_k}, draft_step={self.eagle_draft_step}, verify_tree_size={self.eagle_verify_tree_size}"
        )

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        log.info(f"IssueQuery started with {len(query_samples)} samples")

        with self.sync_lock:
            self.queries_issued = 0
            self.issue_complete = False

        while len(query_samples) > 0:
            batch = query_samples[:self.batch_size]
            self.query_queue.put(batch)

            with self.sync_lock:
                self.queries_issued += len(batch)

            query_samples = query_samples[self.batch_size:]

        log.info(
            f"IssueQuery done - issued {self.queries_issued} total queries")
        self.count += 1
        with self.sync_lock:
            self.issue_complete = True

    def flush_queries(self):
        log.info("FlushQueries started")
        # Don't put None here - let process_queries finish naturally
        # None is only for final shutdown in stop()
