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
TensorRT Edge LLM Inference Example

This script demonstrates how to use the TensorRT Edge LLM Python bindings
for text generation, mirroring the functionality of examples/llm/llm_inference.cpp.

Usage:
    # Standard inference
    python llm_inference.py \\
        --engine-dir /path/to/engine \\
        --input-file /path/to/input.json \\
        --output-file /path/to/output.json

    # Eagle speculative decoding
    python llm_inference.py \\
        --engine-dir /path/to/eagle_engine \\
        --input-file /path/to/input.json \\
        --output-file /path/to/output.json \\
        --eagle

    # With profiling
    python llm_inference.py \\
        --engine-dir /path/to/engine \\
        --input-file /path/to/input.json \\
        --output-file /path/to/output.json \\
        --dump-profile
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensorrt_edgellm.runtime import (EagleRuntime, LLMGenerationResponse,
                                      LLMRuntime, parse_input_file,
                                      set_profiling_enabled)


@dataclass
class EagleConfig:
    """Configuration for Eagle speculative decoding."""
    enabled: bool = False
    draft_top_k: int = 10
    draft_step: int = 6
    verify_tree_size: int = 60


@dataclass
class InferenceArgs:
    """Command line arguments for inference."""
    engine_dir: str
    multimodal_engine_dir: str = ""
    input_file: str = ""
    output_file: str = ""
    profile_output_file: str = ""
    debug: bool = False
    dump_profile: bool = False
    warmup: int = 0
    dump_output: bool = False
    batch_size: Optional[int] = None
    max_generate_length: Optional[int] = None
    eagle: EagleConfig = None

    def __post_init__(self):
        if self.eagle is None:
            self.eagle = EagleConfig()


def parse_args() -> InferenceArgs:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TensorRT Edge LLM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument("--engine-dir",
                        required=True,
                        help="Path to engine directory")
    parser.add_argument("--multimodal-engine-dir",
                        default="",
                        help="Path to multimodal engine directory (optional)")
    parser.add_argument("--input-file",
                        required=True,
                        help="Path to input JSON file with requests")
    parser.add_argument("--output-file",
                        required=True,
                        help="Path to output JSON file")
    parser.add_argument("--profile-output-file",
                        default="",
                        help="Path to profile JSON output file (optional)")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--dump-profile",
                        action="store_true",
                        help="Dump profiling summary to console")
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup runs using the first request (default: 0)")
    parser.add_argument("--dump-output",
                        action="store_true",
                        help="Dump inference output to console")
    parser.add_argument("--batch-size",
                        type=int,
                        default=None,
                        help="Override batch size from input file")
    parser.add_argument("--max-generate-length",
                        type=int,
                        default=None,
                        help="Override max generate length from input file")
    # Eagle speculative decoding options
    parser.add_argument("--eagle",
                        action="store_true",
                        help="Enable Eagle speculative decoding mode")
    parser.add_argument(
        "--eagle-draft-top-k",
        type=int,
        default=10,
        help="Number of tokens selected per drafting step (default: 10)")
    parser.add_argument(
        "--eagle-draft-step",
        type=int,
        default=6,
        help="Number of drafting steps to perform (default: 6)")
    parser.add_argument(
        "--eagle-verify-tree-size",
        type=int,
        default=60,
        help="Number of tokens for base model verification (default: 60)")

    args = parser.parse_args()

    return InferenceArgs(engine_dir=args.engine_dir,
                         multimodal_engine_dir=args.multimodal_engine_dir,
                         input_file=args.input_file,
                         output_file=args.output_file,
                         profile_output_file=args.profile_output_file,
                         debug=args.debug,
                         dump_profile=args.dump_profile,
                         warmup=args.warmup,
                         dump_output=args.dump_output,
                         batch_size=args.batch_size,
                         max_generate_length=args.max_generate_length,
                         eagle=EagleConfig(
                             enabled=args.eagle,
                             draft_top_k=args.eagle_draft_top_k,
                             draft_step=args.eagle_draft_step,
                             verify_tree_size=args.eagle_verify_tree_size))


def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # Parse input file
    print(f"Loading input file: {args.input_file}")
    try:
        lora_weights_map, batched_requests = parse_input_file(
            args.input_file, args.batch_size, args.max_generate_length)
        print(
            f"Successfully parsed {len(lora_weights_map)} LoRA weights from input file."
        )
        print(
            f"Successfully parsed {len(batched_requests)} batches of requests from input file."
        )
    except Exception as e:
        print(f"ERROR: Failed to parse input file: {e}")
        return 1

    if not batched_requests:
        print("ERROR: No valid requests found in input file.")
        return 1

    # Create runtime based on mode
    print(f"Initializing runtime...")
    try:
        if args.eagle.enabled:
            # Eagle mode - LoRA is not supported
            if lora_weights_map:
                print(
                    "WARNING: Eagle mode does not support LoRA weights. Ignoring LoRA weights."
                )

            runtime = EagleRuntime(
                engine_dir=args.engine_dir,
                multimodal_engine_dir=args.multimodal_engine_dir,
                draft_top_k=args.eagle.draft_top_k,
                draft_step=args.eagle.draft_step,
                verify_tree_size=args.eagle.verify_tree_size,
                capture_cuda_graph=True)
            print(f"Eagle mode enabled:")
            print(f"  Draft topK: {args.eagle.draft_top_k}")
            print(f"  Draft step: {args.eagle.draft_step}")
            print(f"  Verify tree size: {args.eagle.verify_tree_size}")
        else:
            # Standard mode
            runtime = LLMRuntime(
                engine_dir=args.engine_dir,
                multimodal_engine_dir=args.multimodal_engine_dir,
                lora_weights_map=lora_weights_map,
                capture_cuda_graph=True)
    except Exception as e:
        print(f"ERROR: Failed to initialize runtime: {e}")
        return 1

    print("Runtime initialized successfully.")

    # Perform warmup runs if requested
    if args.warmup > 0:
        set_profiling_enabled(False)
        print(
            f"Starting warmup with {args.warmup} runs using the first request..."
        )
        first_request = batched_requests[0]

        for warmup_run in range(args.warmup):
            try:
                _ = runtime._runtime.handle_request(first_request)
            except Exception as e:
                print(
                    f"ERROR: Warmup run {warmup_run + 1}/{args.warmup} failed: {e}"
                )
                return 1

        print(
            f"Warmup of {args.warmup} runs completed. Starting actual benchmark runs..."
        )

    if args.dump_profile:
        set_profiling_enabled(True)

    # Structure to collect all responses for JSON export
    output_data = {"input_file": args.input_file, "responses": []}

    has_failed_request = False
    error_message = "TensorRT Edge LLM cannot handle this request. Fails."
    failed_count = 0

    # Process each request with progress indication
    print(f"Processing {len(batched_requests)} batched requests...")
    start_time = time.time()

    for request_idx, request in enumerate(batched_requests):
        # Show progress
        progress_interval = max(1, min(len(batched_requests) // 10, 100))
        if (
                request_idx + 1
        ) % progress_interval == 0 or request_idx == 0 or request_idx == len(
                batched_requests) - 1:
            progress_pct = 100.0 * (request_idx + 1) / len(batched_requests)
            print(
                f"Progress: {request_idx + 1}/{len(batched_requests)} ({progress_pct:.1f}%)"
            )

        try:
            response = runtime._runtime.handle_request(request)
            request_status = True

            # Display inference output to console if --dump-output is enabled
            if args.dump_output:
                for batch_idx, output_text in enumerate(response.output_texts):
                    print(
                        f"Response for request {request_idx} batch {batch_idx}: {output_text}"
                    )

        except Exception as e:
            # Handle failed request
            has_failed_request = True
            failed_count += 1
            request_status = False
            response = LLMGenerationResponse()
            response.output_texts = [error_message] * len(request.requests)
            response.output_ids = [[]] * len(request.requests)
            print(
                f"*** FAILED *** Request {request_idx} failed to process: {e}")

        # Add to JSON output
        for batch_idx in range(len(request.requests)):
            response_json = {
                "output_text":
                response.output_texts[batch_idx]
                if request_status else error_message,
                "request_idx":
                request_idx,
                "batch_idx":
                batch_idx,
            }

            # Store messages for reference
            messages_json = []
            for msg in request.requests[batch_idx].messages:
                msg_json = {"role": msg.role, "content": []}
                for content in msg.contents:
                    content_json = {"type": content.type}
                    if content.type == "text":
                        content_json["text"] = content.content
                    elif content.type == "image":
                        content_json["image"] = content.content
                    msg_json["content"].append(content_json)
                messages_json.append(msg_json)
            response_json["messages"] = messages_json

            # Store formatted prompts if available
            if request.formatted_requests and batch_idx < len(
                    request.formatted_requests):
                response_json[
                    "formatted_system_prompt"] = request.formatted_requests[
                        batch_idx].formatted_system_prompt
                response_json[
                    "formatted_complete_request"] = request.formatted_requests[
                        batch_idx].formatted_complete_request

            output_data["responses"].append(response_json)

    elapsed_time = time.time() - start_time

    # Final processing summary
    print(
        f"Processing complete: {len(batched_requests) - failed_count}/{len(batched_requests)} batched requests successful"
    )
    print(f"Total time: {elapsed_time:.2f} seconds")
    if failed_count > 0:
        print(f"*** {failed_count} BATCHED REQUESTS FAILED ***")

    # Dump profile if requested
    if args.dump_profile:
        set_profiling_enabled(False)
        print("\n=== Performance Summary ===")

        if args.eagle.enabled:
            prefill_metrics = runtime.prefill_metrics
            eagle_metrics = runtime.eagle_generation_metrics

            total_prefill_tokens = prefill_metrics.computed_tokens + prefill_metrics.reused_tokens
            total_generated_tokens = eagle_metrics.total_generated_tokens

            print(
                f"Prefill: {prefill_metrics.computed_tokens} computed tokens, "
                f"{prefill_metrics.reused_tokens} reused tokens over {prefill_metrics.get_total_runs()} runs"
            )
            if eagle_metrics.total_iterations > 0:
                acceptance_rate = eagle_metrics.total_generated_tokens / eagle_metrics.total_iterations
                print(
                    f"Eagle Generation: {eagle_metrics.total_generated_tokens} tokens, "
                    f"{eagle_metrics.total_iterations} iterations, "
                    f"acceptance rate: {acceptance_rate:.2f}")

            # Calculate throughput metrics
            if elapsed_time > 0:
                generation_tps = total_generated_tokens / elapsed_time
                prefill_tps = total_prefill_tokens / elapsed_time
                print(f"\nThroughput:")
                print(f"  Generation: {generation_tps:.2f} tokens/sec")
                print(f"  Prefill: {prefill_tps:.2f} tokens/sec")
        else:
            prefill_metrics = runtime.prefill_metrics
            gen_metrics = runtime.generation_metrics

            total_prefill_tokens = prefill_metrics.computed_tokens + prefill_metrics.reused_tokens
            total_generated_tokens = gen_metrics.generated_tokens

            print(
                f"Prefill: {prefill_metrics.computed_tokens} computed tokens, "
                f"{prefill_metrics.reused_tokens} reused tokens over {prefill_metrics.get_total_runs()} runs"
            )
            print(
                f"Generation: {gen_metrics.generated_tokens} tokens over {gen_metrics.get_total_runs()} runs"
            )

            # Calculate throughput metrics
            if elapsed_time > 0:
                generation_tps = total_generated_tokens / elapsed_time
                prefill_tps = total_prefill_tokens / elapsed_time
                print(f"\nThroughput:")
                print(f"  Generation: {generation_tps:.2f} tokens/sec")
                print(f"  Prefill: {prefill_tps:.2f} tokens/sec")

        multimodal_metrics = runtime.multimodal_metrics
        if multimodal_metrics.total_images > 0:
            print(f"\nMultimodal: {multimodal_metrics.total_images} images, "
                  f"{multimodal_metrics.total_image_tokens} image tokens")

        print("=====================================\n")

    # Export profile to JSON file
    if args.profile_output_file:
        try:
            profile_json = {}

            if args.eagle.enabled:
                prefill_metrics = runtime.prefill_metrics
                eagle_metrics = runtime.eagle_generation_metrics

                total_prefill_tokens = prefill_metrics.computed_tokens + prefill_metrics.reused_tokens
                total_generated_tokens = eagle_metrics.total_generated_tokens

                profile_json["prefill"] = {
                    "computed_tokens": prefill_metrics.computed_tokens,
                    "reused_tokens": prefill_metrics.reused_tokens,
                    "total_runs": prefill_metrics.get_total_runs()
                }
                profile_json["eagle_generation"] = {
                    "total_iterations":
                    eagle_metrics.total_iterations,
                    "total_generated_tokens":
                    eagle_metrics.total_generated_tokens,
                    "total_runs":
                    eagle_metrics.get_total_runs(),
                    "average_acceptance_rate":
                    (eagle_metrics.total_generated_tokens /
                     eagle_metrics.total_iterations)
                    if eagle_metrics.total_iterations > 0 else 0
                }

                # Add throughput metrics
                if elapsed_time > 0:
                    profile_json["throughput"] = {
                        "generation_tokens_per_sec":
                        total_generated_tokens / elapsed_time,
                        "prefill_tokens_per_sec":
                        total_prefill_tokens / elapsed_time
                    }
            else:
                prefill_metrics = runtime.prefill_metrics
                gen_metrics = runtime.generation_metrics

                total_prefill_tokens = prefill_metrics.computed_tokens + prefill_metrics.reused_tokens
                total_generated_tokens = gen_metrics.generated_tokens

                profile_json["prefill"] = {
                    "computed_tokens": prefill_metrics.computed_tokens,
                    "reused_tokens": prefill_metrics.reused_tokens,
                    "total_runs": prefill_metrics.get_total_runs()
                }
                profile_json["generation"] = {
                    "generated_tokens": gen_metrics.generated_tokens,
                    "total_runs": gen_metrics.get_total_runs()
                }

                # Add throughput metrics
                if elapsed_time > 0:
                    profile_json["throughput"] = {
                        "generation_tokens_per_sec":
                        total_generated_tokens / elapsed_time,
                        "prefill_tokens_per_sec":
                        total_prefill_tokens / elapsed_time
                    }

            multimodal_metrics = runtime.multimodal_metrics
            profile_json["multimodal"] = {
                "total_images": multimodal_metrics.total_images,
                "total_image_tokens": multimodal_metrics.total_image_tokens,
                "total_runs": multimodal_metrics.get_total_runs()
            }

            profile_json["timing"] = {"total_elapsed_seconds": elapsed_time}

            with open(args.profile_output_file, 'w') as f:
                json.dump(profile_json, f, indent=2)
            print(f"Profile data exported to: {args.profile_output_file}")

        except Exception as e:
            print(f"ERROR: Failed to write profile output file: {e}")
            return 1

    # Export to JSON file
    try:
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"All responses exported to: {args.output_file}")
    except Exception as e:
        print(f"ERROR: Failed to write output file: {e}")
        return 1

    return 1 if has_failed_request else 0


if __name__ == "__main__":
    sys.exit(main())
