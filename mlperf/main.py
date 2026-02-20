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
import logging
import os
import sys
import time

import mlperf_loadgen as lg
import requests

sys.path.insert(0, os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format=
    '%(asctime)s.%(msecs)03d UTC - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logging.Formatter.converter = time.gmtime
log = logging.getLogger("Llama-8B-MAIN")

# function to check the model name in server matches the user specified one


def verify_model_name(user_specified_name, url):
    response = requests.get(url)
    if response.status_code == 200:
        response_dict = response.json()
        server_model_name = response_dict["data"][0]["id"]
        if user_specified_name == server_model_name:
            return {"matched": True, "error": False}
        else:
            return {
                "matched":
                False,
                "error":
                f"User specified {user_specified_name} and server model name {server_model_name} mismatch!",
            }
    else:
        return {
            "matched":
            False,
            "error":
            f"Failed to get a valid response. Status code: {response.status_code}",
        }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["Offline", "SingleStream"],
        help="Scenario",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        required=True,
                        help="Dataset path")
    parser.add_argument("--accuracy",
                        action="store_true",
                        help="Run accuracy mode")
    parser.add_argument(
        "--audit-conf",
        type=str,
        default="audit.conf",
        help="audit config for LoadGen settings during compliance runs",
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # TODO: This interpretation of 'total-sample-count' is a little
    # misleading. Fix it
    parser.add_argument(
        "--total-sample-count",
        type=int,
        default=5000,
        help="Number of samples to use in benchmark.",
    )
    parser.add_argument("--output-log-dir",
                        type=str,
                        default="output-logs",
                        help="Where logs are saved")
    parser.add_argument(
        "--enable-log-trace",
        action="store_true",
        help="Enable log tracing. This file can become quite large",
    )
    parser.add_argument(
        "--lg-model-name",
        type=str,
        default="llama3_1-8b-edge",
        choices=["llama3_1-8b-edge"],
        help="Model name(specified in llm server)",
    )
    parser.add_argument(
        "--token-output-file",
        type=str,
        default="tokens_output.json",
        help="Output file for token logging (JSON format)",
    )
    parser.add_argument(
        "--engine-dir",
        type=str,
        required=True,
        help="Engine directory",
    )
    # Eagle speculative decoding options
    parser.add_argument(
        "--eagle-draft-top-k",
        type=int,
        default=None,
        help=
        "Number of tokens selected per drafting step (default: 10 for SingleStream, 4 for Offline)",
    )
    parser.add_argument(
        "--eagle-draft-step",
        type=int,
        default=None,
        help=
        "Number of drafting steps to perform (default: 6 for SingleStream, 3 for Offline)",
    )
    parser.add_argument(
        "--eagle-verify-tree-size",
        type=int,
        default=None,
        help=
        "Number of tokens for base model verification (default: 60 for SingleStream, 12 for Offline)",
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=10,
        help="Number of warmup runs to perform before benchmark (default: 3)",
    )

    args = parser.parse_args()

    # Set scenario-specific defaults for Eagle parameters if not provided
    scenario_lower = args.scenario.lower()
    if args.eagle_draft_top_k is None:
        args.eagle_draft_top_k = 10 if scenario_lower == "singlestream" else 4
    if args.eagle_draft_step is None:
        args.eagle_draft_step = 6 if scenario_lower == "singlestream" else 3
    if args.eagle_verify_tree_size is None:
        args.eagle_verify_tree_size = 60 if scenario_lower == "singlestream" else 12

    return args


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    "singlestream": lg.TestScenario.SingleStream,
}


def main():
    args = get_args()
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    # mlperf.conf is automatically loaded by the loadgen
    # settings.FromConfig(args.mlperf_conf, "llama3_1-8b", args.scenario)
    settings.FromConfig(args.user_conf, args.lg_model_name, args.scenario)

    # Override min_query_count and min_duration to respect total_sample_count
    # This ensures LoadGen doesn't use the default values from user.conf
    # For Offline scenario, LoadGen will issue exactly min_query_count queries
    # For SingleStream scenario, LoadGen needs both min_query_count and min_duration=0
    # to limit queries (otherwise it will run for the full min_duration)
    if args.scenario.lower() == "offline":
        settings.min_query_count = args.total_sample_count
        log.info(
            f"Setting Offline scenario min_query_count to {args.total_sample_count} samples"
        )
    elif args.scenario.lower() == "singlestream":
        settings.min_query_count = args.total_sample_count
        settings.min_duration_ms = 0  # Set to 0 so it stops after min_query_count
        log.info(
            f"Setting SingleStream scenario min_query_count to {args.total_sample_count} samples and min_duration_ms to 0"
        )

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    from SUT_edgellm import SUT

    sut_map = {"offline": SUT, "singlestream": SUT}

    sut_cls = sut_map[args.scenario.lower()]

    # Log Eagle parameters being used
    log.info(
        f"Eagle parameters - top_k: {args.eagle_draft_top_k}, draft_step: {args.eagle_draft_step}, verify_tree_size: {args.eagle_verify_tree_size}"
    )

    sut = sut_cls(model_path=args.model_path,
                  dataset_path=args.dataset_path,
                  total_sample_count=args.total_sample_count,
                  scenario=args.scenario.lower(),
                  token_output_file=args.token_output_file,
                  engine_dir=args.engine_dir,
                  eagle_draft_top_k=args.eagle_draft_top_k,
                  eagle_draft_step=args.eagle_draft_step,
                  eagle_verify_tree_size=args.eagle_verify_tree_size,
                  warmup_count=args.warmup_count)

    # Start sut before loadgen starts
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)

    log.info("Starting Benchmark run")
    lg.StartTestWithLogSettings(lgSUT, sut.qsl, settings, log_settings,
                                args.audit_conf)
    log.info("Run Completed!")

    log.info("Cleaning up...")
    sut.stop()

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
