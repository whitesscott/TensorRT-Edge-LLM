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
TensorRT Edge LLM Runtime - Python Interface

This module provides a Pythonic interface to the TensorRT Edge LLM inference runtimes.
It wraps the C++ pybind11 bindings with additional convenience methods.

Example usage:
    >>> from tensorrt_edgellm.runtime import LLMRuntime, EagleRuntime
    >>>
    >>> # Standard inference
    >>> runtime = LLMRuntime("/path/to/engine")
    >>> response = runtime.generate("What is the capital of France?")
    >>> print(response.output_texts[0])
    >>>
    >>> # Eagle speculative decoding
    >>> eagle = EagleRuntime("/path/to/eagle_engine")
    >>> response = eagle.generate([
    ...     {"role": "user", "content": "Explain quantum computing."}
    ... ])
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Import the C++ bindings
try:
    from tensorrt_edgellm import _edgellm_runtime as _rt
except ImportError as e:
    raise ImportError(
        "Could not import _edgellm_runtime. Make sure the C++ extension is built. "
        f"Original error: {e}")

# Re-export core types from C++ bindings
Message = _rt.Message
MessageContent = _rt.MessageContent
LLMGenerationRequest = _rt.LLMGenerationRequest
LLMGenerationResponse = _rt.LLMGenerationResponse
Request = _rt.Request
FormattedRequest = _rt.FormattedRequest
ImageData = _rt.ImageData

# Re-export metrics classes
LLMPrefillMetrics = _rt.LLMPrefillMetrics
LLMGenerationMetrics = _rt.LLMGenerationMetrics
EagleGenerationMetrics = _rt.EagleGenerationMetrics
MultimodalMetrics = _rt.MultimodalMetrics

# Re-export utility functions
set_profiling_enabled = _rt.set_profiling_enabled
get_profiling_enabled = _rt.get_profiling_enabled
load_image_from_path = _rt.load_image_from_path
load_image_from_bytes = _rt.load_image_from_bytes
create_text_message = _rt.create_text_message
create_generation_request = _rt.create_generation_request


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        temperature: Sampling temperature (higher = more random). Default: 1.0
        top_p: Top-p (nucleus) sampling parameter. Default: 0.8
        top_k: Top-k sampling parameter. Default: 50
        max_generate_length: Maximum number of tokens to generate. Default: 256
        apply_chat_template: Whether to apply chat template. Default: True
        add_generation_prompt: Whether to add assistant prompt. Default: True
        enable_thinking: Enable thinking mode for supported models. Default: False
        lora_weights_name: Name of LoRA weights to use. Default: ""
        save_system_prompt_kv_cache: Cache system prompt for reuse. Default: False
        disable_spec_decode: Force disable speculative decoding. Default: False
    """
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 50
    max_generate_length: int = 256
    apply_chat_template: bool = True
    add_generation_prompt: bool = True
    enable_thinking: bool = False
    lora_weights_name: str = ""
    save_system_prompt_kv_cache: bool = False
    disable_spec_decode: bool = False


def _convert_messages_to_cpp(messages: List[Dict[str, Any]]) -> List[Message]:
    """Convert Python message dictionaries to C++ Message objects.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Content can be a string or list of content items.

    Returns:
        List of Message objects for C++ runtime.
    """
    cpp_messages = []

    for msg in messages:
        cpp_msg = Message()
        cpp_msg.role = msg["role"]

        # Collect contents in a Python list first (pybind11 vector append doesn't work)
        contents_list = []

        content = msg["content"]

        # Handle string content (simple text message)
        if isinstance(content, str):
            mc = MessageContent("text", content)
            contents_list.append(mc)

        # Handle list content (multimodal)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    mc = MessageContent("text", item)
                elif isinstance(item, dict):
                    content_type = item.get("type", "text")
                    if content_type == "text":
                        mc = MessageContent("text", item.get("text", ""))
                    elif content_type == "image":
                        mc = MessageContent("image", item.get("image", ""))
                    else:
                        raise ValueError(
                            f"Unknown content type: {content_type}")
                else:
                    raise ValueError(
                        f"Invalid content item type: {type(item)}")
                contents_list.append(mc)
        else:
            raise ValueError(f"Invalid content type: {type(content)}")

        # Assign all contents at once (pybind11 requirement)
        cpp_msg.contents = contents_list
        cpp_messages.append(cpp_msg)

    return cpp_messages


def _load_images_for_request(
        messages: List[Dict[str, Any]]) -> List[ImageData]:
    """Load images referenced in messages.

    Args:
        messages: List of message dicts that may contain image references.

    Returns:
        List of loaded ImageData objects.
    """
    images = []

    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_path = item.get("image", "")
                    if image_path and os.path.isfile(image_path):
                        images.append(load_image_from_path(image_path))

    return images


class LLMRuntime:
    """High-level interface for standard LLM inference.

    This class wraps the C++ LLMInferenceRuntime with a more Pythonic interface.

    Args:
        engine_dir: Directory containing the LLM engine files.
        multimodal_engine_dir: Directory for multimodal engine (optional).
        lora_weights_map: Dictionary mapping LoRA names to weight paths.
        capture_cuda_graph: Whether to capture CUDA graph after init. Default: True

    Example:
        >>> runtime = LLMRuntime("/path/to/engine")
        >>>
        >>> # Simple text generation
        >>> response = runtime.generate("Hello, how are you?")
        >>>
        >>> # Chat-style generation
        >>> response = runtime.generate([
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is Python?"}
        ... ])
        >>>
        >>> # Batch generation
        >>> responses = runtime.generate_batch([
        ...     [{"role": "user", "content": "Question 1"}],
        ...     [{"role": "user", "content": "Question 2"}]
        ... ])
    """

    def __init__(self,
                 engine_dir: str,
                 multimodal_engine_dir: str = "",
                 lora_weights_map: Optional[Dict[str, str]] = None,
                 capture_cuda_graph: bool = True):
        self._lora_weights_map = lora_weights_map or {}
        self._runtime = _rt.LLMInferenceRuntime(engine_dir,
                                                multimodal_engine_dir,
                                                self._lora_weights_map)

        if capture_cuda_graph:
            success = self._runtime.capture_decoding_cuda_graph()
            if not success:
                import warnings
                warnings.warn(
                    "Failed to capture CUDA graph. Inference will proceed without "
                    "graph optimization, which may impact performance.")

    def generate(self,
                 messages: Union[str, List[Dict[str, Any]]],
                 config: Optional[GenerationConfig] = None,
                 **kwargs) -> LLMGenerationResponse:
        """Generate text from a prompt or conversation.

        Args:
            messages: Either a string prompt or a list of message dicts.
            config: Generation configuration. If None, uses defaults.
            **kwargs: Override config parameters.

        Returns:
            LLMGenerationResponse with output_texts and output_ids.
        """
        # Convert string to messages format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        return self.generate_batch([messages], config, **kwargs)

    def generate_batch(self,
                       batch_messages: List[List[Dict[str, Any]]],
                       config: Optional[GenerationConfig] = None,
                       **kwargs) -> LLMGenerationResponse:
        """Generate text for a batch of conversations.

        Args:
            batch_messages: List of conversations (each is a list of message dicts).
            config: Generation configuration. If None, uses defaults.
            **kwargs: Override config parameters.

        Returns:
            LLMGenerationResponse with output_texts and output_ids for each batch item.
        """
        config = config or GenerationConfig()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Build the request
        request = LLMGenerationRequest()
        request.temperature = config.temperature
        request.top_p = config.top_p
        request.top_k = config.top_k
        request.max_generate_length = config.max_generate_length
        request.apply_chat_template = config.apply_chat_template
        request.add_generation_prompt = config.add_generation_prompt
        request.enable_thinking = config.enable_thinking
        request.lora_weights_name = config.lora_weights_name
        request.save_system_prompt_kv_cache = config.save_system_prompt_kv_cache
        request.disable_spec_decode = config.disable_spec_decode

        # Convert messages and create requests
        for messages in batch_messages:
            req = Request()
            req.messages = _convert_messages_to_cpp(messages)
            req.image_buffers = _load_images_for_request(messages)
            request.requests.append(req)

        return self._runtime.handle_request(request)

    def save_system_prompt_cache(self,
                                 prompt: str,
                                 lora_weights_name: str = "") -> bool:
        """Pre-generate and save system prompt KV cache.

        This can speed up inference when the same system prompt is used repeatedly.

        Args:
            prompt: The system prompt to cache.
            lora_weights_name: LoRA weights name (if using LoRA).

        Returns:
            True if cache was saved successfully.
        """
        return self._runtime.save_system_prompt_kv_cache(
            prompt, lora_weights_name)

    @property
    def prefill_metrics(self) -> LLMPrefillMetrics:
        """Get prefill stage metrics."""
        return self._runtime.get_prefill_metrics()

    @property
    def generation_metrics(self) -> LLMGenerationMetrics:
        """Get generation stage metrics."""
        return self._runtime.get_generation_metrics()

    @property
    def multimodal_metrics(self) -> MultimodalMetrics:
        """Get multimodal processing metrics."""
        return self._runtime.get_multimodal_metrics()


class EagleRuntime:
    """High-level interface for Eagle speculative decoding inference.

    This class wraps the C++ LLMInferenceSpecDecodeRuntime with a Pythonic interface.
    Eagle uses a draft model to speculate multiple tokens, then verifies them with
    the base model, achieving higher throughput.

    Args:
        engine_dir: Directory containing Eagle engine files.
        multimodal_engine_dir: Directory for multimodal engine (optional).
        draft_top_k: Tokens per drafting step (default: 10).
        draft_step: Number of drafting steps (default: 6).
        verify_tree_size: Tokens for verification (default: 60).
        capture_cuda_graph: Whether to capture CUDA graph after init. Default: True

    Example:
        >>> runtime = EagleRuntime("/path/to/eagle_engine")
        >>>
        >>> # Generate with speculative decoding
        >>> response = runtime.generate([
        ...     {"role": "user", "content": "Explain machine learning."}
        ... ])
        >>> print(response.output_texts[0])
    """

    def __init__(self,
                 engine_dir: str,
                 multimodal_engine_dir: str = "",
                 draft_top_k: int = 10,
                 draft_step: int = 6,
                 verify_tree_size: int = 60,
                 capture_cuda_graph: bool = True):
        self._runtime = _rt.LLMInferenceSpecDecodeRuntime(
            engine_dir, multimodal_engine_dir, draft_top_k, draft_step,
            verify_tree_size)

        if capture_cuda_graph:
            success = self._runtime.capture_decoding_cuda_graph()
            if not success:
                import warnings
                warnings.warn(
                    "Failed to capture Eagle CUDA graphs. Inference will proceed "
                    "without graph optimization, which may impact performance."
                )

    def generate(self,
                 messages: Union[str, List[Dict[str, Any]]],
                 config: Optional[GenerationConfig] = None,
                 **kwargs) -> LLMGenerationResponse:
        """Generate text using Eagle speculative decoding.

        Args:
            messages: Either a string prompt or a list of message dicts.
            config: Generation configuration. If None, uses defaults.
            **kwargs: Override config parameters.

        Returns:
            LLMGenerationResponse with output_texts and output_ids.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        return self.generate_batch([messages], config, **kwargs)

    def generate_batch(self,
                       batch_messages: List[List[Dict[str, Any]]],
                       config: Optional[GenerationConfig] = None,
                       **kwargs) -> LLMGenerationResponse:
        """Generate text for a batch using Eagle speculative decoding.

        Args:
            batch_messages: List of conversations (each is a list of message dicts).
            config: Generation configuration. If None, uses defaults.
            **kwargs: Override config parameters.

        Returns:
            LLMGenerationResponse with output_texts and output_ids for each batch item.
        """
        config = config or GenerationConfig()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Build the request
        request = LLMGenerationRequest()
        request.temperature = config.temperature
        request.top_p = config.top_p
        request.top_k = config.top_k
        request.max_generate_length = config.max_generate_length
        request.apply_chat_template = config.apply_chat_template
        request.add_generation_prompt = config.add_generation_prompt
        request.enable_thinking = config.enable_thinking
        request.lora_weights_name = config.lora_weights_name
        request.save_system_prompt_kv_cache = config.save_system_prompt_kv_cache
        request.disable_spec_decode = config.disable_spec_decode

        # Convert messages and create requests
        for messages in batch_messages:
            req = Request()
            req.messages = _convert_messages_to_cpp(messages)
            req.image_buffers = _load_images_for_request(messages)
            request.requests.append(req)

        return self._runtime.handle_request(request)

    @property
    def prefill_metrics(self) -> LLMPrefillMetrics:
        """Get prefill stage metrics."""
        return self._runtime.get_prefill_metrics()

    @property
    def eagle_generation_metrics(self) -> EagleGenerationMetrics:
        """Get Eagle generation metrics (iterations, acceptance rate)."""
        return self._runtime.get_eagle_generation_metrics()

    @property
    def multimodal_metrics(self) -> MultimodalMetrics:
        """Get multimodal processing metrics."""
        return self._runtime.get_multimodal_metrics()


def parse_input_file(
        input_file_path: str,
        batch_size_override: Optional[int] = None,
        max_generate_length_override: Optional[int] = None) -> tuple:
    """Parse an input JSON file in the same format as llm_inference.cpp.

    Args:
        input_file_path: Path to input JSON file.
        batch_size_override: Override batch size from file.
        max_generate_length_override: Override max generate length from file.

    Returns:
        Tuple of (lora_weights_map, list of LLMGenerationRequest).
    """
    with open(input_file_path, 'r') as f:
        input_data = json.load(f)

    # Extract global parameters
    batch_size = batch_size_override or input_data.get("batch_size", 1)
    temperature = input_data.get("temperature", 1.0)
    top_p = input_data.get("top_p", 0.8)
    top_k = input_data.get("top_k", 50)
    max_generate_length = max_generate_length_override or input_data.get(
        "max_generate_length", 256)
    apply_chat_template = input_data.get("apply_chat_template", True)
    add_generation_prompt = input_data.get("add_generation_prompt", True)
    enable_thinking = input_data.get("enable_thinking", False)

    # Extract LoRA weights map
    lora_weights_map = {}
    if "available_lora_weights" in input_data:
        lora_weights_map = input_data["available_lora_weights"]

    # Parse requests
    requests_array = input_data.get("requests", [])
    batched_requests = []

    for start_idx in range(0, len(requests_array), batch_size):
        request = LLMGenerationRequest()
        request.temperature = temperature
        request.top_p = top_p
        request.top_k = top_k
        request.max_generate_length = max_generate_length
        request.apply_chat_template = apply_chat_template
        request.add_generation_prompt = add_generation_prompt
        request.enable_thinking = enable_thinking

        batch_lora_name = ""
        first_in_batch = True

        # Collect requests in a Python list first (pybind11 vector append doesn't work)
        batch_requests_list = []

        end_idx = min(start_idx + batch_size, len(requests_array))
        for request_item in requests_array[start_idx:end_idx]:
            # Handle per-request flags
            if request_item.get("save_system_prompt_kv_cache", False):
                request.save_system_prompt_kv_cache = True
            if request_item.get("disable_spec_decode", False):
                request.disable_spec_decode = True

            # Handle LoRA
            request_lora_name = request_item.get("lora_name", "")
            if first_in_batch:
                batch_lora_name = request_lora_name
                first_in_batch = False
            elif request_lora_name != batch_lora_name:
                raise ValueError(
                    "Different LoRA weights within the same batch are not supported"
                )

            # Parse messages
            messages_json = request_item.get("messages", [])
            messages = []

            for msg_json in messages_json:
                msg = {"role": msg_json["role"], "content": []}
                content_json = msg_json["content"]

                if isinstance(content_json, str):
                    msg["content"] = content_json
                elif isinstance(content_json, list):
                    for item in content_json:
                        if item["type"] == "text":
                            msg["content"].append({
                                "type": "text",
                                "text": item["text"]
                            })
                        elif item["type"] == "image":
                            msg["content"].append({
                                "type": "image",
                                "image": item["image"]
                            })

                messages.append(msg)

            req = Request()
            req.messages = _convert_messages_to_cpp(messages)
            req.image_buffers = _load_images_for_request(messages)
            batch_requests_list.append(req)

        # Assign all requests at once (pybind11 requirement)
        request.requests = batch_requests_list

        if batch_lora_name:
            request.lora_weights_name = batch_lora_name

        batched_requests.append(request)

    return lora_weights_map, batched_requests


__all__ = [
    # Config and types
    'GenerationConfig',
    'Message',
    'MessageContent',
    'Request',
    'FormattedRequest',
    'LLMGenerationRequest',
    'LLMGenerationResponse',
    'ImageData',

    # Metrics
    'LLMPrefillMetrics',
    'LLMGenerationMetrics',
    'EagleGenerationMetrics',
    'MultimodalMetrics',

    # Runtime classes
    'LLMRuntime',
    'EagleRuntime',

    # Utility functions
    'set_profiling_enabled',
    'get_profiling_enabled',
    'load_image_from_path',
    'load_image_from_bytes',
    'create_text_message',
    'create_generation_request',
    'parse_input_file',
]
