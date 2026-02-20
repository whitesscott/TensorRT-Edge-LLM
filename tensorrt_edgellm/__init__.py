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
TensorRT Edge-LLM - A Python package for LLM inference and model preparation for edge deployment.

This package provides:
    - Runtime: Lightweight C++ runtime for LLM inference (no dependencies)
    - Quantization: LLM quantization with ModelOpt (requires modelopt)
    - ONNX Export: Model export utilities (requires torch, transformers)

Example Usage:
    .. code-block:: python

        # Runtime inference (no heavy dependencies)
        from tensorrt_edgellm.runtime import LLMRuntime
        
        runtime = LLMRuntime("path/to/engine")
        output = runtime.generate("Hello, world!")

        # Quantization and export (requires additional dependencies)
        from tensorrt_edgellm import (
            quantize_and_save_llm,
            export_llm_model,
            visual_export,
        )
"""

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

# Lazy imports for heavy dependencies
# These are only loaded when accessed, not at package import time
_lazy_imports = {
    # Quantization
    "quantize_and_save_llm":
    ("quantization.llm_quantization", "quantize_and_save_llm"),
    "quantize_and_save_draft":
    ("quantization.llm_quantization", "quantize_and_save_draft"),
    # ONNX export
    "export_llm_model": ("onnx_export.llm_export", "export_llm_model"),
    "export_draft_model": ("onnx_export.llm_export", "export_draft_model"),
    "visual_export": ("onnx_export.visual_export", "visual_export"),
    "audio_export": ("onnx_export.audio_export", "audio_export"),
    "insert_lora_and_save": ("onnx_export.lora", "insert_lora_and_save"),
    "process_lora_weights_and_save":
    ("onnx_export.lora", "process_lora_weights_and_save"),
    # Vocab reduction
    "reduce_vocab_size":
    ("vocab_reduction.vocab_reduction", "reduce_vocab_size"),
}


def __getattr__(name):
    """Lazy import for heavy dependencies."""
    if name in _lazy_imports:
        module_path, attr_name = _lazy_imports[name]
        import importlib
        module = importlib.import_module(f".{module_path}", __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes including lazy imports."""
    return list(_lazy_imports.keys()) + ["__version__", "runtime"]


__all__ = [
    # Quantization
    "quantize_and_save_llm",
    "quantize_and_save_draft",
    # ONNX export
    "export_draft_model",
    "export_llm_model",
    "visual_export",
    "audio_export",
    "insert_lora_and_save",
    "process_lora_weights_and_save",
    # Vocab reduction
    "reduce_vocab_size",
]
