/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

/**
 * @brief RAII wrapper for CUDA stream management
 *
 * Automatically creates and destroys a CUDA stream.
 */
class CudaStreamWrapper
{
public:
    CudaStreamWrapper()
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    ~CudaStreamWrapper()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    // Disable copy
    CudaStreamWrapper(CudaStreamWrapper const&) = delete;
    CudaStreamWrapper& operator=(CudaStreamWrapper const&) = delete;

    // Enable move
    CudaStreamWrapper(CudaStreamWrapper&& other) noexcept
        : mStream(other.mStream)
    {
        other.mStream = nullptr;
    }

    CudaStreamWrapper& operator=(CudaStreamWrapper&& other) noexcept
    {
        if (this != &other)
        {
            if (mStream != nullptr)
            {
                cudaStreamDestroy(mStream);
            }
            mStream = other.mStream;
            other.mStream = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const
    {
        return mStream;
    }

private:
    cudaStream_t mStream{nullptr};
};

/**
 * @brief Python-friendly wrapper for LLMInferenceRuntime
 *
 * Manages CUDA stream lifecycle and provides a cleaner Python interface.
 */
class PyLLMInferenceRuntime
{
public:
    PyLLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap)
    {
        // Load TensorRT plugins
        mPluginHandle = loadEdgellmPluginLib();

        // Create runtime with managed CUDA stream
        mRuntime = std::make_unique<LLMInferenceRuntime>(engineDir, multimodalEngineDir, loraWeightsMap, mStream.get());
    }

    /**
     * @brief Handle a generation request and return the response
     */
    LLMGenerationResponse handleRequest(LLMGenerationRequest const& request)
    {
        LLMGenerationResponse response;
        bool success = mRuntime->handleRequest(request, response, mStream.get());
        if (!success)
        {
            throw std::runtime_error("Failed to handle generation request");
        }
        return response;
    }

    /**
     * @brief Handle a generation request with token IDs directly (token-in, token-out)
     * @param batchedInputTokenIds List of input token ID lists (one per batch item)
     * @param temperature Temperature for sampling
     * @param topP Top-p (nucleus) sampling parameter
     * @param topK Top-k sampling parameter
     * @param maxGenerateLength Maximum number of tokens to generate
     * @param tokenCallback Optional Python callable for streaming tokens (None to disable)
     * @return Response with output token IDs (output_texts will be empty)
     */
    LLMGenerationResponse handleRequestWithTokens(std::vector<std::vector<int32_t>> const& batchedInputTokenIds,
        float temperature, float topP, int64_t topK, int64_t maxGenerateLength, py::object tokenCallback = py::none())
    {
        LLMGenerationResponse response;
        TokenStreamCallback callback = nullptr;

        // Convert Python callable to C++ callback if provided
        if (!tokenCallback.is_none())
        {
            callback = [tokenCallback](int32_t batchIndex, int32_t tokenId, bool isFirstToken) -> bool {
                try
                {
                    py::gil_scoped_acquire acquire;
                    py::object result = tokenCallback(batchIndex, tokenId, isFirstToken);
                    // If callback returns False, stop generation
                    if (py::isinstance<py::bool_>(result))
                    {
                        return py::cast<bool>(result);
                    }
                    // Default to continue if callback doesn't return bool
                    return true;
                }
                catch (py::error_already_set& e)
                {
                    // Log error but continue generation
                    LOG_ERROR("Python callback error: %s", e.what());
                    return true;
                }
            };
        }

        bool success = mRuntime->handleRequestWithTokens(
            batchedInputTokenIds, temperature, topP, topK, maxGenerateLength, response, mStream.get(), callback);
        if (!success)
        {
            throw std::runtime_error("Failed to handle token-based generation request");
        }
        return response;
    }

    /**
     * @brief Capture CUDA graph for optimized decoding
     */
    bool captureDecodingCudaGraph()
    {
        return mRuntime->captureDecodingCUDAGraph(mStream.get());
    }

    /**
     * @brief Pre-generate and save system prompt KV cache
     */
    bool saveSystemPromptKVCache(std::string const& prompt, std::string const& loraWeightsName)
    {
        return mRuntime->genAndSaveSystemPromptKVCache(prompt, loraWeightsName, mStream.get());
    }

    /**
     * @brief Get prefill metrics
     */
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const
    {
        return mRuntime->getPrefillMetrics();
    }

    /**
     * @brief Get generation metrics
     */
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const
    {
        return mRuntime->getGenerationMetrics();
    }

    /**
     * @brief Get multimodal metrics
     */
    metrics::MultimodalMetrics getMultimodalMetrics() const
    {
        return mRuntime->getMultimodalMetrics();
    }

private:
    CudaStreamWrapper mStream;
    std::unique_ptr<LLMInferenceRuntime> mRuntime;
    std::unique_ptr<void, DlDeleter> mPluginHandle;
};

/**
 * @brief Python-friendly wrapper for LLMInferenceSpecDecodeRuntime (Eagle)
 *
 * Manages CUDA stream lifecycle and provides a cleaner Python interface.
 */
class PyLLMInferenceSpecDecodeRuntime
{
public:
    PyLLMInferenceSpecDecodeRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        int32_t draftTopK, int32_t draftStep, int32_t verifyTreeSize)
    {
        // Load TensorRT plugins
        mPluginHandle = loadEdgellmPluginLib();

        // Create drafting config
        EagleDraftingConfig draftingConfig{draftTopK, draftStep, verifyTreeSize};

        // Create runtime with managed CUDA stream
        mRuntime = std::make_unique<LLMInferenceSpecDecodeRuntime>(
            engineDir, multimodalEngineDir, draftingConfig, mStream.get());
    }

    /**
     * @brief Handle a generation request and return the response
     */
    LLMGenerationResponse handleRequest(LLMGenerationRequest const& request)
    {
        LLMGenerationResponse response;
        bool success = mRuntime->handleRequest(request, response, mStream.get());
        if (!success)
        {
            throw std::runtime_error("Failed to handle generation request");
        }
        return response;
    }

    /**
     * @brief Handle a generation request with token IDs directly (token-in, token-out)
     * @param batchedInputTokenIds List of input token ID lists (one per batch item)
     * @param temperature Temperature for sampling
     * @param topP Top-p (nucleus) sampling parameter
     * @param topK Top-k sampling parameter
     * @param maxGenerateLength Maximum number of tokens to generate
     * @param enableSpecDecode Whether to enable speculative decoding
     * @param tokenCallback Optional Python callable for streaming tokens (None to disable)
     * @return Response with output token IDs (output_texts will be empty)
     */
    LLMGenerationResponse handleRequestWithTokens(std::vector<std::vector<int32_t>> const& batchedInputTokenIds,
        float temperature, float topP, int64_t topK, int64_t maxGenerateLength, bool enableSpecDecode,
        py::object tokenCallback = py::none())
    {
        LLMGenerationResponse response;
        TokenStreamCallback callback = nullptr;

        // Convert Python callable to C++ callback if provided
        if (!tokenCallback.is_none())
        {
            callback = [tokenCallback](int32_t batchIndex, int32_t tokenId, bool isFirstToken) -> bool {
                try
                {
                    py::gil_scoped_acquire acquire;
                    py::object result = tokenCallback(batchIndex, tokenId, isFirstToken);
                    // If callback returns False, stop generation
                    if (py::isinstance<py::bool_>(result))
                    {
                        return py::cast<bool>(result);
                    }
                    // Default to continue if callback doesn't return bool
                    return true;
                }
                catch (py::error_already_set& e)
                {
                    // Log error but continue generation
                    LOG_ERROR("Python callback error: %s", e.what());
                    return true;
                }
            };
        }

        bool success = mRuntime->handleRequestWithTokens(batchedInputTokenIds, temperature, topP, topK,
            maxGenerateLength, enableSpecDecode, response, mStream.get(), callback);
        if (!success)
        {
            throw std::runtime_error("Failed to handle token-based generation request");
        }
        return response;
    }

    /**
     * @brief Capture CUDA graphs for optimized Eagle decoding
     */
    bool captureDecodingCudaGraph()
    {
        return mRuntime->captureDecodingCudaGraph(mStream.get());
    }

    /**
     * @brief Get prefill metrics
     */
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const
    {
        return mRuntime->getPrefillMetrics();
    }

    /**
     * @brief Get Eagle generation metrics
     */
    metrics::EagleGenerationMetrics const& getEagleGenerationMetrics() const
    {
        return mRuntime->getEagleGenerationMetrics();
    }

    /**
     * @brief Get multimodal metrics
     */
    metrics::MultimodalMetrics getMultimodalMetrics() const
    {
        return mRuntime->getMultimodalMetrics();
    }

private:
    CudaStreamWrapper mStream;
    std::unique_ptr<LLMInferenceSpecDecodeRuntime> mRuntime;
    std::unique_ptr<void, DlDeleter> mPluginHandle;
};

/**
 * @brief Load image from file path
 */
imageUtils::ImageData loadImageFromPath(std::string const& path)
{
    return imageUtils::loadImageFromFile(path);
}

/**
 * @brief Load image from bytes (Python bytes object)
 */
imageUtils::ImageData loadImageFromBytes(py::bytes const& data)
{
    std::string dataStr = data;
    return imageUtils::loadImageFromMemory(reinterpret_cast<unsigned char const*>(dataStr.data()), dataStr.size());
}

} // anonymous namespace

PYBIND11_MODULE(_edgellm_runtime, m)
{
    m.doc() = "TensorRT Edge LLM Python Bindings";

    // ========================================================================
    // Profiling control functions
    // ========================================================================
    m.def("set_profiling_enabled", &setProfilingEnabled, py::arg("enabled"),
        "Enable or disable profiling data collection");
    m.def("get_profiling_enabled", &getProfilingEnabled, "Check if profiling is currently enabled");

    // ========================================================================
    // Metrics classes
    // ========================================================================
    py::class_<metrics::LLMPrefillMetrics>(m, "LLMPrefillMetrics", "Metrics for LLM prefill stage")
        .def_readonly(
            "reused_tokens", &metrics::LLMPrefillMetrics::reusedTokens, "Number of tokens reused from KV cache")
        .def_readonly("computed_tokens", &metrics::LLMPrefillMetrics::computedTokens, "Number of newly computed tokens")
        .def("get_total_runs", &metrics::LLMPrefillMetrics::getTotalRuns, "Get total number of prefill runs");

    py::class_<metrics::LLMGenerationMetrics>(m, "LLMGenerationMetrics", "Metrics for LLM generation stage")
        .def_readonly(
            "generated_tokens", &metrics::LLMGenerationMetrics::generatedTokens, "Total number of generated tokens")
        .def("get_total_runs", &metrics::LLMGenerationMetrics::getTotalRuns, "Get total number of generation runs");

    py::class_<metrics::EagleGenerationMetrics>(m, "EagleGenerationMetrics", "Metrics for Eagle speculative decoding")
        .def_readonly(
            "total_iterations", &metrics::EagleGenerationMetrics::totalIterations, "Total number of Eagle iterations")
        .def_readonly("total_generated_tokens", &metrics::EagleGenerationMetrics::totalGeneratedTokens,
            "Total number of generated tokens")
        .def("get_total_runs", &metrics::EagleGenerationMetrics::getTotalRuns,
            "Get total number of Eagle generation runs");

    py::class_<metrics::MultimodalMetrics>(m, "MultimodalMetrics", "Metrics for multimodal processing")
        .def_readonly("total_images", &metrics::MultimodalMetrics::totalImages, "Total number of processed images")
        .def_readonly("total_image_tokens", &metrics::MultimodalMetrics::totalImageTokens,
            "Total number of image tokens generated")
        .def("get_total_runs", &metrics::MultimodalMetrics::getTotalRuns,
            "Get total number of multimodal processing runs");

    // ========================================================================
    // Image utilities
    // ========================================================================
    py::class_<imageUtils::ImageData>(m, "ImageData", "Container for image data")
        .def(py::init<>())
        .def_readonly("width", &imageUtils::ImageData::width, "Image width")
        .def_readonly("height", &imageUtils::ImageData::height, "Image height")
        .def_readonly("channels", &imageUtils::ImageData::channels, "Number of channels");

    m.def("load_image_from_path", &loadImageFromPath, py::arg("path"), "Load image from file path");
    m.def("load_image_from_bytes", &loadImageFromBytes, py::arg("data"), "Load image from bytes");

    // ========================================================================
    // Message structures
    // ========================================================================
    py::class_<Message::MessageContent>(m, "MessageContent", "Content item within a message")
        .def(py::init<>())
        .def(py::init([](std::string const& type, std::string const& content) {
            Message::MessageContent mc;
            mc.type = type;
            mc.content = content;
            return mc;
        }),
            py::arg("type"), py::arg("content"))
        .def_readwrite("type", &Message::MessageContent::type, "Content type (text, image)")
        .def_readwrite("content", &Message::MessageContent::content, "Content string (text content or image path)");

    py::class_<Message>(m, "Message", "Chat message with role and contents")
        .def(py::init<>())
        .def(py::init([](std::string const& role, std::vector<Message::MessageContent> const& contents) {
            Message msg;
            msg.role = role;
            msg.contents = contents;
            return msg;
        }),
            py::arg("role"), py::arg("contents"))
        .def_readwrite("role", &Message::role, "Message role (system, user, assistant)")
        .def_readwrite("contents", &Message::contents, "List of content items");

    // Convenience function to create a text message
    m.def(
        "create_text_message",
        [](std::string const& role, std::string const& text) {
            Message msg;
            msg.role = role;
            Message::MessageContent content;
            content.type = "text";
            content.content = text;
            msg.contents.push_back(content);
            return msg;
        },
        py::arg("role"), py::arg("text"), "Create a simple text message with the given role and text content");

    // ========================================================================
    // Request/Response structures
    // ========================================================================
    py::class_<LLMGenerationRequest::FormattedRequest>(
        m, "FormattedRequest", "Formatted request after chat template application")
        .def(py::init<>())
        .def_readwrite("formatted_system_prompt", &LLMGenerationRequest::FormattedRequest::formattedSystemPrompt,
            "Formatted system prompt for KV cache saving")
        .def_readwrite("formatted_complete_request", &LLMGenerationRequest::FormattedRequest::formattedCompleteRequest,
            "Complete formatted request string");

    py::class_<LLMGenerationRequest::Request>(m, "Request", "Single request containing messages and optional images")
        .def(py::init<>())
        .def(py::init([](std::vector<Message> const& messages) {
            LLMGenerationRequest::Request req;
            req.messages = messages;
            return req;
        }),
            py::arg("messages"))
        .def_readwrite("messages", &LLMGenerationRequest::Request::messages, "List of messages in the conversation")
        .def_readwrite(
            "image_buffers", &LLMGenerationRequest::Request::imageBuffers, "Optional image data for multimodal inputs");

    py::class_<LLMGenerationRequest>(
        m, "LLMGenerationRequest", "Generation request containing batch of requests and sampling parameters")
        .def(py::init<>())
        .def_readwrite("requests", &LLMGenerationRequest::requests, "Vector of requests for batch processing")
        .def_readwrite("formatted_requests", &LLMGenerationRequest::formattedRequests,
            "Formatted requests after template application (populated by runtime)")
        .def_readwrite("temperature", &LLMGenerationRequest::temperature, "Temperature for sampling (default: 1.0)")
        .def_readwrite("top_p", &LLMGenerationRequest::topP, "Top-p (nucleus) sampling parameter (default: 0.8)")
        .def_readwrite("top_k", &LLMGenerationRequest::topK, "Top-k sampling parameter (default: 50)")
        .def_readwrite(
            "max_generate_length", &LLMGenerationRequest::maxGenerateLength, "Maximum number of tokens to generate")
        .def_readwrite("lora_weights_name", &LLMGenerationRequest::loraWeightsName,
            "Name of LoRA weights to use (empty string for no LoRA)")
        .def_readwrite("save_system_prompt_kv_cache", &LLMGenerationRequest::saveSystemPromptKVCache,
            "Whether to save system prompt KV cache for reuse")
        .def_readwrite("apply_chat_template", &LLMGenerationRequest::applyChatTemplate,
            "Whether to apply chat template formatting (default: true)")
        .def_readwrite("add_generation_prompt", &LLMGenerationRequest::addGenerationPrompt,
            "Whether to add generation prompt at the end (default: true)")
        .def_readwrite("enable_thinking", &LLMGenerationRequest::enableThinking,
            "Whether to enable thinking mode for supported models (default: false)")
        .def_readwrite("disable_spec_decode", &LLMGenerationRequest::disableSpecDecode,
            "Force disable speculative decoding even if Eagle engine is loaded");

    py::class_<LLMGenerationResponse>(
        m, "LLMGenerationResponse", "Generation response containing output tokens and text")
        .def(py::init<>())
        .def_readwrite(
            "output_ids", &LLMGenerationResponse::outputIds, "Generated token IDs for each request in the batch")
        .def_readwrite("output_texts", &LLMGenerationResponse::outputTexts,
            "Generated text strings for each request in the batch");

    // ========================================================================
    // Runtime classes
    // ========================================================================
    py::class_<PyLLMInferenceRuntime>(m, "LLMInferenceRuntime",
        R"doc(
        LLM Inference Runtime for standard (non-speculative) generation.

        This runtime handles text generation using a single LLM model with optional
        multimodal support and LoRA weights.

        Args:
            engine_dir: Directory containing the LLM engine files
            multimodal_engine_dir: Directory for multimodal engine (empty string if not used)
            lora_weights_map: Dictionary mapping LoRA names to weight file paths

        Example:
            >>> runtime = LLMInferenceRuntime(
            ...     engine_dir="/path/to/engine",
            ...     multimodal_engine_dir="",
            ...     lora_weights_map={}
            ... )
            >>> runtime.capture_decoding_cuda_graph()
            >>> request = LLMGenerationRequest()
            >>> # ... configure request ...
            >>> response = runtime.handle_request(request)
        )doc")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir") = "",
            py::arg("lora_weights_map") = std::unordered_map<std::string, std::string>{})
        .def("handle_request", &PyLLMInferenceRuntime::handleRequest, py::arg("request"),
            "Process a generation request and return the response")
        .def("handle_request_with_tokens", &PyLLMInferenceRuntime::handleRequestWithTokens,
            py::arg("batched_input_token_ids"), py::arg("temperature") = 1.0f, py::arg("top_p") = 0.8f,
            py::arg("top_k") = 50, py::arg("max_generate_length") = 256, py::arg("token_callback") = py::none(),
            "Process a generation request with token IDs directly (token-in, token-out). Returns response with "
            "output_ids only (output_texts will be empty). Optionally accepts a callback(batch_index, token_id, "
            "is_first_token) -> bool for streaming tokens.")
        .def("capture_decoding_cuda_graph", &PyLLMInferenceRuntime::captureDecodingCudaGraph,
            "Capture CUDA graph for optimized decoding (call after construction)")
        .def("save_system_prompt_kv_cache", &PyLLMInferenceRuntime::saveSystemPromptKVCache, py::arg("prompt"),
            py::arg("lora_weights_name") = "", "Pre-generate and cache system prompt KV cache for faster inference")
        .def("get_prefill_metrics", &PyLLMInferenceRuntime::getPrefillMetrics,
            py::return_value_policy::reference_internal, "Get prefill stage metrics")
        .def("get_generation_metrics", &PyLLMInferenceRuntime::getGenerationMetrics,
            py::return_value_policy::reference_internal, "Get generation stage metrics")
        .def("get_multimodal_metrics", &PyLLMInferenceRuntime::getMultimodalMetrics,
            "Get multimodal processing metrics");

    py::class_<PyLLMInferenceSpecDecodeRuntime>(m, "LLMInferenceSpecDecodeRuntime",
        R"doc(
        LLM Inference Runtime with Eagle Speculative Decoding.

        This runtime uses Eagle speculative decoding for improved throughput,
        coordinating between a base model and a draft model.

        Args:
            engine_dir: Directory containing Eagle engine files (eagle_base.engine, eagle_draft.engine)
            multimodal_engine_dir: Directory for multimodal engine (empty string if not used)
            draft_top_k: Number of tokens selected per drafting step (default: 10)
            draft_step: Number of drafting steps to perform (default: 6)
            verify_tree_size: Number of tokens for base model verification (default: 60)

        Example:
            >>> runtime = LLMInferenceSpecDecodeRuntime(
            ...     engine_dir="/path/to/eagle_engine",
            ...     multimodal_engine_dir="",
            ...     draft_top_k=10,
            ...     draft_step=6,
            ...     verify_tree_size=60
            ... )
            >>> runtime.capture_decoding_cuda_graph()
            >>> request = LLMGenerationRequest()
            >>> # ... configure request ...
            >>> response = runtime.handle_request(request)
        )doc")
        .def(py::init<std::string const&, std::string const&, int32_t, int32_t, int32_t>(), py::arg("engine_dir"),
            py::arg("multimodal_engine_dir") = "", py::arg("draft_top_k") = 10, py::arg("draft_step") = 6,
            py::arg("verify_tree_size") = 60)
        .def("handle_request", &PyLLMInferenceSpecDecodeRuntime::handleRequest, py::arg("request"),
            "Process a generation request using Eagle speculative decoding")
        .def("handle_request_with_tokens", &PyLLMInferenceSpecDecodeRuntime::handleRequestWithTokens,
            py::arg("batched_input_token_ids"), py::arg("temperature") = 1.0f, py::arg("top_p") = 0.8f,
            py::arg("top_k") = 50, py::arg("max_generate_length") = 256, py::arg("enable_spec_decode") = true,
            py::arg("token_callback") = py::none(),
            "Process a generation request with token IDs directly (token-in, token-out) using Eagle speculative "
            "decoding. Returns response with output_ids only (output_texts will be empty). Optionally accepts a "
            "callback(batch_index, token_id, is_first_token) -> bool for streaming tokens.")
        .def("capture_decoding_cuda_graph", &PyLLMInferenceSpecDecodeRuntime::captureDecodingCudaGraph,
            "Capture CUDA graphs for all Eagle decoding stages (call after construction)")
        .def("get_prefill_metrics", &PyLLMInferenceSpecDecodeRuntime::getPrefillMetrics,
            py::return_value_policy::reference_internal, "Get prefill stage metrics")
        .def("get_eagle_generation_metrics", &PyLLMInferenceSpecDecodeRuntime::getEagleGenerationMetrics,
            py::return_value_policy::reference_internal, "Get Eagle generation metrics")
        .def("get_multimodal_metrics", &PyLLMInferenceSpecDecodeRuntime::getMultimodalMetrics,
            "Get multimodal processing metrics");

    // ========================================================================
    // Convenience functions for building requests
    // ========================================================================
    m.def(
        "create_generation_request",
        [](std::vector<std::vector<Message>> const& batch_messages, float temperature, float top_p, int64_t top_k,
            int64_t max_generate_length, bool apply_chat_template, bool add_generation_prompt, bool enable_thinking,
            std::string const& lora_weights_name, bool save_system_prompt_kv_cache, bool disable_spec_decode) {
            LLMGenerationRequest request;
            request.temperature = temperature;
            request.topP = top_p;
            request.topK = top_k;
            request.maxGenerateLength = max_generate_length;
            request.applyChatTemplate = apply_chat_template;
            request.addGenerationPrompt = add_generation_prompt;
            request.enableThinking = enable_thinking;
            request.loraWeightsName = lora_weights_name;
            request.saveSystemPromptKVCache = save_system_prompt_kv_cache;
            request.disableSpecDecode = disable_spec_decode;

            for (auto const& messages : batch_messages)
            {
                LLMGenerationRequest::Request req;
                req.messages = messages;
                request.requests.push_back(std::move(req));
            }

            return request;
        },
        py::arg("batch_messages"), py::arg("temperature") = 1.0f, py::arg("top_p") = 0.8f, py::arg("top_k") = 50,
        py::arg("max_generate_length") = 256, py::arg("apply_chat_template") = true,
        py::arg("add_generation_prompt") = true, py::arg("enable_thinking") = false, py::arg("lora_weights_name") = "",
        py::arg("save_system_prompt_kv_cache") = false, py::arg("disable_spec_decode") = false,
        R"doc(
        Create a generation request from a batch of message lists.

        This is a convenience function to create a properly configured
        LLMGenerationRequest from Python-native data structures.

        Args:
            batch_messages: List of conversation message lists (one per batch item)
            temperature: Sampling temperature (default: 1.0)
            top_p: Top-p (nucleus) sampling parameter (default: 0.8)
            top_k: Top-k sampling parameter (default: 50)
            max_generate_length: Maximum tokens to generate (default: 256)
            apply_chat_template: Apply chat template formatting (default: True)
            add_generation_prompt: Add assistant generation prompt (default: True)
            enable_thinking: Enable thinking mode for supported models (default: False)
            lora_weights_name: LoRA weights name to use (default: "")
            save_system_prompt_kv_cache: Cache system prompt KV (default: False)
            disable_spec_decode: Force disable speculative decoding (default: False)

        Returns:
            Configured LLMGenerationRequest
        )doc");
}
