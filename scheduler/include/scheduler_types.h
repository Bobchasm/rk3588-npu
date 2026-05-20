#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "distributed/protocol.h"

using SessionId = distributed::SessionId;
using RequestId = distributed::RequestId;
using WorkerId = distributed::WorkerId;
using StageId = distributed::StageId;

using TokenCallback = std::function<void(int step, int id, float elapsed_ms)>;

// 调度器对外使用的生成结果。
// 与 worker 内部的 GenerationResult 解耦，便于后续在调度层增加文本、路由信息与错误语义。
struct GenerationResult {
    std::vector<int> output_ids;
    std::string output_text;
    std::vector<int> prompt_ids;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    float prefill_ms = 0.0f;
    float decode_ms = 0.0f;
    bool hit_stop = false;
    bool hit_repetition = false;
    std::string error_message;
};

using RequestStatus = distributed::StatusCode;
using TensorBuffer = distributed::TensorPayload;
using TensorShape = std::vector<int32_t>;
using DType = distributed::TensorDataType;
using GenerateTokensRequest = distributed::GenerateTokensRequest;
using GenerateTokensResponse = distributed::GenerateTokensResponse;
using StageForwardRequest = distributed::StageForwardRequest;
using StageForwardResponse = distributed::StageForwardResponse;
