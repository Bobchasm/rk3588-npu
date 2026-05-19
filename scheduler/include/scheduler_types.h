#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using SessionId = std::string;
using RequestId = uint64_t;
using WorkerId  = std::string;
using StageId   = int;

using TokenCallback = std::function<void(int step, int id, float elapsed_ms)>;

struct GenerationResult {
    std::vector<int> output_ids;
    int prefill_tokens = 0;
    int decode_tokens  = 0;
    float prefill_ms   = 0.0f;
    float decode_ms    = 0.0f;
    bool hit_stop      = false;
    bool hit_repetition = false;
    std::string error_message;
};

enum class DType {
    FP16,
    FP32,
};

enum class RequestStatus {
    OK,
    ERROR,
};

struct TensorShape {
    std::vector<int> dims;
};

struct TensorBuffer {
    std::vector<uint8_t> data;
    TensorShape shape;
    DType dtype = DType::FP16;
};

struct PrefillRequest {
    SessionId session_id;
    RequestId request_id;
    std::vector<int> input_ids;
    int max_new_tokens = 10;
};

struct PrefillResponse {
    RequestId request_id;
    RequestStatus status = RequestStatus::OK;
    std::string message;
};

struct StageRunRequest {
    SessionId session_id;
    RequestId request_id;
    StageId stage_id = 0;
    TensorBuffer input;
    int pos_base = 0;
};

struct StageRunResponse {
    RequestId request_id;
    RequestStatus status = RequestStatus::OK;
    std::string message;
    TensorBuffer output;
};

struct DecodeStepRequest {
    SessionId session_id;
    RequestId request_id;
    TensorBuffer input;
    int pos_base = 0;
};

struct DecodeStepResponse {
    RequestId request_id;
    RequestStatus status = RequestStatus::OK;
    std::string message;
    TensorBuffer output;
    std::vector<int> next_tokens;
};
