#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace distributed {

// ============================================================
// 分布式调度协议（调度器 <-> worker / worker <-> worker）
//
// 设计目标：
// 1. 明确区分“输入 token id 的 head/full-model 请求”与“输入 hidden state 的 stage 请求”。
// 2. 当前单 worker 可以直接走 FullGenerate 链路。
// 3. 后续多 stage pipeline 可以复用同一套上下文元数据与张量封装。
// 4. 不引入额外三方 JSON 库，使用简单可控的文本协议完成序列化。
// ============================================================

using SessionId = std::string;
using RequestId = uint64_t;
using WorkerId = std::string;
using StageId = int32_t;

enum class RpcCommand {
    kUnknown = 0,
    kPing,
    kResetCache,
    kGenerateTokens,
    kForwardStage,
    kTokensToHidden,
    kHiddenToToken,
};

enum class StatusCode {
    kOk = 0,
    kError,
    kUnsupported,
};

enum class TensorDataType {
    kUnknown = 0,
    kFloat16,
    kFloat32,
    kInt32,
    kUInt8,
};

// 路由信息独立出来，方便未来做多 stage / 多 hop 转发。
struct RouteInfo {
    StageId stage_id = 0;              // 当前请求命中的 stage 编号
    int32_t total_stages = 1;          // 当前 pipeline 总 stage 数
    int32_t hop_index = 0;             // 当前是第几跳，调度器中继时可递增
    int32_t pos_base = 0;              // 当前 hidden state / token 所在的绝对位置
    WorkerId source_worker_id;         // 上游 worker，调度器发起时可为空
    WorkerId target_worker_id;         // 当前目标 worker
};

// 所有 RPC 请求都带一份统一上下文，便于日志、追踪、重试和限时控制。
struct RequestContext {
    SessionId session_id;              // 逻辑会话 id，跨多轮请求保持稳定
    RequestId request_id = 0;          // 单次请求 id，每次提交生成任务都应唯一
    std::string trace_id;              // 链路追踪 id，可用于日志聚合
    int32_t priority = 0;              // 调度优先级，值越大优先级越高
    int32_t timeout_ms = 0;            // 0 表示不显式限制，由上层兜底
    RouteInfo route;                   // pipeline / stage 路由元数据
};

// 生成参数单独抽象，避免混入业务逻辑。
struct GenerationParameters {
    int32_t max_new_tokens = 64;       // 本次最多生成多少个新 token
    int32_t max_prompt_chars = 8192;   // 调度器历史拼 prompt 时的截断上限（字符级）
    int32_t repetition_window = 6;     // 与当前 LLMEngine 的重复检测窗口保持一致
};

// 通用张量封装。后续 stage worker 之间传递的 hidden state / logits 都复用这里。
struct TensorPayload {
    TensorDataType dtype = TensorDataType::kUnknown;
    std::vector<int32_t> shape;        // 例如 [batch, seq, hidden]
    std::vector<uint8_t> bytes;        // 原始字节，当前先保留最通用表示
};

// head worker 或单 worker full-model 的输入：
// 输入是 token ids，输出通常是新生成 token ids。
struct GenerateTokensRequest {
    RequestContext context;
    GenerationParameters generation;
    std::vector<int32_t> input_token_ids;
};

struct GenerateTokensResponse {
    RequestContext context;
    StatusCode status = StatusCode::kOk;
    std::string message;
    std::vector<int32_t> output_token_ids;
    int32_t prefill_tokens = 0;
    int32_t decode_tokens = 0;
    float prefill_ms = 0.0f;
    float decode_ms = 0.0f;
    bool hit_stop = false;
    bool hit_repetition = false;
};

// 中间 stage worker 的输入：
// 输入是上一跳的 hidden state，输出是本 stage 计算后的 hidden state。
struct StageForwardRequest {
    RequestContext context;
    TensorPayload input_tensor;
};

struct StageForwardResponse {
    RequestContext context;
    StatusCode status = StatusCode::kOk;
    std::string message;
    TensorPayload output_tensor;
};

// head worker 的输入：
// 输入是 token ids，输出是经过本地 embedding + 若干层后的 hidden state。
struct TokensToHiddenRequest {
    RequestContext context;
    std::vector<int32_t> input_token_ids;
};

struct TokensToHiddenResponse {
    RequestContext context;
    StatusCode status = StatusCode::kOk;
    std::string message;
    TensorPayload output_tensor;
};

// tail worker 的输入：
// 输入是上一跳 hidden state，输出是本轮 greedy 选出的 token id。
struct HiddenToTokenRequest {
    RequestContext context;
    TensorPayload input_tensor;
};

struct HiddenToTokenResponse {
    RequestContext context;
    StatusCode status = StatusCode::kOk;
    std::string message;
    int32_t output_token_id = 0;
};

namespace detail {

inline std::string to_hex(const std::vector<uint8_t>& bytes) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint8_t value : bytes) {
        oss << std::setw(2) << static_cast<int>(value);
    }
    return oss.str();
}

inline bool from_hex(const std::string& hex, std::vector<uint8_t>& out) {
    if (hex.size() % 2 != 0) {
        return false;
    }
    out.clear();
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        if (!std::isxdigit(static_cast<unsigned char>(hex[i])) ||
            !std::isxdigit(static_cast<unsigned char>(hex[i + 1]))) {
            return false;
        }
        uint32_t value = 0;
        std::istringstream iss(hex.substr(i, 2));
        iss >> std::hex >> value;
        out.push_back(static_cast<uint8_t>(value));
    }
    return true;
}

inline std::string escape(const std::string& text) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase;
    for (unsigned char ch : text) {
        const bool keep =
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '-' || ch == '_' || ch == '.' || ch == '~';
        if (keep) {
            oss << static_cast<char>(ch);
        } else {
            oss << '%' << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
        }
    }
    return oss.str();
}

inline bool unescape(const std::string& text, std::string& out) {
    out.clear();
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '%') {
            out.push_back(text[i]);
            continue;
        }
        if (i + 2 >= text.size()) {
            return false;
        }
        if (!std::isxdigit(static_cast<unsigned char>(text[i + 1])) ||
            !std::isxdigit(static_cast<unsigned char>(text[i + 2]))) {
            return false;
        }
        uint32_t value = 0;
        std::istringstream iss(text.substr(i + 1, 2));
        iss >> std::hex >> value;
        out.push_back(static_cast<char>(value));
        i += 2;
    }
    return true;
}

template <typename T>
inline std::string join_numbers(const std::vector<T>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << ',';
        }
        oss << values[i];
    }
    return oss.str();
}

template <typename T>
inline bool parse_numbers(const std::string& text, std::vector<T>& out) {
    out.clear();
    if (text.empty()) {
        return true;
    }
    std::istringstream iss(text);
    std::string item;
    while (std::getline(iss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        std::istringstream item_stream(item);
        long long value = 0;
        item_stream >> value;
        if (!item_stream || !item_stream.eof()) {
            return false;
        }
        out.push_back(static_cast<T>(value));
    }
    return true;
}

inline std::string command_to_string(RpcCommand command) {
    switch (command) {
    case RpcCommand::kPing:
        return "PING";
    case RpcCommand::kResetCache:
        return "RESET_CACHE";
    case RpcCommand::kGenerateTokens:
        return "GENERATE_TOKENS";
    case RpcCommand::kForwardStage:
        return "FORWARD_STAGE";
    case RpcCommand::kTokensToHidden:
        return "TOKENS_TO_HIDDEN";
    case RpcCommand::kHiddenToToken:
        return "HIDDEN_TO_TOKEN";
    default:
        return "UNKNOWN";
    }
}

inline RpcCommand string_to_command(const std::string& text) {
    if (text == "PING") {
        return RpcCommand::kPing;
    }
    if (text == "RESET_CACHE") {
        return RpcCommand::kResetCache;
    }
    if (text == "GENERATE_TOKENS") {
        return RpcCommand::kGenerateTokens;
    }
    if (text == "FORWARD_STAGE") {
        return RpcCommand::kForwardStage;
    }
    if (text == "TOKENS_TO_HIDDEN") {
        return RpcCommand::kTokensToHidden;
    }
    if (text == "HIDDEN_TO_TOKEN") {
        return RpcCommand::kHiddenToToken;
    }
    return RpcCommand::kUnknown;
}

inline std::string status_to_string(StatusCode status) {
    switch (status) {
    case StatusCode::kOk:
        return "OK";
    case StatusCode::kError:
        return "ERROR";
    case StatusCode::kUnsupported:
        return "UNSUPPORTED";
    default:
        return "ERROR";
    }
}

inline StatusCode string_to_status(const std::string& text) {
    if (text == "OK") {
        return StatusCode::kOk;
    }
    if (text == "UNSUPPORTED") {
        return StatusCode::kUnsupported;
    }
    return StatusCode::kError;
}

inline std::string dtype_to_string(TensorDataType dtype) {
    switch (dtype) {
    case TensorDataType::kFloat16:
        return "FP16";
    case TensorDataType::kFloat32:
        return "FP32";
    case TensorDataType::kInt32:
        return "INT32";
    case TensorDataType::kUInt8:
        return "UINT8";
    default:
        return "UNKNOWN";
    }
}

inline TensorDataType string_to_dtype(const std::string& text) {
    if (text == "FP16") {
        return TensorDataType::kFloat16;
    }
    if (text == "FP32") {
        return TensorDataType::kFloat32;
    }
    if (text == "INT32") {
        return TensorDataType::kInt32;
    }
    if (text == "UINT8") {
        return TensorDataType::kUInt8;
    }
    return TensorDataType::kUnknown;
}

inline std::string encode_record(RpcCommand command,
                                 const std::map<std::string, std::string>& kvs) {
    std::ostringstream oss;
    oss << command_to_string(command);
    for (const auto& entry : kvs) {
        oss << "|" << entry.first << "=" << escape(entry.second);
    }
    return oss.str();
}

inline bool decode_record(const std::string& text,
                          RpcCommand& command,
                          std::map<std::string, std::string>& kvs) {
    kvs.clear();
    std::istringstream iss(text);
    std::string part;
    if (!std::getline(iss, part, '|')) {
        return false;
    }
    command = string_to_command(part);
    while (std::getline(iss, part, '|')) {
        const size_t pos = part.find('=');
        if (pos == std::string::npos) {
            return false;
        }
        std::string key = part.substr(0, pos);
        std::string value;
        if (!unescape(part.substr(pos + 1), value)) {
            return false;
        }
        kvs[key] = value;
    }
    return command != RpcCommand::kUnknown;
}

inline std::string get_or_default(const std::map<std::string, std::string>& kvs,
                                  const std::string& key,
                                  const std::string& fallback = "") {
    auto it = kvs.find(key);
    return it == kvs.end() ? fallback : it->second;
}

inline bool parse_int32(const std::map<std::string, std::string>& kvs,
                        const std::string& key,
                        int32_t& out) {
    auto it = kvs.find(key);
    if (it == kvs.end()) {
        return false;
    }
    std::istringstream iss(it->second);
    int64_t value = 0;
    iss >> value;
    if (!iss || !iss.eof()) {
        return false;
    }
    out = static_cast<int32_t>(value);
    return true;
}

inline bool parse_uint64(const std::map<std::string, std::string>& kvs,
                         const std::string& key,
                         uint64_t& out) {
    auto it = kvs.find(key);
    if (it == kvs.end()) {
        return false;
    }
    std::istringstream iss(it->second);
    iss >> out;
    return static_cast<bool>(iss) && iss.eof();
}

inline bool parse_float(const std::map<std::string, std::string>& kvs,
                        const std::string& key,
                        float& out) {
    auto it = kvs.find(key);
    if (it == kvs.end()) {
        return false;
    }
    std::istringstream iss(it->second);
    iss >> out;
    return static_cast<bool>(iss) && iss.eof();
}

inline void put_context(std::map<std::string, std::string>& kvs, const RequestContext& context) {
    kvs["session_id"] = context.session_id;
    kvs["request_id"] = std::to_string(context.request_id);
    kvs["trace_id"] = context.trace_id;
    kvs["priority"] = std::to_string(context.priority);
    kvs["timeout_ms"] = std::to_string(context.timeout_ms);
    kvs["stage_id"] = std::to_string(context.route.stage_id);
    kvs["total_stages"] = std::to_string(context.route.total_stages);
    kvs["hop_index"] = std::to_string(context.route.hop_index);
    kvs["pos_base"] = std::to_string(context.route.pos_base);
    kvs["source_worker_id"] = context.route.source_worker_id;
    kvs["target_worker_id"] = context.route.target_worker_id;
}

inline bool read_context(const std::map<std::string, std::string>& kvs, RequestContext& context) {
    context.session_id = get_or_default(kvs, "session_id");
    context.trace_id = get_or_default(kvs, "trace_id");
    context.route.source_worker_id = get_or_default(kvs, "source_worker_id");
    context.route.target_worker_id = get_or_default(kvs, "target_worker_id");
    return parse_uint64(kvs, "request_id", context.request_id) &&
           parse_int32(kvs, "priority", context.priority) &&
           parse_int32(kvs, "timeout_ms", context.timeout_ms) &&
           parse_int32(kvs, "stage_id", context.route.stage_id) &&
           parse_int32(kvs, "total_stages", context.route.total_stages) &&
           parse_int32(kvs, "hop_index", context.route.hop_index) &&
           parse_int32(kvs, "pos_base", context.route.pos_base);
}

inline void put_tensor(std::map<std::string, std::string>& kvs,
                       const std::string& prefix,
                       const TensorPayload& tensor) {
    kvs[prefix + "_dtype"] = dtype_to_string(tensor.dtype);
    kvs[prefix + "_shape"] = join_numbers(tensor.shape);
    kvs[prefix + "_bytes"] = to_hex(tensor.bytes);
}

inline bool read_tensor(const std::map<std::string, std::string>& kvs,
                        const std::string& prefix,
                        TensorPayload& tensor) {
    tensor.dtype = string_to_dtype(get_or_default(kvs, prefix + "_dtype", "UNKNOWN"));
    if (!parse_numbers(get_or_default(kvs, prefix + "_shape"), tensor.shape)) {
        return false;
    }
    return from_hex(get_or_default(kvs, prefix + "_bytes"), tensor.bytes);
}

}  // namespace detail

inline std::string serialize_generate_request(const GenerateTokensRequest& request) {
    std::map<std::string, std::string> kvs;
    detail::put_context(kvs, request.context);
    kvs["max_new_tokens"] = std::to_string(request.generation.max_new_tokens);
    kvs["max_prompt_chars"] = std::to_string(request.generation.max_prompt_chars);
    kvs["repetition_window"] = std::to_string(request.generation.repetition_window);
    kvs["input_token_ids"] = detail::join_numbers(request.input_token_ids);
    return detail::encode_record(RpcCommand::kGenerateTokens, kvs);
}

inline bool deserialize_generate_request(const std::string& payload, GenerateTokensRequest& request) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    if (!detail::decode_record(payload, command, kvs) ||
        command != RpcCommand::kGenerateTokens) {
        return false;
    }
    if (!detail::read_context(kvs, request.context)) {
        return false;
    }
    return detail::parse_int32(kvs, "max_new_tokens", request.generation.max_new_tokens) &&
           detail::parse_int32(kvs, "max_prompt_chars", request.generation.max_prompt_chars) &&
           detail::parse_int32(kvs, "repetition_window", request.generation.repetition_window) &&
           detail::parse_numbers(detail::get_or_default(kvs, "input_token_ids"),
                                 request.input_token_ids);
}

inline std::string serialize_generate_response(const GenerateTokensResponse& response) {
    std::map<std::string, std::string> kvs;
    detail::put_context(kvs, response.context);
    kvs["status"] = detail::status_to_string(response.status);
    kvs["message"] = response.message;
    kvs["output_token_ids"] = detail::join_numbers(response.output_token_ids);
    kvs["prefill_tokens"] = std::to_string(response.prefill_tokens);
    kvs["decode_tokens"] = std::to_string(response.decode_tokens);
    kvs["prefill_ms"] = std::to_string(response.prefill_ms);
    kvs["decode_ms"] = std::to_string(response.decode_ms);
    kvs["hit_stop"] = response.hit_stop ? "1" : "0";
    kvs["hit_repetition"] = response.hit_repetition ? "1" : "0";
    return detail::encode_record(RpcCommand::kGenerateTokens, kvs);
}

inline bool deserialize_generate_response(const std::string& payload,
                                          GenerateTokensResponse& response) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    if (!detail::decode_record(payload, command, kvs) ||
        command != RpcCommand::kGenerateTokens) {
        return false;
    }
    if (!detail::read_context(kvs, response.context)) {
        return false;
    }
    response.status = detail::string_to_status(detail::get_or_default(kvs, "status", "ERROR"));
    response.message = detail::get_or_default(kvs, "message");
    const bool ok =
        detail::parse_numbers(detail::get_or_default(kvs, "output_token_ids"),
                              response.output_token_ids) &&
        detail::parse_int32(kvs, "prefill_tokens", response.prefill_tokens) &&
        detail::parse_int32(kvs, "decode_tokens", response.decode_tokens) &&
        detail::parse_float(kvs, "prefill_ms", response.prefill_ms) &&
        detail::parse_float(kvs, "decode_ms", response.decode_ms);
    response.hit_stop = detail::get_or_default(kvs, "hit_stop", "0") != "0";
    response.hit_repetition = detail::get_or_default(kvs, "hit_repetition", "0") != "0";
    return ok;
}

inline std::string serialize_stage_request(const StageForwardRequest& request) {
    std::map<std::string, std::string> kvs;
    detail::put_context(kvs, request.context);
    detail::put_tensor(kvs, "input", request.input_tensor);
    return detail::encode_record(RpcCommand::kForwardStage, kvs);
}

inline bool deserialize_stage_request(const std::string& payload, StageForwardRequest& request) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    if (!detail::decode_record(payload, command, kvs) ||
        command != RpcCommand::kForwardStage) {
        return false;
    }
    return detail::read_context(kvs, request.context) &&
           detail::read_tensor(kvs, "input", request.input_tensor);
}

inline std::string serialize_stage_response(const StageForwardResponse& response) {
    std::map<std::string, std::string> kvs;
    detail::put_context(kvs, response.context);
    kvs["status"] = detail::status_to_string(response.status);
    kvs["message"] = response.message;
    detail::put_tensor(kvs, "output", response.output_tensor);
    return detail::encode_record(RpcCommand::kForwardStage, kvs);
}

inline bool deserialize_stage_response(const std::string& payload, StageForwardResponse& response) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    if (!detail::decode_record(payload, command, kvs) ||
        command != RpcCommand::kForwardStage) {
        return false;
    }
    response.status = detail::string_to_status(detail::get_or_default(kvs, "status", "ERROR"));
    response.message = detail::get_or_default(kvs, "message");
    return detail::read_context(kvs, response.context) &&
           detail::read_tensor(kvs, "output", response.output_tensor);
}

inline std::string serialize_tokens_to_hidden_request(const TokensToHiddenRequest& request) {
    std::map<std::string, std::string> kvs;
    detail::put_context(kvs, request.context);
    kvs["input_token_ids"] = detail::join_numbers(request.input_token_ids);
    return detail::encode_record(RpcCommand::kTokensToHidden, kvs);
}

inline bool deserialize_tokens_to_hidden_request(const std::string& payload,
                                                 TokensToHiddenRequest& request) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    if (!detail::decode_record(payload, command, kvs) ||
        command != RpcCommand::kTokensToHidden) {
        return false;
    }
    return detail::read_context(kvs, request.context) &&
           detail::parse_numbers(detail::get_or_default(kvs, "input_token_ids"),
                                 request.input_token_ids);
}

inline std::string serialize_tokens_to_hidden_response(const TokensToHiddenResponse& response) {
    std::map<std::string, std::string> kvs;
    detail::put_context(kvs, response.context);
    kvs["status"] = detail::status_to_string(response.status);
    kvs["message"] = response.message;
    detail::put_tensor(kvs, "output", response.output_tensor);
    return detail::encode_record(RpcCommand::kTokensToHidden, kvs);
}

inline bool deserialize_tokens_to_hidden_response(const std::string& payload,
                                                  TokensToHiddenResponse& response) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    if (!detail::decode_record(payload, command, kvs) ||
        command != RpcCommand::kTokensToHidden) {
        return false;
    }
    response.status = detail::string_to_status(detail::get_or_default(kvs, "status", "ERROR"));
    response.message = detail::get_or_default(kvs, "message");
    return detail::read_context(kvs, response.context) &&
           detail::read_tensor(kvs, "output", response.output_tensor);
}

inline std::string serialize_hidden_to_token_request(const HiddenToTokenRequest& request) {
    std::map<std::string, std::string> kvs;
    detail::put_context(kvs, request.context);
    detail::put_tensor(kvs, "input", request.input_tensor);
    return detail::encode_record(RpcCommand::kHiddenToToken, kvs);
}

inline bool deserialize_hidden_to_token_request(const std::string& payload,
                                                HiddenToTokenRequest& request) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    if (!detail::decode_record(payload, command, kvs) ||
        command != RpcCommand::kHiddenToToken) {
        return false;
    }
    return detail::read_context(kvs, request.context) &&
           detail::read_tensor(kvs, "input", request.input_tensor);
}

inline std::string serialize_hidden_to_token_response(const HiddenToTokenResponse& response) {
    std::map<std::string, std::string> kvs;
    detail::put_context(kvs, response.context);
    kvs["status"] = detail::status_to_string(response.status);
    kvs["message"] = response.message;
    kvs["output_token_id"] = std::to_string(response.output_token_id);
    return detail::encode_record(RpcCommand::kHiddenToToken, kvs);
}

inline bool deserialize_hidden_to_token_response(const std::string& payload,
                                                 HiddenToTokenResponse& response) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    if (!detail::decode_record(payload, command, kvs) ||
        command != RpcCommand::kHiddenToToken) {
        return false;
    }
    response.status = detail::string_to_status(detail::get_or_default(kvs, "status", "ERROR"));
    response.message = detail::get_or_default(kvs, "message");
    return detail::read_context(kvs, response.context) &&
           detail::parse_int32(kvs, "output_token_id", response.output_token_id);
}

inline std::string serialize_ping_request() {
    return detail::encode_record(RpcCommand::kPing, {});
}

inline std::string serialize_reset_cache_request() {
    return detail::encode_record(RpcCommand::kResetCache, {});
}

inline bool is_ping_payload(const std::string& payload) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    return detail::decode_record(payload, command, kvs) && command == RpcCommand::kPing;
}

inline bool is_reset_cache_payload(const std::string& payload) {
    RpcCommand command = RpcCommand::kUnknown;
    std::map<std::string, std::string> kvs;
    return detail::decode_record(payload, command, kvs) && command == RpcCommand::kResetCache;
}

inline std::string serialize_error_response(RpcCommand command,
                                            const RequestContext& context,
                                            StatusCode status,
                                            const std::string& message) {
    std::map<std::string, std::string> kvs;
    detail::put_context(kvs, context);
    kvs["status"] = detail::status_to_string(status);
    kvs["message"] = message;
    return detail::encode_record(command, kvs);
}

inline bool is_success(StatusCode status) {
    return status == StatusCode::kOk;
}

}  // namespace distributed
