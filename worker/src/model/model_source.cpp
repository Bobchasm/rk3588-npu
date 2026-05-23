#include "model/model_source.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace {

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_http_url(const std::string& locator) {
    return starts_with(locator, "http://") || starts_with(locator, "https://");
}

bool looks_like_explicit_path(const std::string& locator) {
    return !locator.empty() &&
           (locator[0] == '/' ||
            starts_with(locator, "./") ||
            starts_with(locator, "../"));
}

bool is_regular_file(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool is_directory(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string join_path(const std::string& lhs, const std::string& rhs) {
    if (lhs.empty()) {
        return rhs;
    }
    if (lhs.back() == '/') {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

std::string env_or_default(const char* name, const char* fallback) {
    const char* env = std::getenv(name);
    if (env && env[0] != '\0') {
        return env;
    }
    return fallback;
}

std::string model_source_mode() {
    return env_or_default("RKLLM_MODEL_SOURCE_MODE", "local");
}

std::string local_root_dir() {
    return env_or_default("RKLLM_MODEL_LOCAL_ROOT", ".");
}

std::string remote_base_url() {
    return env_or_default("RKLLM_MODEL_REMOTE_BASE", "");
}

ResolvedModelSource resolve_local_source(const std::string& locator) {
    ResolvedModelSource resolved;
    resolved.original_locator = locator;

    if (is_directory(locator)) {
        resolved.kind = ModelSourceKind::LocalDirectory;
        resolved.resolved_file_path = join_path(locator, "model.safetensors");
        if (!is_regular_file(resolved.resolved_file_path)) {
            throw std::runtime_error("model.safetensors not found under directory: " + locator);
        }
        return resolved;
    }

    if (is_regular_file(locator) && ends_with(locator, ".safetensors")) {
        resolved.kind = ModelSourceKind::LocalFile;
        resolved.resolved_file_path = locator;
        return resolved;
    }

    throw std::runtime_error("invalid local model locator: " + locator);
}

ResolvedModelSource resolve_remote_source(const std::string& locator) {
    ResolvedModelSource resolved;
    resolved.kind = ModelSourceKind::RemoteHttpFile;
    resolved.original_locator = locator;
    resolved.remote_url = locator;
    return resolved;
}

ResolvedModelSource resolve_logical_local_source(const std::string& locator) {
    const std::string full_dir = join_path(local_root_dir(), locator);
    return resolve_local_source(full_dir);
}

ResolvedModelSource resolve_logical_remote_source(const std::string& locator) {
    const std::string base = remote_base_url();
    if (base.empty()) {
        throw std::runtime_error(
            "RKLLM_MODEL_REMOTE_BASE is empty while RKLLM_MODEL_SOURCE_MODE=http");
    }

    std::string normalized = base;
    while (!normalized.empty() && normalized.back() == '/') {
        normalized.pop_back();
    }

    return resolve_remote_source(normalized + "/" + locator + "/model.safetensors");
}

}  // namespace

ResolvedModelSource resolve_model_source(const std::string& model_locator) {
    // 1. 显式 URL：优先按远端单文件处理
    if (is_http_url(model_locator)) {
        return resolve_remote_source(model_locator);
    }

    // 2. 显式本地路径或已存在本地文件/目录：保持旧行为不变
    if (looks_like_explicit_path(model_locator) ||
        is_directory(model_locator) ||
        is_regular_file(model_locator)) {
        return resolve_local_source(model_locator);
    }

    // 3. 逻辑模型名：由配置决定映射到本地目录还是远端文件服务
    const std::string mode = model_source_mode();
    if (mode == "http" || mode == "HTTP") {
        return resolve_logical_remote_source(model_locator);
    }
    if (mode == "local" || mode == "LOCAL") {
        return resolve_logical_local_source(model_locator);
    }

    throw std::runtime_error(
        "invalid RKLLM_MODEL_SOURCE_MODE=" + mode + ", expected local or http");
}
