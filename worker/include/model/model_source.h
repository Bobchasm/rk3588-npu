#pragma once

#include <string>

// ============================================================
// ModelSource: 模型来源适配层
//
// 目标：
// 1. 将“模型权重文件从哪里来”与“如何解析 safetensors”解耦。
// 2. 兼容板子本地目录、本地单文件、网络挂载目录、HTTP(S) 远端文件。
// 3. 对现有 Qwen2Model 保持最小侵入：上层只关心“模型来自哪里”，
//    具体是本地文件还是远端 HTTP Range 读取，由底层 reader 决定。
//
// 当前支持的 locator 形式：
//   - 本地目录：/root/matmul/worker_test/Qwen1.5B
//       -> 解析为 /root/.../Qwen1.5B/model.safetensors
//   - 本地单文件：/root/matmul/worker_test/Qwen1.5B/model.safetensors
//   - 远端单文件：http://host:8000/model.safetensors
//       -> 直接记录 URL，后续由 weight_loader 用 HTTP Range 随机读取
//   - 逻辑模型名：Qwen1.5B
//       -> 由环境变量决定映射到本地目录或远端 URL
//
// 说明：
//   safetensors 的读取模式本质是“随机读若干 byte range”。
//   因此这里不再强制把远端模型整文件下载到板子本地，而是把“来源信息”
//   交给 weight_loader，再由后者决定使用本地 fread 还是 HTTP Range。
//
// 配置约定：
//   - RKLLM_MODEL_SOURCE_MODE = local | http
//       控制逻辑模型名的解析方式，默认 local。
//   - RKLLM_MODEL_LOCAL_ROOT = /root/matmul/worker_test
//       当 mode=local 时，逻辑模型名会映射为 <LOCAL_ROOT>/<model_name>
//   - RKLLM_MODEL_REMOTE_BASE = https://bobchasm.cn/rk3588-files
//       当 mode=http 时，逻辑模型名会映射为
//       <REMOTE_BASE>/<model_name>/model.safetensors
// ============================================================

enum class ModelSourceKind {
    LocalDirectory,
    LocalFile,
    RemoteHttpFile,
};

struct ResolvedModelSource {
    ModelSourceKind kind = ModelSourceKind::LocalDirectory;
    std::string original_locator;      // 用户传入的原始 locator
    std::string resolved_file_path;    // 本地模式下最终使用的 safetensors 路径
    std::string remote_url;            // 远端模式下最终使用的 HTTP(S) URL
    bool used_cache = false;           // 预留字段：后续若引入分块缓存可复用
};

// 解析模型来源。
// 本地模式下返回 resolved_file_path；
// 远端模式下返回 remote_url。
// 失败时抛出 std::runtime_error。
ResolvedModelSource resolve_model_source(const std::string& model_locator);
