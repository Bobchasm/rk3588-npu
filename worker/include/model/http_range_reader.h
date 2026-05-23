#pragma once

#include "model/model_reader.h"

#include <memory>
#include <string>

// ============================================================
// HttpRangeReader 工厂
//
// 职责：
// 1. 负责 HTTP(S) 远端模型的随机读取实现。
// 2. 内部封装命令行 curl/wget、原生 http socket fallback、块缓存等细节。
// 3. 对上层仅暴露 IModelReader 接口，避免网络逻辑泄漏到权重解析层。
// ============================================================

std::unique_ptr<IModelReader> create_http_range_reader(const std::string& url);
