#pragma once

#include <cstdint>
#include <memory>
#include <string>

// ============================================================
// 模型随机读取抽象层
//
// 职责：
// 1. 屏蔽“模型来自本地文件还是远端 HTTP”的差异。
// 2. 上层统一按 offset/size 精确读取字节，不关心底层实现。
// 3. 作为 safetensors 解析层和网络/文件访问层之间的稳定边界。
// ============================================================

class IModelReader {
public:
    virtual ~IModelReader() = default;

    // 从模型的绝对字节偏移 offset 开始，精确读取 size 个字节到 out。
    // 读取失败时返回 false，并把错误描述写入 last_error()。
    virtual bool read_exact(int64_t offset, size_t size, void* out) = 0;

    // 返回当前 reader 对应的数据源描述，便于日志输出。
    virtual std::string describe() const = 0;

    // 返回最近一次失败的错误信息。
    virtual std::string last_error() const = 0;
};

// 根据 locator 创建合适的 reader。
// 支持：
// 1. 本地 safetensors 路径
// 2. http(s)://.../model.safetensors
std::unique_ptr<IModelReader> create_model_reader(const std::string& locator);

// 按 locator 复用 reader。
// 这样同一次模型加载过程中，多次张量读取可以共享底层 reader 的状态，
// 比如远端 HTTP Range 的块缓存，而不是每次重新建一个 reader。
IModelReader& get_cached_model_reader(const std::string& locator);
