#include "backend/npu_weight_cache.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

namespace npu_weight_cache {
namespace {

constexpr uint32_t kCacheVersion = 1;

// 文件头保存所有会影响 native B 解释方式的字段。即使文件名碰巧相同，
// read() 也会再次校验 header，防止不同形状/不同量化配置混用。
struct CacheHeader {
    char magic[8];
    uint32_t version;
    uint32_t kind;
    int32_t K;
    int32_t N;
    int32_t K_matmul;
    uint32_t flags;
    uint64_t packed_bytes;
    uint64_t aux_bytes;
};

struct CacheState {
    bool configured = false;
    bool enabled = false;
    bool profile = false;
    std::string root_dir;
    std::string model_stamp;
};

struct CacheStats {
    int hits = 0;
    int misses = 0;
    int writes = 0;
    int write_failures = 0;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    int64_t read_us = 0;
    int64_t write_us = 0;
};

CacheState& state() {
    static CacheState s;
    return s;
}

CacheStats& stats() {
    static CacheStats s;
    return s;
}

int64_t now_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec;
}

bool env_disabled(const char* v) {
    return v && (std::strcmp(v, "0") == 0 ||
                 std::strcmp(v, "false") == 0 ||
                 std::strcmp(v, "FALSE") == 0 ||
                 std::strcmp(v, "off") == 0 ||
                 std::strcmp(v, "OFF") == 0);
}

bool env_enabled(const char* v) {
    return v && (std::strcmp(v, "1") == 0 ||
                 std::strcmp(v, "true") == 0 ||
                 std::strcmp(v, "TRUE") == 0 ||
                 std::strcmp(v, "on") == 0 ||
                 std::strcmp(v, "ON") == 0);
}

std::string sanitize(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char ch : in) {
        const unsigned char c = (unsigned char)ch;
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '.' || c == '-' || c == '_') {
            out.push_back((char)c);
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? std::string("unnamed") : out;
}

bool ensure_dir(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    if (::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

bool ensure_cache_dirs() {
    const CacheState& s = state();
    if (!ensure_dir(s.root_dir)) {
        return false;
    }
    return ensure_dir(s.root_dir + "/" + s.model_stamp);
}

std::string path_for(const CacheSpec& spec) {
    const CacheState& s = state();
    char suffix[160];
    // packed_bytes/aux_bytes 不放文件名，避免 cold path 需要先创建 RKNN context
    // 才知道 tensor size；这些字段仍写在 header 里由 read() 严格校验。
    std::snprintf(suffix, sizeof(suffix),
                  ".kind%u.k%d.n%d.km%d.f%08x.bin",
                  spec.kind, spec.K, spec.N, spec.K_matmul, spec.flags);
    return s.root_dir + "/" + s.model_stamp + "/" + sanitize(spec.key) + suffix;
}

bool read_exact(FILE* fp, void* dst, size_t bytes) {
    if (bytes == 0) {
        return true;
    }
    return std::fread(dst, 1, bytes, fp) == bytes;
}

bool write_exact(FILE* fp, const void* src, size_t bytes) {
    if (bytes == 0) {
        return true;
    }
    return std::fwrite(src, 1, bytes, fp) == bytes;
}

bool header_matches(const CacheHeader& h, const CacheSpec& spec,
                    size_t packed_bytes, size_t aux_bytes) {
    return std::memcmp(h.magic, "RKNCB01", 8) == 0 &&
           h.version == kCacheVersion &&
           h.kind == spec.kind &&
           h.K == spec.K &&
           h.N == spec.N &&
           h.K_matmul == spec.K_matmul &&
           h.flags == spec.flags &&
           h.packed_bytes == (uint64_t)packed_bytes &&
           h.aux_bytes == (uint64_t)aux_bytes;
}

}  // namespace

void configure_for_model(const std::string& model_dir,
                         const std::string& model_path) {
    CacheState& s = state();
    s.configured = true;
    s.enabled = !env_disabled(std::getenv("RKLLM_NATIVE_CACHE"));
    s.profile = env_enabled(std::getenv("RKLLM_LOAD_PROFILE")) ||
                env_enabled(std::getenv("RKLLM_NATIVE_CACHE_PROFILE"));

    const char* override_dir = std::getenv("RKLLM_NATIVE_CACHE_DIR");
    s.root_dir = (override_dir && override_dir[0] != '\0')
        ? std::string(override_dir)
        : model_dir + "/.rknn_native_cache";

    struct stat st {};
    if (::stat(model_path.c_str(), &st) != 0) {
        s.enabled = false;
        s.model_stamp = "unknown_model";
        return;
    }

    // 用 safetensors 文件大小和 mtime 做模型版本戳。它不是加密哈希，
    // 但足够区分本项目常见的模型文件替换和重新导出。
    char stamp[128];
    std::snprintf(stamp, sizeof(stamp),
                  "model_safetensors_%llu_%llu",
                  (unsigned long long)st.st_size,
                  (unsigned long long)st.st_mtime);
    s.model_stamp = stamp;
}

bool enabled() {
    const CacheState& s = state();
    return s.configured && s.enabled && !s.root_dir.empty() && !s.model_stamp.empty();
}

bool load_profile_enabled() {
    return state().profile;
}

bool exists(const CacheSpec& spec) {
    if (!enabled()) {
        return false;
    }
    const std::string path = path_for(spec);
    if (::access(path.c_str(), R_OK) == 0) {
        return true;
    }
    ++stats().misses;
    return false;
}

bool read(const CacheSpec& spec,
          void* packed_dst,
          size_t packed_bytes,
          void* aux_dst,
          size_t aux_bytes) {
    CacheStats& st = stats();
    if (!enabled() || !packed_dst || packed_bytes == 0 ||
        packed_bytes != spec.packed_bytes || aux_bytes != spec.aux_bytes ||
        (aux_bytes > 0 && !aux_dst)) {
        ++st.misses;
        return false;
    }

    const std::string path = path_for(spec);
    const int64_t t0 = now_us();
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        ++st.misses;
        return false;
    }

    // 先读 header 再读 payload。header 不匹配时视为 miss，调用方回退到
    // 原始 safetensors 加载并重写缓存。
    CacheHeader h {};
    const bool ok = read_exact(fp, &h, sizeof(h)) &&
                    header_matches(h, spec, packed_bytes, aux_bytes) &&
                    read_exact(fp, packed_dst, packed_bytes) &&
                    read_exact(fp, aux_dst, aux_bytes);
    std::fclose(fp);

    st.read_us += now_us() - t0;
    if (!ok) {
        ++st.misses;
        return false;
    }

    ++st.hits;
    st.bytes_read += (uint64_t)packed_bytes + (uint64_t)aux_bytes;
    return true;
}

void write(const CacheSpec& spec,
           const void* packed_src,
           size_t packed_bytes,
           const void* aux_src,
           size_t aux_bytes) {
    CacheStats& st = stats();
    if (!enabled() || !packed_src || packed_bytes == 0 ||
        packed_bytes != spec.packed_bytes || aux_bytes != spec.aux_bytes ||
        (aux_bytes > 0 && !aux_src)) {
        return;
    }
    if (!ensure_cache_dirs()) {
        ++st.write_failures;
        return;
    }

    const std::string path = path_for(spec);
    // 先写临时文件再 rename，避免程序中途退出留下半截缓存被下次误读。
    const std::string tmp = path + ".tmp." + std::to_string((long long)::getpid());
    const int64_t t0 = now_us();
    FILE* fp = std::fopen(tmp.c_str(), "wb");
    if (!fp) {
        ++st.write_failures;
        return;
    }

    CacheHeader h {};
    std::memcpy(h.magic, "RKNCB01", 8);
    h.version = kCacheVersion;
    h.kind = spec.kind;
    h.K = spec.K;
    h.N = spec.N;
    h.K_matmul = spec.K_matmul;
    h.flags = spec.flags;
    h.packed_bytes = (uint64_t)packed_bytes;
    h.aux_bytes = (uint64_t)aux_bytes;

    const bool ok = write_exact(fp, &h, sizeof(h)) &&
                    write_exact(fp, packed_src, packed_bytes) &&
                    write_exact(fp, aux_src, aux_bytes) &&
                    std::fflush(fp) == 0;
    std::fclose(fp);

    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        ++st.write_failures;
        return;
    }

    st.write_us += now_us() - t0;
    ++st.writes;
    st.bytes_written += (uint64_t)packed_bytes + (uint64_t)aux_bytes;
}

void print_summary_if_enabled() {
    if (!load_profile_enabled()) {
        return;
    }
    const CacheStats& st = stats();
    std::fprintf(stderr,
                 "[load] native_cache hits=%d misses=%d writes=%d write_failures=%d "
                 "read=%.2f MB %.2f ms write=%.2f MB %.2f ms\n",
                 st.hits, st.misses, st.writes, st.write_failures,
                 (double)st.bytes_read / (1024.0 * 1024.0),
                 (double)st.read_us / 1000.0,
                 (double)st.bytes_written / (1024.0 * 1024.0),
                 (double)st.write_us / 1000.0);
}

}  // namespace npu_weight_cache
