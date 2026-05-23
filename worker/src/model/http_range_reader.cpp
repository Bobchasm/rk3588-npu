#include "model/http_range_reader.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <netdb.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

bool http_debug_enabled() {
    const char* env = std::getenv("RKLLM_HTTP_DEBUG");
    return env && env[0] != '\0' && !(env[0] == '0' && env[1] == '\0');
}

void http_debug_log(const std::string& message) {
    if (http_debug_enabled()) {
        std::fprintf(stderr, "[HttpRangeReader] %s\n", message.c_str());
    }
}

std::size_t env_to_size_t_mb(const char* name, std::size_t default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (!end || *end != '\0' || value == 0) {
        return default_value;
    }
    return static_cast<std::size_t>(value);
}

std::size_t http_cache_block_bytes() {
    return env_to_size_t_mb("RKLLM_HTTP_CACHE_BLOCK_MB", 1) * 1024ULL * 1024ULL;
}

std::size_t http_cache_max_blocks() {
    return env_to_size_t_mb("RKLLM_HTTP_CACHE_MAX_BLOCKS", 16);
}

std::string shell_escape_single_quotes(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() + 8);
    escaped.push_back('\'');
    for (char ch : text) {
        if (ch == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

std::string find_command_path(const std::string& name) {
    if (name.empty()) {
        return "";
    }
    if (name.find('/') != std::string::npos) {
        return ::access(name.c_str(), X_OK) == 0 ? name : "";
    }

    const char* path_env = std::getenv("PATH");
    if (path_env && path_env[0] != '\0') {
        std::stringstream ss(path_env);
        std::string dir;
        while (std::getline(ss, dir, ':')) {
            if (dir.empty()) {
                dir = ".";
            }
            const std::string full = dir + "/" + name;
            if (::access(full.c_str(), X_OK) == 0) {
                return full;
            }
        }
    }

    static const char* kCommonDirs[] = {
        "/usr/bin",
        "/bin",
        "/usr/local/bin",
        "/sbin",
        "/usr/sbin",
    };
    for (const char* dir : kCommonDirs) {
        const std::string full = std::string(dir) + "/" + name;
        if (::access(full.c_str(), X_OK) == 0) {
            return full;
        }
    }
    return "";
}

struct ParsedHttpUrl {
    std::string host;
    std::string port;
    std::string path;
};

struct RangeResponseInfo {
    int64_t range_begin = -1;
    int64_t range_end = -1;
    int64_t total_size = -1;
};

bool parse_http_url(const std::string& url, ParsedHttpUrl& parsed, std::string& error) {
    const std::string prefix = "http://";
    if (!starts_with(url, prefix)) {
        error = "only plain http:// URLs can use the built-in socket fallback";
        return false;
    }

    std::string remainder = url.substr(prefix.size());
    std::size_t slash_pos = remainder.find('/');
    std::string host_port = slash_pos == std::string::npos ? remainder : remainder.substr(0, slash_pos);
    parsed.path = slash_pos == std::string::npos ? "/" : remainder.substr(slash_pos);

    if (host_port.empty()) {
        error = "empty host in http url";
        return false;
    }

    std::size_t colon_pos = host_port.rfind(':');
    if (colon_pos == std::string::npos) {
        parsed.host = host_port;
        parsed.port = "80";
    } else {
        parsed.host = host_port.substr(0, colon_pos);
        parsed.port = host_port.substr(colon_pos + 1);
    }

    if (parsed.host.empty() || parsed.port.empty()) {
        error = "bad host/port in http url";
        return false;
    }
    return true;
}

bool parse_content_range_header(const std::string& header,
                                RangeResponseInfo& info,
                                std::string& error) {
    const std::string key = "Content-Range:";
    std::size_t pos = header.find(key);
    if (pos == std::string::npos) {
        pos = header.find("Content-range:");
    }
    if (pos == std::string::npos) {
        pos = header.find("content-range:");
    }
    if (pos == std::string::npos) {
        error = "missing Content-Range header";
        return false;
    }

    std::size_t line_end = header.find("\r\n", pos);
    std::string line = line_end == std::string::npos
        ? header.substr(pos)
        : header.substr(pos, line_end - pos);

    std::size_t bytes_pos = line.find("bytes ");
    if (bytes_pos == std::string::npos) {
        error = "bad Content-Range header: " + line;
        return false;
    }

    std::string payload = line.substr(bytes_pos + 6);
    std::size_t dash_pos = payload.find('-');
    std::size_t slash_pos = payload.find('/');
    if (dash_pos == std::string::npos || slash_pos == std::string::npos || dash_pos >= slash_pos) {
        error = "bad Content-Range payload: " + payload;
        return false;
    }

    try {
        info.range_begin = std::stoll(payload.substr(0, dash_pos));
        info.range_end = std::stoll(payload.substr(dash_pos + 1, slash_pos - dash_pos - 1));
        info.total_size = std::stoll(payload.substr(slash_pos + 1));
    } catch (const std::exception&) {
        error = "failed to parse Content-Range payload: " + payload;
        return false;
    }

    return true;
}

bool socket_read_range_http(const std::string& url,
                            int64_t offset,
                            size_t size,
                            void* out,
                            std::size_t* actual_bytes_out,
                            int64_t* total_size_out,
                            std::string& error) {
    ParsedHttpUrl parsed;
    if (!parse_http_url(url, parsed, error)) {
        return false;
    }

    http_debug_log("use built-in http socket fallback");
    http_debug_log("socket host=" + parsed.host + " port=" + parsed.port + " path=" + parsed.path);

    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo* result = nullptr;
    const int gai_ret = ::getaddrinfo(parsed.host.c_str(), parsed.port.c_str(), &hints, &result);
    if (gai_ret != 0) {
        error = "getaddrinfo failed: " + std::string(::gai_strerror(gai_ret));
        return false;
    }

    int sock_fd = -1;
    for (struct addrinfo* it = result; it != nullptr; it = it->ai_next) {
        sock_fd = ::socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (sock_fd < 0) {
            continue;
        }
        if (::connect(sock_fd, it->ai_addr, it->ai_addrlen) == 0) {
            break;
        }
        ::close(sock_fd);
        sock_fd = -1;
    }
    ::freeaddrinfo(result);

    if (sock_fd < 0) {
        error = "connect failed: " + std::string(std::strerror(errno));
        return false;
    }

    const long long begin = static_cast<long long>(offset);
    const long long end = static_cast<long long>(offset + static_cast<int64_t>(size) - 1);
    const std::string request =
        "GET " + parsed.path + " HTTP/1.1\r\n"
        "Host: " + parsed.host + "\r\n"
        "Range: bytes=" + std::to_string(begin) + "-" + std::to_string(end) + "\r\n"
        "Connection: close\r\n\r\n";

    std::size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t n = ::send(sock_fd, request.data() + sent, request.size() - sent, 0);
        if (n <= 0) {
            error = "send failed: " + std::string(std::strerror(errno));
            ::close(sock_fd);
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }

    std::string header;
    char ch = '\0';
    while (header.find("\r\n\r\n") == std::string::npos) {
        const ssize_t n = ::recv(sock_fd, &ch, 1, 0);
        if (n <= 0) {
            error = "recv header failed: " + std::string(std::strerror(errno));
            ::close(sock_fd);
            return false;
        }
        header.push_back(ch);
        if (header.size() > 64 * 1024) {
            error = "http header too large";
            ::close(sock_fd);
            return false;
        }
    }
    http_debug_log("response_header=" + header);

    if (header.find("HTTP/1.1 206") == std::string::npos &&
        header.find("HTTP/1.0 206") == std::string::npos) {
        error = "range request was not honored, expected HTTP 206";
        ::close(sock_fd);
        return false;
    }

    RangeResponseInfo range_info;
    if (!parse_content_range_header(header, range_info, error)) {
        ::close(sock_fd);
        return false;
    }
    if (total_size_out) {
        *total_size_out = range_info.total_size;
    }

    const std::size_t expected_body =
        static_cast<std::size_t>(range_info.range_end - range_info.range_begin + 1);
    if (expected_body > size) {
        error = "server returned more bytes than requested";
        ::close(sock_fd);
        return false;
    }

    auto* bytes = static_cast<unsigned char*>(out);
    std::size_t total = 0;
    while (total < expected_body) {
        const ssize_t n = ::recv(sock_fd, bytes + total, expected_body - total, 0);
        if (n < 0) {
            error = "recv body failed: " + std::string(std::strerror(errno));
            ::close(sock_fd);
            return false;
        }
        if (n == 0) {
            break;
        }
        total += static_cast<std::size_t>(n);
    }
    ::close(sock_fd);

    http_debug_log("socket_body_bytes=" + std::to_string(total) +
                   " expected=" + std::to_string(expected_body));
    if (total != expected_body) {
        error = "short socket http read, expected=" + std::to_string(expected_body) +
                ", actual=" + std::to_string(total);
        return false;
    }
    if (actual_bytes_out) {
        *actual_bytes_out = total;
    }
    return true;
}

bool http_command_read_range(const std::string& url,
                             int64_t offset,
                             size_t size,
                             void* out,
                             std::string& error) {
    const long long begin = static_cast<long long>(offset);
    const long long end = static_cast<long long>(offset + static_cast<int64_t>(size) - 1);
    const std::string escaped_url = shell_escape_single_quotes(url);
    const std::string curl_path = find_command_path("curl");
    const std::string wget_path = find_command_path("wget");
    const char* path_env = std::getenv("PATH");

    http_debug_log("url=" + url +
                   " offset=" + std::to_string(begin) +
                   " end=" + std::to_string(end) +
                   " size=" + std::to_string(size));
    http_debug_log("PATH=" + std::string(path_env ? path_env : "<empty>"));
    http_debug_log("curl_path=" + (curl_path.empty() ? std::string("<not found>") : curl_path));
    http_debug_log("wget_path=" + (wget_path.empty() ? std::string("<not found>") : wget_path));

    std::string cmd;
    if (!curl_path.empty()) {
        cmd = shell_escape_single_quotes(curl_path) +
              " -L --fail --silent --show-error --retry 2 --connect-timeout 10 "
              "-r " + std::to_string(begin) + "-" + std::to_string(end) + " " + escaped_url;
    } else if (!wget_path.empty()) {
        const std::string range_header =
            "Range: bytes=" + std::to_string(begin) + "-" + std::to_string(end);
        cmd = shell_escape_single_quotes(wget_path) +
              " -q -O - --header=" + shell_escape_single_quotes(range_header) +
              " " + escaped_url;
    } else {
        error = "neither curl nor wget is available for HTTP range read; "
                "PATH=" + std::string(path_env ? path_env : "<empty>");
        return false;
    }
    http_debug_log("command=" + cmd);

    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (!pipe) {
        error = "failed to start HTTP range command";
        return false;
    }

    std::size_t total = 0;
    auto* bytes = static_cast<unsigned char*>(out);
    while (total < size) {
        const std::size_t n = std::fread(bytes + total, 1, size - total, pipe);
        if (n == 0) {
            break;
        }
        total += n;
    }

    const int status = ::pclose(pipe);
    http_debug_log("bytes_read=" + std::to_string(total) +
                   " expected=" + std::to_string(size) +
                   " pclose_status=" + std::to_string(status));
    if (status != 0) {
        error = "HTTP range command failed for url: " + url +
                ", status=" + std::to_string(status);
        return false;
    }
    if (total != size) {
        error = "short HTTP range read for url: " + url +
                ", expected=" + std::to_string(size) +
                ", actual=" + std::to_string(total);
        return false;
    }
    return true;
}

class HttpRangeReader final : public IModelReader {
public:
    explicit HttpRangeReader(std::string url)
        : url_(std::move(url)),
          block_bytes_(http_cache_block_bytes()),
          max_blocks_(http_cache_max_blocks()) {}

    bool read_exact(int64_t offset, size_t size, void* out) override {
        last_error_.clear();

        if (size == 0) {
            return true;
        }

        auto* dst = static_cast<unsigned char*>(out);
        std::size_t copied = 0;
        while (copied < size) {
            const int64_t current_offset = offset + static_cast<int64_t>(copied);
            const std::size_t block_offset =
                static_cast<std::size_t>(current_offset % static_cast<int64_t>(block_bytes_));
            CacheBlock* block = get_or_fetch_block(current_offset);
            if (!block) {
                return false;
            }

            const std::size_t available = block->data.size() - block_offset;
            const std::size_t need = size - copied;
            const std::size_t take = available < need ? available : need;
            std::memcpy(dst + copied, block->data.data() + block_offset, take);
            copied += take;
        }
        return true;
    }

    std::string describe() const override {
        return url_;
    }

    std::string last_error() const override {
        return last_error_;
    }

private:
    struct CacheBlock {
        int64_t block_index = 0;
        std::vector<unsigned char> data;
    };

    CacheBlock* get_or_fetch_block(int64_t absolute_offset) {
        const int64_t block_index = absolute_offset / static_cast<int64_t>(block_bytes_);
        auto hit = block_map_.find(block_index);
        if (hit != block_map_.end()) {
            lru_.splice(lru_.begin(), lru_, hit->second);
            http_debug_log("cache hit block=" + std::to_string(block_index));
            return &(*lru_.begin());
        }

        http_debug_log("cache miss block=" + std::to_string(block_index));
        CacheBlock block;
        block.block_index = block_index;

        std::string fetch_error;
        const int64_t block_start = block_index * static_cast<int64_t>(block_bytes_);
        if (remote_file_size_ >= 0 && block_start >= remote_file_size_) {
            last_error_ = "block start exceeds remote file size";
            return nullptr;
        }

        std::size_t fetch_size = block_bytes_;
        if (remote_file_size_ >= 0) {
            const int64_t remain = remote_file_size_ - block_start;
            if (remain <= 0) {
                last_error_ = "block start exceeds remote file size";
                return nullptr;
            }
            if (remain < static_cast<int64_t>(fetch_size)) {
                fetch_size = static_cast<std::size_t>(remain);
            }
        }
        block.data.resize(fetch_size);

        if (!http_command_read_range(url_, block_start, fetch_size, block.data.data(), fetch_error)) {
            if (starts_with(url_, "http://")) {
                fetch_error.clear();
                std::size_t actual_bytes = 0;
                int64_t total_size = -1;
                if (!socket_read_range_http(url_,
                                            block_start,
                                            fetch_size,
                                            block.data.data(),
                                            &actual_bytes,
                                            &total_size,
                                            fetch_error)) {
                    http_debug_log("socket fallback failed: " + fetch_error);
                    last_error_ = fetch_error;
                    return nullptr;
                }
                if (total_size >= 0) {
                    remote_file_size_ = total_size;
                }
                block.data.resize(actual_bytes);
            } else {
                last_error_ = fetch_error;
                return nullptr;
            }
        }

        lru_.push_front(std::move(block));
        block_map_[block_index] = lru_.begin();
        evict_if_needed();
        return &(*lru_.begin());
    }

    void evict_if_needed() {
        while (lru_.size() > max_blocks_) {
            auto last = lru_.end();
            --last;
            http_debug_log("evict block=" + std::to_string(last->block_index));
            block_map_.erase(last->block_index);
            lru_.pop_back();
        }
    }

    std::string url_;
    std::string last_error_;
    std::size_t block_bytes_;
    std::size_t max_blocks_;
    int64_t remote_file_size_ = -1;
    std::list<CacheBlock> lru_;
    std::unordered_map<int64_t, std::list<CacheBlock>::iterator> block_map_;
};

}  // namespace

std::unique_ptr<IModelReader> create_http_range_reader(const std::string& url) {
    return std::unique_ptr<IModelReader>(new HttpRangeReader(url));
}
