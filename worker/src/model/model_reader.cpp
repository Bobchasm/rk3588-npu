#include "model/model_reader.h"
#include "model/http_range_reader.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <utility>

namespace {

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

bool is_http_url(const std::string& locator) {
    return starts_with(locator, "http://") || starts_with(locator, "https://");
}

class LocalFileReader final : public IModelReader {
public:
    explicit LocalFileReader(std::string path) : path_(std::move(path)) {}

    bool read_exact(int64_t offset, size_t size, void* out) override {
        last_error_.clear();

        FILE* fp = std::fopen(path_.c_str(), "rb");
        if (!fp) {
            last_error_ = "cannot open local file: " + path_;
            return false;
        }

        if (std::fseek(fp, static_cast<long>(offset), SEEK_SET) != 0) {
            std::fclose(fp);
            last_error_ = "fseek failed for local file: " + path_;
            return false;
        }

        const size_t read_n = std::fread(out, 1, size, fp);
        std::fclose(fp);
        if (read_n != size) {
            last_error_ = "short read on local file: " + path_;
            return false;
        }
        return true;
    }

    std::string describe() const override {
        return path_;
    }

    std::string last_error() const override {
        return last_error_;
    }

private:
    std::string path_;
    std::string last_error_;
};

}  // namespace

std::unique_ptr<IModelReader> create_model_reader(const std::string& locator) {
    if (is_http_url(locator)) {
        return create_http_range_reader(locator);
    }
    return std::unique_ptr<IModelReader>(new LocalFileReader(locator));
}

IModelReader& get_cached_model_reader(const std::string& locator) {
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<IModelReader>> readers;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = readers.find(locator);
    if (it != readers.end()) {
        return *it->second;
    }

    std::shared_ptr<IModelReader> reader = create_model_reader(locator);
    IModelReader& ref = *reader;
    readers[locator] = std::move(reader);
    return ref;
}
