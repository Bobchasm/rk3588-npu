#include "api/env_preset.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

bool env_set(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0';
}

void set_default_env(const char* name, const char* value) {
    if (!env_set(name)) {
        setenv(name, value, 0);
    }
}

bool str_eq(const char* a, const char* b) {
    return std::strcmp(a, b) == 0;
}

}  // namespace

void apply_rkllm_env_preset() {
    const char* preset = std::getenv("RKLLM_PRESET");
    if (!preset || preset[0] == '\0' || str_eq(preset, "custom")) {
        return;
    }

    if (str_eq(preset, "accurate")) {
        set_default_env("RKLLM_LINEAR_BATCH", "30");
        set_default_env("RKLLM_SHARD_DYNAMIC_M", "1");
        set_default_env("RKLLM_NPU_WEIGHT_DTYPE", "fp16");
        return;
    }

    if (str_eq(preset, "balanced")) {
        set_default_env("RKLLM_LINEAR_BATCH", "30");
        set_default_env("RKLLM_SHARD_DYNAMIC_M", "1");
        set_default_env("RKLLM_NPU_WEIGHT_DTYPE", "int8");
        set_default_env("RKLLM_NPU_INT8_SCOPE", "gate_up,attn");
        set_default_env("RKLLM_A8W8_NEON_QUANT", "1");
        return;
    }

    if (str_eq(preset, "fast")) {
        set_default_env("RKLLM_LINEAR_BATCH", "30");
        set_default_env("RKLLM_SHARD_DYNAMIC_M", "1");
        set_default_env("RKLLM_NPU_WEIGHT_DTYPE", "int8");
        set_default_env("RKLLM_NPU_INT8_SCOPE", "all");
        set_default_env("RKLLM_A8W8_NEON_QUANT", "1");
        return;
    }

    std::fprintf(stderr,
                 "[env_preset] unknown RKLLM_PRESET=%s, use existing environment\n",
                 preset);
}
