#include "ops/op_sampling.h"
#include <cfloat>

int op_greedy_sample(const std::vector<float>& logits) {
    int best = 0;
    float best_val = -FLT_MAX;
    for (int i = 0; i < (int)logits.size(); ++i) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}
