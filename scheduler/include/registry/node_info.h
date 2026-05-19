#pragma once

#include "scheduler_types.h"
#include <string>

struct WorkerNodeInfo {
    WorkerId worker_id;
    std::string endpoint;
    bool supports_full_model = true;
    bool supports_stage = false;
    int max_pending_requests = 1;
};
