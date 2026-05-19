#pragma once

#include "registry/node_info.h"

class SchedulePolicy {
public:
    SchedulePolicy() = default;
    bool allow_request(const WorkerNodeInfo& node_info) const;
};
