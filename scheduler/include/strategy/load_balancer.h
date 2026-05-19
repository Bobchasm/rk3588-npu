#pragma once

#include "registry/node_info.h"
#include <vector>

class LoadBalancer {
public:
    LoadBalancer();
    WorkerId select_node(const std::vector<WorkerNodeInfo>& nodes);
};
