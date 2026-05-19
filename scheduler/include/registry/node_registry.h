#pragma once

#include "registry/node_info.h"
#include <map>
#include <mutex>
#include <string>
#include <vector>

class NodeRegistry {
public:
    NodeRegistry();
    bool register_node(const WorkerNodeInfo& info);
    bool deregister_node(const WorkerId& id);
    std::vector<WorkerNodeInfo> list_nodes() const;

private:
    mutable std::mutex mutex_;
    std::map<WorkerId, WorkerNodeInfo> nodes_;
};
