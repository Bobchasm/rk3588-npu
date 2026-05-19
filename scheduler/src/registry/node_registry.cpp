#include "registry/node_registry.h"

NodeRegistry::NodeRegistry() = default;

bool NodeRegistry::register_node(const WorkerNodeInfo& info) {
    std::lock_guard<std::mutex> guard(mutex_);
    nodes_[info.worker_id] = info;
    return true;
}

bool NodeRegistry::deregister_node(const WorkerId& id) {
    std::lock_guard<std::mutex> guard(mutex_);
    return nodes_.erase(id) > 0;
}

std::vector<WorkerNodeInfo> NodeRegistry::list_nodes() const {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<WorkerNodeInfo> result;
    result.reserve(nodes_.size());
    for (const auto& kv : nodes_) {
        result.push_back(kv.second);
    }
    return result;
}
