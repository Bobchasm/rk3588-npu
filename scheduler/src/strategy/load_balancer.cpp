#include "strategy/load_balancer.h"

LoadBalancer::LoadBalancer() = default;

WorkerId LoadBalancer::select_node(const std::vector<WorkerNodeInfo>& nodes) {
    if (nodes.empty()) {
        return {};
    }
    return nodes.front().worker_id;
}
