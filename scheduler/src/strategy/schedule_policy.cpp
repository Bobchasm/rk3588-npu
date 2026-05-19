#include "strategy/schedule_policy.h"

bool SchedulePolicy::allow_request(const WorkerNodeInfo& node_info) const {
    return node_info.max_pending_requests > 0;
}
