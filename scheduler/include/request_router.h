#pragma once

#include "scheduler_types.h"
#include <vector>

class RequestRouter {
public:
    RequestRouter();
    bool should_route(const PrefillRequest& req) const;
    bool should_route_stage(const StageRunRequest& req) const;
};
