#pragma once

#include "scheduler_types.h"
#include <vector>

class RequestRouter {
public:
    RequestRouter();
    bool should_route_generate(const GenerateTokensRequest& req) const;
    bool should_route_stage(const StageForwardRequest& req) const;
};
