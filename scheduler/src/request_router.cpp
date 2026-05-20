#include "request_router.h"

RequestRouter::RequestRouter() = default;

bool RequestRouter::should_route_generate(const GenerateTokensRequest& req) const {
    (void)req;
    return true;
}

bool RequestRouter::should_route_stage(const StageForwardRequest& req) const {
    (void)req;
    return true;
}
