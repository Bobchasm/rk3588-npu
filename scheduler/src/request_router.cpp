#include "request_router.h"

RequestRouter::RequestRouter() = default;

bool RequestRouter::should_route(const PrefillRequest& req) const {
    return true;
}

bool RequestRouter::should_route_stage(const StageRunRequest& req) const {
    return true;
}
