/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_PLATFORM_PROFILING_H
#define FLASHMOE_PLATFORM_PROFILING_H

#include "platform.h"

// -------------------------------------------------------
// Profiling / tracing — NVTX ↔ rocTX
// -------------------------------------------------------
// Both libraries have similar push/pop/range semantics.
// We provide a uniform interface that maps to the available backend.

#if defined(FLASHMOE_PLATFORM_HIP)

// ROCm tracing via rocTX (roctx.h)
#include <roctracer/roctx.h>
#include <string>

namespace flashmoe {

struct flashmoeDomain {
    static constexpr auto const* name{"FlashMoE"};
};

// Scoped range — constructor pushes, destructor pops
struct flashmoeRange {
    explicit flashmoeRange(const char* msg) {
        roctxRangePush(msg);
    }
    explicit flashmoeRange(const std::string& msg) {
        roctxRangePush(msg.c_str());
    }
    ~flashmoeRange() {
        roctxRangePop();
    }
    // Non-copyable, non-movable
    flashmoeRange(const flashmoeRange&) = delete;
    flashmoeRange& operator=(const flashmoeRange&) = delete;
};

} // namespace flashmoe

// Convenience macros
#define FLASHMOE_RANGE_PUSH(msg)  roctxRangePush(msg)
#define FLASHMOE_RANGE_POP()      roctxRangePop()
#define FLASHMOE_MARK(msg)        roctxMark(msg)

#else // CUDA — use NVTX3

#include <nvtx3/nvtx3.hpp>

namespace flashmoe {

struct flashmoeDomain {
    static constexpr auto const* name{"FlashMoE"};
};

using flashmoeRange = nvtx3::scoped_range_in<flashmoeDomain>;

} // namespace flashmoe

#define FLASHMOE_RANGE_PUSH(msg)  nvtxRangePushA(msg)
#define FLASHMOE_RANGE_POP()      nvtxRangePop()
#define FLASHMOE_MARK(msg)        nvtxMarkA(msg)

#endif // FLASHMOE_PLATFORM_HIP

#endif // FLASHMOE_PLATFORM_PROFILING_H
