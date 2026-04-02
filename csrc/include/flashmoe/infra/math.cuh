//
// Created by osayamen on 1/18/26.
//

#ifndef FLASHMOE_MATH_CUH
#define FLASHMOE_MATH_CUH

#include "flashmoe/platform/platform.h"
#include "flashmoe/platform/math_compat.h"

#if !defined(FLASHMOE_PLATFORM_HIP)
#include <cuda/cmath>
#endif

#include "constants.cuh"

namespace flashmoe
{
  // Also applies to shared memory banks
  template <typename Element>
  __host__ __device__ __forceinline__
  constexpr auto rTCL(uint const& len) {
    return cuda::round_up(len * sizeof(Element), SMEM_BANKS_TOTAL_BYTE_WIDTH);
  }
}
#endif //FLASHMOE_MATH_CUH
