//
// Created by Osayamen on 1/18/26.
//

#ifndef FLASHMOE_CONSTANTS_CUH
#define FLASHMOE_CONSTANTS_CUH

#include "flashmoe/platform/platform.h"

namespace flashmoe {
#if defined(FLASHMOE_PLATFORM_HIP)
  // MI300X: no 256-byte access; max vectorized load is 128 bytes
  constexpr int MAX_ACCESS_ALIGNMENT = 16;
#elif (__CUDA_ARCH__ >= 1000) && (defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 12) && (defined(__CUDACC_VER_MINOR__) && __CUDACC_VER_MINOR__ >= 9)
  constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
  constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif
  constexpr int WARP_SIZE = FLASHMOE_WARP_SIZE;
  // HIP: LDS has 64 banks of 4 bytes each = 256 bytes; CUDA: 32 banks of 4 bytes = 128 bytes
  constexpr unsigned long SMEM_BANKS_TOTAL_BYTE_WIDTH = 4 * FLASHMOE_WARP_SIZE;
}

namespace flashmoe::scheduler {
  constexpr int SCHEDULER_COUNT = WARP_SIZE; // warp/wavefront size
  // upper bounds for register state sizes
#if defined(FLASHMOE_PLATFORM_HIP)
  // ROCm: wavefront-64 means fewer wavefronts per CU, adjust state size
  #if defined(FLASHMOE_ARCH_MI300X) || defined(FLASHMOE_ARCH_MI300A)
  constexpr int PROCESSOR_STATE_SIZE = 5;
  #elif defined(FLASHMOE_ARCH_MI250)
  constexpr int PROCESSOR_STATE_SIZE = 6;
  #else
  constexpr int PROCESSOR_STATE_SIZE = 8;
  #endif
#elif defined(FLASHMOE_ARCH)
#if FLASHMOE_ARCH >= 1000
  constexpr int PROCESSOR_STATE_SIZE = 6;
#elif FLASHMOE_ARCH >= 900
  constexpr int PROCESSOR_STATE_SIZE = 5;
#else
  constexpr int PROCESSOR_STATE_SIZE = 8;
#endif
#else
  constexpr int PROCESSOR_STATE_SIZE = 8; // 8 is recommended
#endif
  static_assert(PROCESSOR_STATE_SIZE <= 16);
  constexpr int MAX_PROCESSORS = WARP_SIZE * PROCESSOR_STATE_SIZE; // can be relaxed but with slower perf
  constexpr int WORK_SET_SIZE = 4;
  constexpr int QUEUE_STATE_SIZE = 2;
}
#endif //FLASHMOE_CONSTANTS_CUH
