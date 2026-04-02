/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_PLATFORM_DEVICE_H
#define FLASHMOE_PLATFORM_DEVICE_H

#include "platform.h"

// -------------------------------------------------------
// Device function qualifiers
// -------------------------------------------------------
// __device__, __global__, __host__, __shared__, __constant__ are
// identical in both CUDA and HIP — no mapping required.
//
// __forceinline__ is also supported by both compilers (nvcc and hipcc).
//
// This header exists to document that fact and provide a central
// location for any future divergences.

// -------------------------------------------------------
// __launch_bounds__ — identical syntax on both platforms
// -------------------------------------------------------
// Usage: __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
// No mapping needed.

// -------------------------------------------------------
// __grid_constant__ — CUDA 12+ feature
// -------------------------------------------------------
#if defined(FLASHMOE_PLATFORM_HIP)
// HIP does not support __grid_constant__; make it a no-op.
#  if !defined(__grid_constant__)
#    define __grid_constant__
#  endif
#else
// CUDA: __grid_constant__ is available in CUDA 12+.
// For older CUDA, define as no-op.
#  if !defined(__grid_constant__) && (!defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 12)
#    define __grid_constant__
#  endif
#endif

// -------------------------------------------------------
// Half / BFloat16 type headers
// -------------------------------------------------------
#if defined(FLASHMOE_PLATFORM_HIP)
#  include <hip/hip_fp16.h>
#  include <hip/hip_bfloat16.h>
#  include <hip/amd_detail/amd_hip_bf16.h>
// HIP uses __half and hip_bfloat16; alias for compatibility
using __nv_bfloat16 = hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#else
#  include <cuda_fp16.h>
#  include <cuda_bf16.h>
#endif

#endif // FLASHMOE_PLATFORM_DEVICE_H
