/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_PLATFORM_INTRINSICS_H
#define FLASHMOE_PLATFORM_INTRINSICS_H

#include "platform.h"

// -------------------------------------------------------
// Warp / Wavefront shuffle primitives
// -------------------------------------------------------
// CUDA requires a mask argument; HIP does not (whole-wavefront only).
// We provide wrapper macros/functions that accept CUDA-style signatures
// and strip the mask on HIP.

#if defined(FLASHMOE_PLATFORM_HIP)

// Shuffle — map CUDA's __shfl_sync to HIP's __shfl
// CUDA supports both 3-arg (mask, val, src) and 4-arg (mask, val, src, width) forms.
// We use variadic macros to handle both.
#define __shfl_sync(mask, val, srcLane, ...)      __shfl((val), (srcLane) __VA_OPT__(,) __VA_ARGS__)
#define __shfl_up_sync(mask, val, delta, ...)     __shfl_up((val), (delta) __VA_OPT__(,) __VA_ARGS__)
#define __shfl_down_sync(mask, val, delta, ...)   __shfl_down((val), (delta) __VA_OPT__(,) __VA_ARGS__)
#define __shfl_xor_sync(mask, val, laneMask, ...) __shfl_xor((val), (laneMask) __VA_OPT__(,) __VA_ARGS__)

// -------------------------------------------------------
// Ballot
// -------------------------------------------------------
// CUDA: uint32_t __ballot_sync(uint32_t mask, int predicate)
// HIP:  uint64_t __ballot(int predicate)  — returns 64-bit for wavefront-64
//
// We provide a wrapper that accepts the CUDA signature and ignores the mask.
#define __ballot_sync(mask, predicate)  __ballot((predicate))

// -------------------------------------------------------
// __syncwarp / wavefront barrier
// -------------------------------------------------------
// On CUDA, __syncwarp(mask) synchronizes threads within a warp.
// HIP wavefronts execute in lock-step (SIMD); there is no sub-wavefront
// divergence, so __syncwarp is a no-op / compiler fence.
#if !defined(__syncwarp)
#  define __syncwarp(...)  __builtin_amdgcn_wave_barrier()
#endif

// -------------------------------------------------------
// __activemask
// -------------------------------------------------------
// Returns a mask of all active lanes. On HIP wavefront-64, returns uint64_t.
#if !defined(__activemask)
#  define __activemask()  __ballot(1)
#endif

// -------------------------------------------------------
// __popc — population count
// -------------------------------------------------------
// CUDA __popc works on 32-bit. HIP __popcll works on 64-bit.
// Provide a lane-mask-width-aware version.
__device__ __forceinline__
int flashmoe_popc(flashmoe_lane_mask_t mask) {
    return __popcll(static_cast<unsigned long long>(mask));
}

// -------------------------------------------------------
// __ffs — find first set bit (1-indexed)
// -------------------------------------------------------
__device__ __forceinline__
int flashmoe_ffs(flashmoe_lane_mask_t mask) {
    return __ffsll(static_cast<long long>(mask));
}

#else // CUDA

__device__ __forceinline__
int flashmoe_popc(flashmoe_lane_mask_t mask) {
    return __popc(mask);
}

__device__ __forceinline__
int flashmoe_ffs(flashmoe_lane_mask_t mask) {
    return __ffs(static_cast<int>(mask));
}

#endif // FLASHMOE_PLATFORM_HIP

// -------------------------------------------------------
// Thread indexing helpers (identical on both platforms)
// -------------------------------------------------------
// threadIdx, blockIdx, blockDim, gridDim are the same in CUDA and HIP.

#endif // FLASHMOE_PLATFORM_INTRINSICS_H
