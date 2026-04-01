/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_PLATFORM_H
#define FLASHMOE_PLATFORM_H

// -------------------------------------------------------
// Platform detection
// -------------------------------------------------------
// Users may force a platform by defining FLASHMOE_PLATFORM_CUDA or
// FLASHMOE_PLATFORM_HIP before including this header. If neither is
// defined we auto-detect based on compiler macros.

#if !defined(FLASHMOE_PLATFORM_CUDA) && !defined(FLASHMOE_PLATFORM_HIP)
#  if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#    define FLASHMOE_PLATFORM_HIP 1
#  else
#    define FLASHMOE_PLATFORM_CUDA 1
#  endif
#endif

// -------------------------------------------------------
// Warp / wavefront sizing
// -------------------------------------------------------
#if defined(FLASHMOE_PLATFORM_HIP)
#  define FLASHMOE_WARP_SIZE 64
#else
#  define FLASHMOE_WARP_SIZE 32
#endif

// -------------------------------------------------------
// Lane-mask type — ballot returns uint32_t on CUDA, uint64_t on HIP
// -------------------------------------------------------
#include <cstdint>

#if defined(FLASHMOE_PLATFORM_HIP)
using flashmoe_lane_mask_t = uint64_t;
#  define FLASHMOE_LANE_MASK_TYPE uint64_t
#  define FLASHMOE_FULL_LANE_MASK (~uint64_t(0))
#else
using flashmoe_lane_mask_t = uint32_t;
#  define FLASHMOE_LANE_MASK_TYPE uint32_t
#  define FLASHMOE_FULL_LANE_MASK 0xFFFFFFFFu
#endif

// -------------------------------------------------------
// Architecture detection helpers
// -------------------------------------------------------
#if defined(FLASHMOE_PLATFORM_HIP)
// AMD CDNA architecture IDs
// gfx90a  = MI250
// gfx940/gfx941/gfx942 = MI300 family
#  if defined(__gfx942__)
#    define FLASHMOE_ARCH_MI300X 1
#  elif defined(__gfx940__) || defined(__gfx941__)
#    define FLASHMOE_ARCH_MI300A 1
#  elif defined(__gfx90a__)
#    define FLASHMOE_ARCH_MI250 1
#  endif
#  define FLASHMOE_ARCH_IS_CDNA 1
#else
// NVIDIA arch via __CUDA_ARCH__ (e.g. 700, 800, 900, 1000)
#  if defined(__CUDA_ARCH__)
#    if __CUDA_ARCH__ >= 900
#      define FLASHMOE_ARCH_IS_HOPPER 1
#    elif __CUDA_ARCH__ >= 800
#      define FLASHMOE_ARCH_IS_AMPERE 1
#    elif __CUDA_ARCH__ >= 700
#      define FLASHMOE_ARCH_IS_VOLTA 1
#    endif
#  endif
#endif

#endif // FLASHMOE_PLATFORM_H
