/*
 * Copyright (c) 2025, FlashMoE ROCm Porting Project
 * All rights reserved.
 * BSD 3-Clause License — see LICENSE in the project root.
 *
 * rocBLASDx — Device-side GEMM library for AMD MI300X (gfx942)
 *
 * Header-only library providing cuBLASDx-compatible device-side GEMM
 * for AMD GPUs using MFMA (Matrix Fused Multiply-Add) instructions.
 *
 * Target: MI300X (gfx942), CDNA3 architecture
 * Wavefront size: 64
 *
 * Usage:
 *   #include <rocblasdx/rocblasdx.hpp>
 *
 *   // Build a BLAS descriptor (same pattern as cuBLASDx):
 *   using BLAS = decltype(
 *       rocblasdx::Size<64, 64, 32>() +
 *       rocblasdx::Precision<__half, __half, float>() +
 *       rocblasdx::Type<rocblasdx::type::real>() +
 *       rocblasdx::Function<rocblasdx::function::MM>() +
 *       rocblasdx::Arrangement<rocblasdx::row_major, rocblasdx::col_major, rocblasdx::row_major>() +
 *       rocblasdx::Block() +
 *       rocblasdx::Alignment<16, 16, 16>() +
 *       rocblasdx::BlockDim<256>() +
 *       rocblasdx::StaticBlockDim() +
 *       rocblasdx::EnableInputStreaming() +
 *       rocblasdx::SM<942, rocblasdx::sm_modifier::arch_specific>());
 *
 *   // Use the BLAS type:
 *   auto accumulator = BLAS::suggest_accumulator();
 *   BLAS().execute(sA, sB, accumulator, transformA, transformB);
 *   rocblasdx::copy<BLAS, alignment>(src, dst);
 *   rocblasdx::copy_fragment<alignment>(frag, smem, accumulator);
 *
 * Supported element types: __half, hip_bfloat16, float, double
 * Supported arrangements: row_major, col_major
 */

#ifndef ROCBLASDX_ROCBLASDX_HPP
#define ROCBLASDX_ROCBLASDX_HPP

#include "types.hpp"
#include "layout.hpp"
#include "fragment.hpp"
#include "mfma.hpp"
#include "gemm.hpp"

#endif // ROCBLASDX_ROCBLASDX_HPP
