/*
 * Copyright (c) 2025, FlashMoE ROCm Porting Project
 * All rights reserved.
 * BSD 3-Clause License — see LICENSE in the project root.
 *
 * rocBLASDx — Device-side GEMM library for AMD MI300X (gfx942)
 * mfma.hpp — MFMA intrinsic wrappers for CDNA3
 *
 * This header wraps the __builtin_amdgcn_mfma_* intrinsics for gfx942
 * into type-dispatched callable structs.
 *
 * MFMA = Matrix Fused Multiply-Add.
 * On MI300X (CDNA3, gfx942), the key MFMA instructions used:
 *
 *   fp16:   mfma_f32_32x32x8f16   (32x32 output, K=8, fp16 inputs, fp32 accum)
 *           mfma_f32_16x16x16f16  (16x16 output, K=16, fp16 inputs, fp32 accum)
 *   bf16:   mfma_f32_32x32x8bf16_1k (32x32 output, K=8, bf16 inputs, fp32 accum)
 *           mfma_f32_16x16x16bf16_1k (16x16 output, K=16, bf16 inputs, fp32 accum)
 *   fp32:   mfma_f32_32x32x2f32   (32x32 output, K=2, fp32 inputs, fp32 accum)
 *           mfma_f32_16x16x4f32   (16x16 output, K=4, fp32 inputs, fp32 accum)
 *   fp64:   mfma_f64_16x16x4f64   (16x16 output, K=4, fp64 inputs, fp64 accum)
 *
 * Wavefront size: 64 threads.
 */

#ifndef ROCBLASDX_MFMA_HPP
#define ROCBLASDX_MFMA_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <type_traits>

#include "types.hpp"

namespace rocblasdx {
namespace mfma {

// -------------------------------------------------------
// MFMA instruction traits
// -------------------------------------------------------
// Each MFMA instruction has:
//   - MPerInstr, NPerInstr, KPerInstr: the M/N/K tile processed
//   - InputVecLen: number of input elements packed per lane
//   - AccumRegsPerThread: number of accumulator registers per thread
//   - CRegsPerThread: number of C-output registers per thread

// FP16 -> FP32 accum, 32x32xK=8
// NOTE: __half is a class type on AMD HIP and cannot be used with ext_vector_type.
// The fp16 MFMA intrinsics expect vectors of _Float16, so we use _Float16 vectors
// and reinterpret_cast when loading half data.
struct MfmaF32_32x32x8_F16 {
    using AType = __half;
    using BType = __half;
    using CType = float;

    static constexpr int M = 32;
    static constexpr int N = 32;
    static constexpr int K = 8;
    static constexpr int c_per_thread = 16;  // each thread holds 16 fp32 accum regs

    using AVec = _Float16 __attribute__((ext_vector_type(4)));
    using BVec = _Float16 __attribute__((ext_vector_type(4)));
    using CVec = float __attribute__((ext_vector_type(16)));

    __device__ __forceinline__
    static void run(CVec& c, const AVec& a, const BVec& b) {
        c = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
    }
};

// FP16 -> FP32 accum, 16x16xK=16
struct MfmaF32_16x16x16_F16 {
    using AType = __half;
    using BType = __half;
    using CType = float;

    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 16;
    static constexpr int c_per_thread = 4;

    using AVec = _Float16 __attribute__((ext_vector_type(4)));
    using BVec = _Float16 __attribute__((ext_vector_type(4)));
    using CVec = float __attribute__((ext_vector_type(4)));

    __device__ __forceinline__
    static void run(CVec& c, const AVec& a, const BVec& b) {
        c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    }
};

// BF16 -> FP32 accum, 32x32xK=8
// NOTE: hip_bfloat16 is a class type and cannot be used with ext_vector_type.
// The bf16_1k MFMA intrinsics expect vectors of short, so we use short vectors
// and reinterpret_cast when loading bf16 data.
struct MfmaF32_32x32x8_BF16 {
    using AType = hip_bfloat16;
    using BType = hip_bfloat16;
    using CType = float;

    static constexpr int M = 32;
    static constexpr int N = 32;
    static constexpr int K = 8;
    static constexpr int c_per_thread = 16;

    using AVec = short __attribute__((ext_vector_type(4)));
    using BVec = short __attribute__((ext_vector_type(4)));
    using CVec = float __attribute__((ext_vector_type(16)));

    __device__ __forceinline__
    static void run(CVec& c, const AVec& a, const BVec& b) {
        c = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c, 0, 0, 0);
    }
};

// BF16 -> FP32 accum, 16x16xK=16
struct MfmaF32_16x16x16_BF16 {
    using AType = hip_bfloat16;
    using BType = hip_bfloat16;
    using CType = float;

    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 16;
    static constexpr int c_per_thread = 4;

    using AVec = short __attribute__((ext_vector_type(4)));
    using BVec = short __attribute__((ext_vector_type(4)));
    using CVec = float __attribute__((ext_vector_type(4)));

    __device__ __forceinline__
    static void run(CVec& c, const AVec& a, const BVec& b) {
        c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
    }
};

// FP32 -> FP32 accum, 32x32xK=2
struct MfmaF32_32x32x2_F32 {
    using AType = float;
    using BType = float;
    using CType = float;

    static constexpr int M = 32;
    static constexpr int N = 32;
    static constexpr int K = 2;
    static constexpr int c_per_thread = 16;

    using AVec = float;  // scalar
    using BVec = float;  // scalar
    using CVec = float __attribute__((ext_vector_type(16)));

    __device__ __forceinline__
    static void run(CVec& c, const AVec& a, const BVec& b) {
        c = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, c, 0, 0, 0);
    }
};

// FP32 -> FP32 accum, 16x16xK=4
struct MfmaF32_16x16x4_F32 {
    using AType = float;
    using BType = float;
    using CType = float;

    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 4;
    static constexpr int c_per_thread = 4;

    using AVec = float;  // scalar
    using BVec = float;  // scalar
    using CVec = float __attribute__((ext_vector_type(4)));

    __device__ __forceinline__
    static void run(CVec& c, const AVec& a, const BVec& b) {
        c = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
    }
};

// FP64 -> FP64 accum, 16x16xK=4
struct MfmaF64_16x16x4_F64 {
    using AType = double;
    using BType = double;
    using CType = double;

    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 4;
    static constexpr int c_per_thread = 4;

    using AVec = double;
    using BVec = double;
    using CVec = double __attribute__((ext_vector_type(4)));

    __device__ __forceinline__
    static void run(CVec& c, const AVec& a, const BVec& b) {
        c = __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
    }
};

// -------------------------------------------------------
// SelectMfma — pick the best MFMA instruction for a given
// element type and tile shape preference.
// -------------------------------------------------------
// Strategy: prefer 32x32 instructions (higher throughput per wave),
// fall back to 16x16 when tile M or N < 32.

template <typename Element, int TileM, int TileN>
struct SelectMfma {
    // Default: fall back to fp32 16x16 for unknown types (used as safe default
    // when BlasBag is in intermediate state during composition)
    using type = MfmaF32_16x16x4_F32;
};

// FP16: prefer 32x32x8 when tile >= 32
template <int TileM, int TileN>
struct SelectMfma<__half, TileM, TileN> {
    using type = std::conditional_t<(TileM >= 32 && TileN >= 32),
                                    MfmaF32_32x32x8_F16,
                                    MfmaF32_16x16x16_F16>;
};

// BF16: prefer 32x32x8 when tile >= 32
template <int TileM, int TileN>
struct SelectMfma<hip_bfloat16, TileM, TileN> {
    using type = std::conditional_t<(TileM >= 32 && TileN >= 32),
                                    MfmaF32_32x32x8_BF16,
                                    MfmaF32_16x16x16_BF16>;
};

// FP32 (including tfloat32_t which falls back to fp32 MFMA on MI300X)
template <int TileM, int TileN>
struct SelectMfma<float, TileM, TileN> {
    using type = std::conditional_t<(TileM >= 32 && TileN >= 32),
                                    MfmaF32_32x32x2_F32,
                                    MfmaF32_16x16x4_F32>;
};

// tfloat32_t -> use float MFMA (rocblasdx::tfloat32_t defined in types.hpp)
template <int TileM, int TileN>
struct SelectMfma<rocblasdx::tfloat32_t, TileM, TileN> {
    using type = typename SelectMfma<float, TileM, TileN>::type;
};

// FP64
template <int TileM, int TileN>
struct SelectMfma<double, TileM, TileN> {
    using type = MfmaF64_16x16x4_F64;
};

template <typename Element, int TileM, int TileN>
using select_mfma_t = typename SelectMfma<Element, TileM, TileN>::type;

// -------------------------------------------------------
// MFMA output layout mapping
// -------------------------------------------------------
// For v_mfma_f32_MxNx*: maps (lane_id, vgpr_idx) to (row, col) within
// the MxN output sub-tile.
//
// Hardware layout (CDNA3, gfx942):
//   col = lane_id % mfma_n
//   num_groups = wavefront_size / mfma_n  (2 for 32x32, 4 for 16x16)
//   group_id = lane_id / mfma_n
//   block_stride = num_groups * 4
//   row = (vgpr / 4) * block_stride + group_id * 4 + (vgpr % 4)
//
template <typename MfmaInstr>
struct MfmaOutputLayout {
    static constexpr int mfma_m = MfmaInstr::M;
    static constexpr int mfma_n = MfmaInstr::N;
    static constexpr int c_per_thread = MfmaInstr::c_per_thread;
    static constexpr int wavefront_size = 64;
    static constexpr int num_groups = wavefront_size / mfma_n;
    static constexpr int block_stride = num_groups * 4;

    __device__ __forceinline__
    static int row(int lane_id, int vgpr) {
        int group_id = lane_id / mfma_n;
        return (vgpr / 4) * block_stride + group_id * 4 + (vgpr % 4);
    }

    __device__ __forceinline__
    static int col(int lane_id) {
        return lane_id % mfma_n;
    }
};

} // namespace mfma
} // namespace rocblasdx

#endif // ROCBLASDX_MFMA_HPP
