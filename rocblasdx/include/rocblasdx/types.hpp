/*
 * Copyright (c) 2025, FlashMoE ROCm Porting Project
 * All rights reserved.
 * BSD 3-Clause License — see LICENSE in the project root.
 *
 * rocBLASDx — Device-side GEMM library for AMD MI300X (gfx942)
 * types.hpp — Element types, accumulator type selection, tfloat32_t stub
 */

#ifndef ROCBLASDX_TYPES_HPP
#define ROCBLASDX_TYPES_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <type_traits>

namespace rocblasdx {

// -------------------------------------------------------
// tfloat32_t — TF32 stub for ROCm
// -------------------------------------------------------
// MI300X does not have native TF32 support like NVIDIA Hopper.
// We provide a storage type that round-trips through float.
// On ROCm, TF32 compute falls back to full FP32 MFMA.
struct alignas(4) tfloat32_t {
    float storage;

    __host__ __device__ tfloat32_t() = default;
    __host__ __device__ tfloat32_t(float v) : storage(v) {}
    __host__ __device__ operator float() const { return storage; }

    __host__ __device__ tfloat32_t operator+(const tfloat32_t& rhs) const {
        return tfloat32_t{storage + rhs.storage};
    }
    __host__ __device__ tfloat32_t operator*(const tfloat32_t& rhs) const {
        return tfloat32_t{storage * rhs.storage};
    }
    __host__ __device__ tfloat32_t& operator+=(const tfloat32_t& rhs) {
        storage += rhs.storage;
        return *this;
    }
    __host__ __device__ bool operator==(const tfloat32_t& rhs) const {
        return storage == rhs.storage;
    }
};

// -------------------------------------------------------
// Accumulator type selection
// -------------------------------------------------------
// For half/bfloat16 inputs, accumulate in float.
// For float inputs (including tfloat32_t on ROCm), accumulate in float.
// For double inputs, accumulate in double.
template <typename Element>
struct AccumulatorTypeTraits {
    using type = float;  // default: fp16, bf16, tf32, float all accumulate in float
};

template <>
struct AccumulatorTypeTraits<double> {
    using type = double;
};

template <typename Element>
using accumulator_type_t = typename AccumulatorTypeTraits<Element>::type;

// -------------------------------------------------------
// identity functor (used as no-op transform)
// -------------------------------------------------------
struct identity {
    template <typename T>
    __host__ __device__ __forceinline__
    constexpr T operator()(const T& x) const { return x; }
};

// -------------------------------------------------------
// BLAS descriptor building blocks (compile-time)
// -------------------------------------------------------
namespace type {
    enum value { real, complex };
}

namespace function {
    enum value { MM };
}

// Size descriptor — tile M, N, K
template <int M_, int N_, int K_>
struct Size {
    static constexpr int m = M_;
    static constexpr int n = N_;
    static constexpr int k = K_;
};

// Precision descriptor — types for A, B, C
template <typename A_, typename B_, typename C_>
struct Precision {
    using A = A_;
    using B = B_;
    using C = C_;
};

// Type descriptor
template <type::value V>
struct Type {
    static constexpr type::value value = V;
};

// Function descriptor
template <function::value V>
struct Function {
    static constexpr function::value value = V;
};

// Block descriptor (marker)
struct Block {};

// BlockDim descriptor
template <int Threads_>
struct BlockDim {
    static constexpr int threads = Threads_;
};

// StaticBlockDim (marker)
struct StaticBlockDim {};

// EnableInputStreaming (marker)
struct EnableInputStreaming {};

// SM descriptor — on ROCm this captures the gfx version
enum class sm_modifier { generic, arch_specific };
template <int Arch_, sm_modifier Mod_ = sm_modifier::generic>
struct SM {
    static constexpr int arch = Arch_;
    static constexpr sm_modifier modifier = Mod_;
};

// Alignment descriptor
template <int AlignA_, int AlignB_, int AlignC_>
struct Alignment {
    static constexpr int a = AlignA_;
    static constexpr int b = AlignB_;
    static constexpr int c = AlignC_;
};

} // namespace rocblasdx

#endif // ROCBLASDX_TYPES_HPP
