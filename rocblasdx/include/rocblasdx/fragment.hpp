/*
 * Copyright (c) 2025, FlashMoE ROCm Porting Project
 * All rights reserved.
 * BSD 3-Clause License — see LICENSE in the project root.
 *
 * rocBLASDx — Device-side GEMM library for AMD MI300X (gfx942)
 * fragment.hpp — Register-resident tile fragments (accumulator, input tiles)
 */

#ifndef ROCBLASDX_FRAGMENT_HPP
#define ROCBLASDX_FRAGMENT_HPP

#include <hip/hip_runtime.h>
#include "types.hpp"

namespace rocblasdx {

// -------------------------------------------------------
// Fragment — a fixed-size register-resident array
// -------------------------------------------------------
// This is the register tile that holds partial GEMM results
// (accumulator) or loaded input data. Each thread in the
// warp/wavefront owns a portion of the tile.
//
// Mirrors cuBLASDx's accumulator and fragment types.
// The key property: operator()(int i) for element access,
// clear() to zero, get_results() to access the result tensor.

template <typename T, int N>
struct Fragment {
    using value_type = T;
    static constexpr int num_elements = N;

    T data[N];

    __device__ __forceinline__ void clear() {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            data[i] = T{0};
        }
    }

    __device__ __forceinline__ T& operator()(int i) { return data[i]; }
    __device__ __forceinline__ const T& operator()(int i) const { return data[i]; }

    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return data[i]; }

    // get_results() — returns a reference to this fragment.
    // In cuBLASDx, accumulator.get_results() returns a Tensor;
    // here we return the fragment itself which supports (i) indexing.
    __device__ __forceinline__ Fragment& get_results() { return *this; }
    __device__ __forceinline__ const Fragment& get_results() const { return *this; }
};

// -------------------------------------------------------
// Accumulator — alias for Fragment with accumulator semantics
// -------------------------------------------------------
// The accumulator is a Fragment with the appropriate type and
// size for the GEMM tile owned by one thread.
template <typename AccumT, int NumElements>
using Accumulator = Fragment<AccumT, NumElements>;

// -------------------------------------------------------
// make_fragment_like — create a fragment with different element type
// but same size as another fragment
// -------------------------------------------------------
template <typename NewT, typename OldFrag>
__device__ __forceinline__
auto make_fragment_like(const OldFrag& src) {
    Fragment<NewT, OldFrag::num_elements> frag;
    frag.clear();
    return frag;
}

// -------------------------------------------------------
// size() — get the number of elements in a fragment
// -------------------------------------------------------
template <typename T, int N>
__host__ __device__ __forceinline__
constexpr int size(const Fragment<T, N>&) {
    return N;
}

} // namespace rocblasdx

#endif // ROCBLASDX_FRAGMENT_HPP
