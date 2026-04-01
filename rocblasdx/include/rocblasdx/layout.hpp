/*
 * Copyright (c) 2025, FlashMoE ROCm Porting Project
 * All rights reserved.
 * BSD 3-Clause License — see LICENSE in the project root.
 *
 * rocBLASDx — Device-side GEMM library for AMD MI300X (gfx942)
 * layout.hpp — Memory layout descriptors and arrangement enums
 */

#ifndef ROCBLASDX_LAYOUT_HPP
#define ROCBLASDX_LAYOUT_HPP

#include <hip/hip_runtime.h>
#include <cstdint>

namespace rocblasdx {

// -------------------------------------------------------
// arrangement enum — matches cuBLASDx's arrangement
// -------------------------------------------------------
enum arrangement : int {
    row_major = 0,
    col_major = 1
};

// -------------------------------------------------------
// Arrangement descriptor (compile-time)
// -------------------------------------------------------
template <arrangement ArA_, arrangement ArB_, arrangement ArC_>
struct Arrangement {
    static constexpr arrangement ar_a = ArA_;
    static constexpr arrangement ar_b = ArB_;
    static constexpr arrangement ar_c = ArC_;
};

// -------------------------------------------------------
// Shared memory layout for tiles
// -------------------------------------------------------
// For MFMA on MI300X, we use simple row-major or column-major
// layouts in shared memory with appropriate padding to avoid
// bank conflicts (LDS has 32 banks, 4 bytes each on gfx942).

// SmemLayout: a simple 2D layout with compile-time extents and strides
template <int Rows_, int Cols_, int Stride_>
struct SmemLayout {
    static constexpr int rows   = Rows_;
    static constexpr int cols   = Cols_;
    static constexpr int stride = Stride_;  // leading dimension (elements between rows/cols)

    __host__ __device__ __forceinline__
    static constexpr int size() { return rows * stride; }

    // cosize: total number of elements needed to represent this layout
    __host__ __device__ __forceinline__
    static constexpr int cosize() { return size(); }
};

// -------------------------------------------------------
// LDS bank-conflict-free padding
// -------------------------------------------------------
// MI300X LDS: 32 banks, 4 bytes per bank.
// For half/bf16 (2 bytes): pad to avoid 2-element stride aliasing.
// For float (4 bytes): pad every 32 elements.
// A simple heuristic: add 8 elements of padding for float, 16 for half.

template <typename Element, int Cols>
struct PaddedStride {
    // For LDS bank-conflict avoidance.
    // 32 banks * 4 bytes = 128 bytes per bank cycle.
    // Pad by sizeof(Element) such that consecutive rows hit different banks.
    static constexpr int bank_width = 128 / sizeof(Element); // elements per bank cycle
    static constexpr int pad = (Cols % bank_width == 0) ? (128 / sizeof(Element) / 8) : 0;
    static constexpr int value = Cols + pad;
};

// -------------------------------------------------------
// suggest_layout helpers
// -------------------------------------------------------
// A-tile in smem: row-major for row_major arrangement, col-major for col_major
template <arrangement Ar, int Rows, int Cols, typename Element>
struct SuggestSmemLayout;

// Row-major: shape [Rows, Cols], stride = Cols + padding
template <int Rows, int Cols, typename Element>
struct SuggestSmemLayout<row_major, Rows, Cols, Element> {
    using type = SmemLayout<Rows, Cols, PaddedStride<Element, Cols>::value>;
};

// Col-major: shape [Rows, Cols], stride = Rows + padding (stored as [Cols, Rows] transposed)
template <int Rows, int Cols, typename Element>
struct SuggestSmemLayout<col_major, Rows, Cols, Element> {
    using type = SmemLayout<Rows, Cols, PaddedStride<Element, Rows>::value>;
};

} // namespace rocblasdx

#endif // ROCBLASDX_LAYOUT_HPP
