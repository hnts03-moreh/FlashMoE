/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by Osayamen on 1/5/26.
//

#ifndef FLASHMOE_VT_CUH
#define FLASHMOE_VT_CUH

#include "flashmoe/platform/platform.h"
#include "flashmoe/platform/math_compat.h"

#if defined(FLASHMOE_PLATFORM_HIP)
// cutlass::is_pow2 shim for HIP
namespace cutlass {
  template <int N>
  struct is_pow2 {
    static constexpr bool value = (N > 0) && ((N & (N - 1)) == 0);
  };
  // Also provide ispow2 as a function for bootstrap.cuh
  __host__ __device__ __forceinline__
  constexpr bool ispow2(size_t n) {
    return (n > 0) && ((n & (n - 1)) == 0);
  }
}
#endif

#include "constants.cuh"

namespace flashmoe {
    constexpr int MAX_ALIGNMENT = 16;
    template<typename T, int Alignment = MAX_ALIGNMENT>
    struct VectorTypeDescriptor {
        using VectorWidth = cute::Int<Alignment / sizeof(T)>;
        using VectorType = cutlass::AlignedArray<T, VectorWidth::value, Alignment>;
    };
    template<typename Element, int dim, int MAX_ALIGN = MAX_ALIGNMENT>
    requires(MAX_ALIGN <= MAX_ACCESS_ALIGNMENT && cutlass::is_pow2<MAX_ALIGN>::value && MAX_ALIGN >= 1)
    constexpr int ElementWidth = cute::min(dim, MAX_ALIGN / sizeof(Element));
    template<typename Element, int dim>
    constexpr uint32_t ElementAlignment = (cutlass::is_pow2<ElementWidth<Element, dim>>::value ?
        ElementWidth<Element, dim> : 1) * sizeof(Element);
    template<typename Element, int dim, int width>
    constexpr int ElementAlignmentForWidth = (cutlass::is_pow2<width>::value ? width : 1) * sizeof(Element);
}
#endif //FLASHMOE_VT_CUH
