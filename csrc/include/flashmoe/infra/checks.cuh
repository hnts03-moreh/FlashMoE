//
// Created by osayamen on 1/10/26.
//

#ifndef FLASHMOE_CHECKS_CUH
#define FLASHMOE_CHECKS_CUH

#include "flashmoe/platform/platform.h"
#include "flashmoe/platform/math_compat.h"

#if defined(FLASHMOE_PLATFORM_HIP)
// On HIP, use the compatibility stubs from math_compat.h
// cutlass::AlignedArray, cutlass::Array, cute::is_rmem_v, cuda::std:: are all provided there.
#else
#include <cutlass/array.h>
#include <cute/tensor.hpp>
#endif

namespace flashmoe
{
    /// A more apropos name would be "static storage" rather than registers.
    template<class T>
    struct isRegister : cuda::std::false_type {};

    template<class T, int N, int Alignment>
    struct isRegister<cutlass::AlignedArray<T, N, Alignment>> : cuda::std::true_type {};

    template<class T, int N, bool RegisterSized>
    struct isRegister<cutlass::Array<T, N, RegisterSized>> : cuda::std::true_type {};

#if !defined(FLASHMOE_PLATFORM_HIP)
    // CuTe Tensor specialization -- only available on CUDA where CuTe is present
    template<class Engine, class Layout>
    struct isRegister<cute::Tensor<Engine, Layout>> :
    cuda::std::conditional_t<cute::is_rmem_v<cute::Tensor<Engine, Layout>>,
    cuda::std::true_type, cuda::std::false_type> {};
#endif

    template <class T>
    constexpr bool isRegisterV = isRegister<T>::value;
}
#endif //FLASHMOE_CHECKS_CUH
