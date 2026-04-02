//
// Created by osayamen on 1/15/26.
//

#ifndef FLASHMOE_ACTIVATION_CUH
#define FLASHMOE_ACTIVATION_CUH

#include "flashmoe/platform/platform.h"

#if defined(FLASHMOE_PLATFORM_HIP)
#include <rocblasdx/rocblasdx.hpp>
// ROCm: no CUTLASS epilogue; provide minimal activation functors
namespace flashmoe::hip_compat {
  template<typename Element>
  struct ReLU {
    __host__ __device__ __forceinline__
    Element operator()(const Element& x) const {
      return x > Element(0) ? x : Element(0);
    }
  };
  template<typename Element>
  struct GELU {
    __host__ __device__ __forceinline__
    Element operator()(const Element& x) const {
      // Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      constexpr float kAlpha = 0.7978845608f; // sqrt(2/pi)
      constexpr float kBeta = 0.044715f;
      float xf = static_cast<float>(x);
      float inner = kAlpha * (xf + kBeta * xf * xf * xf);
      return static_cast<Element>(xf * 0.5f * (1.0f + tanhf(inner)));
    }
  };
  template<typename Element>
  struct SiLu {
    __host__ __device__ __forceinline__
    Element operator()(const Element& x) const {
      float xf = static_cast<float>(x);
      return static_cast<Element>(xf / (1.0f + expf(-xf)));
    }
  };
} // namespace flashmoe::hip_compat
#else
#include <cublasdx.hpp>
#include <cutlass/epilogue/thread/activation.h>
#endif

namespace flashmoe {
  enum class Activation {
    identity = 0,
    silu = 1,
    gelu = 2,
    relu = 3 // keep numeric values aligned with defineAct()
  };

  template<int a>
  consteval Activation defineAct() {
    static_assert(a >= 0 && a < 4);
    switch (a) {
      case 0: return Activation::identity;
      case 1: return Activation::silu;
      case 2: return Activation::gelu;
      case 3: return Activation::relu;
      default: return Activation::identity;
    }
  }

  template<typename Element, Activation a = Activation::identity>
  struct ActivationType {
    static_assert(a == Activation::identity);
#if defined(FLASHMOE_PLATFORM_HIP)
    using AT = rocblasdx::identity;
#else
    using AT = cublasdx::identity;
#endif
  };

  template<typename Element>
  struct ActivationType<Element, Activation::relu> {
#if defined(FLASHMOE_PLATFORM_HIP)
    using AT = hip_compat::ReLU<Element>;
#else
    using AT = cutlass::epilogue::thread::ReLU<Element>;
#endif
  };

  template<typename Element>
  struct ActivationType<Element, Activation::gelu> {
#if defined(FLASHMOE_PLATFORM_HIP)
    using AT = hip_compat::GELU<Element>;
#else
    using AT = cutlass::epilogue::thread::GELU<Element>;
#endif
  };

  template<typename Element>
  struct ActivationType<Element, Activation::silu> {
#if defined(FLASHMOE_PLATFORM_HIP)
    using AT = hip_compat::SiLu<Element>;
#else
    using AT = cutlass::epilogue::thread::SiLu<Element>;
#endif
  };
}
#endif //FLASHMOE_ACTIVATION_CUH
