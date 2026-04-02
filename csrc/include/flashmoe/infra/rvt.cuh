/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by osayamen on 1/12/26.
//

#ifndef FLASHMOE_RVT_CUH
#define FLASHMOE_RVT_CUH

#include "flashmoe/platform/platform.h"
#include "flashmoe/platform/device.h"
#include "flashmoe/platform/math_compat.h"

#if defined(FLASHMOE_PLATFORM_HIP)
#include <hip/hip_runtime.h>
// On HIP, cuda::std::bit_cast is provided by math_compat.h
#else
#include <cuda/std/bit>
#endif

namespace flashmoe {
  template<int Arch>
  constexpr int RedArch = Arch < 800 ? 700 : (Arch < 900 ? 800 : 900);
  constexpr int RED_MAX_ALIGNMENT = 16;

  template<typename Element, int Alignment>
    requires(Alignment > 0 && Alignment <= RED_MAX_ALIGNMENT && cutlass::is_pow2<Alignment>::value)
  struct RedAddType {
    using Type = Element;
    using Width = cute::Int<1>;
  };

  template<int Alignment>
  struct RedAddType<__half, Alignment> {
    // Alignment > sizeof(__half) means that Alignment is 2, 4, 8 or 16
    // This means we can safely promote to __half2
    using Type = cuda::std::conditional_t<(Alignment > sizeof(__half)), __half2, __half>;
    using Width = cute::Int<sizeof(Type) / sizeof(__half)>;
  };

  template<int Alignment>
  struct RedAddType<__nv_bfloat16, Alignment> {
    using Type = cuda::std::conditional_t<(Alignment > sizeof(__nv_bfloat16)), __nv_bfloat162, __nv_bfloat16>;
    using Width = cute::Int<sizeof(Type) / sizeof(__nv_bfloat16)>;
  };

  template<int Arch, typename Element, int VectorWidth>
  struct RedAdd {
    static_assert(VectorWidth >= 1 && VectorWidth <= (RED_MAX_ALIGNMENT / sizeof(Element)) &&
                  cutlass::is_pow2<VectorWidth>::value);
#if defined(FLASHMOE_PLATFORM_HIP)
    // ROCm: map all arch values to the generic atomicAdd path
    // AMD CDNA does not have PTX-like global reduction instructions;
    // we use atomicAdd which is well-optimized on MI300X.
    using VectorWidth_ = cute::Int<1>;
#else
    static_assert(Arch == 700 || Arch == 800 || Arch == 900);
#endif
    static_assert(cuda::std::is_same_v<Element, double> || cuda::std::is_same_v<Element, float> ||
                  cuda::std::is_same_v<Element, __half> || cuda::std::is_same_v<Element, __half2> ||
                  cuda::std::is_same_v<Element, __nv_bfloat16> || cuda::std::is_same_v<Element, __nv_bfloat162>);
#if !defined(FLASHMOE_PLATFORM_HIP)
    static_assert(!cuda::std::is_same_v<Element, __nv_bfloat16> || Arch >= 800, "bfloat16 requires at least sm_80");
#endif
  };

// -------------------------------------------------------
// HIP: Generic atomicAdd-based reduction for all types
// -------------------------------------------------------
#if defined(FLASHMOE_PLATFORM_HIP)

  // Generic RedAdd for HIP -- uses atomicAdd which supports fp16/bf16/fp32/fp64 on MI300X
  // VectorWidth is always 1 on HIP (no vectorized global reductions)
  template<int Arch, int MaxVectorWidth>
  struct RedAdd<Arch, double, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;
    template<typename T>
    __device__ __forceinline__
    void operator()(double *__restrict__ const&addr, const T &v) const {
      atomicAdd(addr, v[0]);
    }
  };

  template<int Arch, int MaxVectorWidth>
  struct RedAdd<Arch, float, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;
    template<typename T>
    __device__ __forceinline__
    void operator()(float *__restrict__ const&addr, const T &v) const {
      atomicAdd(addr, v[0]);
    }
  };

  // CAS-based atomicAdd for __half on HIP (ROCm 7.0 lacks native half atomicAdd)
  __device__ __forceinline__
  void hip_atomicAdd_half(__half* addr, __half val) {
      // Pack __half into the appropriate 16-bit slot of a 32-bit word
      unsigned int* base = reinterpret_cast<unsigned int*>(
          reinterpret_cast<uintptr_t>(addr) & ~uintptr_t(3));
      unsigned int shift = (reinterpret_cast<uintptr_t>(addr) & 2) ? 16 : 0;
      unsigned int old_val = *base;
      unsigned int assumed;
      do {
          assumed = old_val;
          unsigned short old_h = static_cast<unsigned short>((assumed >> shift) & 0xFFFF);
          __half old_half = *reinterpret_cast<__half*>(&old_h);
          __half new_half = __hadd(old_half, val);
          unsigned short new_h = *reinterpret_cast<unsigned short*>(&new_half);
          unsigned int new_val = (assumed & ~(0xFFFFu << shift)) |
                                 (static_cast<unsigned int>(new_h) << shift);
          old_val = atomicCAS(base, assumed, new_val);
      } while (old_val != assumed);
  }

  template<int Arch, int MaxVectorWidth>
  struct RedAdd<Arch, __half, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;
    template<typename T>
    __device__ __forceinline__
    void operator()(__half *__restrict__ const&addr, const T &v) const {
      hip_atomicAdd_half(addr, v[0]);
    }
  };

  template<int Arch, int MaxVectorWidth>
  struct RedAdd<Arch, __half2, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;
    template<typename T>
    __device__ __forceinline__
    void operator()(__half2 *__restrict__ const&addr, const T &v) const {
      __half* hp = reinterpret_cast<__half*>(addr);
      __half2 val = v[0];
      hip_atomicAdd_half(hp,     __low2half(val));
      hip_atomicAdd_half(hp + 1, __high2half(val));
    }
  };

  template<int Arch, int MaxVectorWidth>
  struct RedAdd<Arch, __nv_bfloat16, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;
    template<typename T>
    __device__ __forceinline__
    void operator()(__nv_bfloat16 *__restrict__ const&addr, const T &v) const {
      // HIP doesn't have native atomicAdd for bf16; use CAS-based approach via float
      float* fp = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(addr) & ~uintptr_t(3));
      unsigned int offset = (reinterpret_cast<uintptr_t>(addr) & 3) ? 16 : 0;
      float old_val = *fp;
      float new_val;
      unsigned int* ui = reinterpret_cast<unsigned int*>(fp);
      unsigned int old_bits, new_bits;
      do {
        old_bits = __float_as_uint(old_val);
        // Extract the bf16, add, store back
        unsigned short old_bf16 = (old_bits >> offset) & 0xFFFF;
        hip_bfloat16 old_hb; memcpy(&old_hb, &old_bf16, 2);
        float sum = static_cast<float>(old_hb) + static_cast<float>(v[0]);
        hip_bfloat16 new_hb(sum);
        unsigned short new_bf16; memcpy(&new_bf16, &new_hb, 2);
        new_bits = (old_bits & ~(0xFFFFu << offset)) | (static_cast<unsigned int>(new_bf16) << offset);
        new_val = __uint_as_float(new_bits);
        old_val = __uint_as_float(atomicCAS(ui, old_bits, new_bits));
      } while (__float_as_uint(old_val) != old_bits);
    }
  };

  template<int Arch, int MaxVectorWidth>
  struct RedAdd<Arch, __nv_bfloat162, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;
    template<typename T>
    __device__ __forceinline__
    void operator()(__nv_bfloat162 *__restrict__ const&addr, const T &v) const {
      // Decompose bf162 atomicAdd into two scalar bf16 additions
      __nv_bfloat16* bp = reinterpret_cast<__nv_bfloat16*>(addr);
      __nv_bfloat162 val = v[0];
      // Extract low and high halves
      __nv_bfloat16 lo, hi;
      memcpy(&lo, reinterpret_cast<const char*>(&val), 2);
      memcpy(&hi, reinterpret_cast<const char*>(&val) + 2, 2);
      RedAdd<Arch, __nv_bfloat16, 1> scalar_op;
      cutlass::AlignedArray<__nv_bfloat16, 1, sizeof(__nv_bfloat16)> lo_arr, hi_arr;
      lo_arr[0] = lo;
      hi_arr[0] = hi;
      scalar_op(bp, lo_arr);
      scalar_op(bp + 1, hi_arr);
    }
  };

#else // CUDA -- original PTX inline asm implementations

  template<int MaxVectorWidth>
  struct RedAdd<700, double, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, double>)
    __device__ __forceinline__
    void operator()(double *__restrict__ const&addr, const T &v) const {
      asm volatile("red.global.add.f64 [%0], %1;"
        :
        : "l"(addr), "d"(v[0])
        : "memory");
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<700, float, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, float>)
    __device__ __forceinline__
    void operator()(float *__restrict__ const&addr, const T &v) const {
      asm volatile("red.global.add.f32 [%0], %1;"
        :
        : "l"(addr), "f"(v[0])
        : "memory");
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<700, __half, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __half>)
    __device__ __forceinline__
    void operator()(__half *__restrict__ const&addr, const T &v) const {
      asm volatile("red.global.add.noftz.f16 [%0], %1;"
        :
        : "l"(addr), "h"(v[0])
        : "memory");
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<700, __half2, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __half2>)
    __device__ __forceinline__
    void operator()(__half2 *__restrict__ const&addr, const T &v) const {
      auto v0 = cuda::std::bit_cast<uint32_t>(static_cast<__half2_raw>(v[0]));
      asm volatile("red.global.add.noftz.f16x2 [%0], %1;"
        :
        : "l"(addr), "r"(v0)
        : "memory");
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<800, double, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, double>)
    __device__ __forceinline__
    void operator()(double *__restrict__ const&addr, const T &v) const {
      asm volatile("red.global.add.f64 [%0], %1;"
        :
        : "l"(addr), "d"(v[0])
        : "memory");
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<800, float, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, float>)
    __device__ __forceinline__
    void operator()(float *__restrict__ const&addr, const T &v) const {
      asm volatile("red.global.add.f32 [%0], %1;"
        :
        : "l"(addr), "f"(v[0])
        : "memory");
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<800, __half, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __half>)
    __device__ __forceinline__
    void operator()(__half *__restrict__ const&addr, const T &v) const {
      asm volatile("red.global.add.noftz.f16 [%0], %1;"
        :
        : "l"(addr), "h"(v[0])
        : "memory");
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<800, __half2, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __half2>)
    __device__ __forceinline__
    void operator()(__half2 *__restrict__ const&addr, const T &v) const {
      auto v0 = cuda::std::bit_cast<uint32_t>(static_cast<__half2_raw>(v[0]));
      asm volatile("red.global.add.noftz.f16x2 [%0], %1;"
        :
        : "l"(addr), "r"(v0)
        : "memory");
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<800, __nv_bfloat16, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __nv_bfloat16>)
    __device__ __forceinline__
    void operator()(__nv_bfloat16 *__restrict__ const&addr, const T &v) const {
      atomicAdd(addr, v[0]);
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<800, __nv_bfloat162, MaxVectorWidth> {
    // TODO extend to bf162x2
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __nv_bfloat162>)
    __device__ __forceinline__
    void operator()(__nv_bfloat162 *__restrict__ const&addr, const T &v) const {
      atomicAdd(addr, v[0]);
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<900, double, MaxVectorWidth> {
    using VectorWidth = cute::Int<1>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, double>)
    __device__ __forceinline__
    void operator()(double *__restrict__ const&addr, const T &v) const {
      asm volatile("red.global.add.f64 [%0], %1;"
        :
        : "l"(addr), "d"(v[0])
        : "memory");
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<900, float, MaxVectorWidth> {
    using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 4)>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, float>
        && (VectorWidth::value == 1 || VectorWidth::value == 2 || VectorWidth::value == 4))
    __device__ __forceinline__
    void operator()(float *__restrict__ const&addr, const T &v) const {
      if constexpr (VectorWidth::value == 1) {
        asm volatile("red.global.add.f32 [%0], %1;"
          :
          : "l"(addr), "f"(v[0])
          : "memory");
      } else if (VectorWidth::value == 2) {
        asm volatile("red.global.v2.f32.add [%0], {%1, %2};"
          :
          : "l"(addr), "f"(v[0]), "f"(v[1])
          : "memory");
      } else if (VectorWidth::value == 4) {
        asm volatile("red.global.v4.f32.add [%0], {%1, %2, %3, %4};"
          :
          : "l"(addr), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3])
          : "memory");
      }
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<900, __half, MaxVectorWidth> {
    using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 8)>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __half>
        && (VectorWidth::value == 1 || VectorWidth::value == 2 || VectorWidth::value == 4 || VectorWidth::value == 8))
    __device__ __forceinline__
    void operator()(__half *__restrict__ const&addr, const T &v) const {
      if constexpr (VectorWidth::value == 1) {
        asm volatile("red.global.add.noftz.f16 [%0], %1;"
          :
          : "l"(addr), "h"(v[0])
          : "memory");
      } else if constexpr (VectorWidth::value == 2) {
        asm volatile("red.global.v2.f16.add.noftz [%0], {%1, %2};"
          :
          : "l"(addr), "h"(v[0]), "h"(v[1])
          : "memory");
      } else if constexpr (VectorWidth::value == 4) {
        asm volatile("red.global.v4.f16.add.noftz [%0], {%1, %2, %3, %4};"
          :
          : "l"(addr), "h"(v[0]), "h"(v[1]), "h"(v[2]), "h"(v[3])
          : "memory");
      } else if constexpr (VectorWidth::value == 8) {
        asm volatile("red.global.v8.f16.add.noftz [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
          :
          : "l"(addr),
          "h"(v[0]), "h"(v[1]), "h"(v[2]), "h"(v[3]),
          "h"(v[4]), "h"(v[5]), "h"(v[6]), "h"(v[7])
          : "memory");
      }
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<900, __half2, MaxVectorWidth> {
    using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 4)>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __half2>
        && (VectorWidth::value == 1 || VectorWidth::value == 2 || VectorWidth::value == 4))
    __device__ __forceinline__
    void operator()(__half2 *__restrict__ const&addr, const T &v) const {
      if constexpr (VectorWidth::value == 1) {
        auto v0 = cuda::std::bit_cast<uint32_t>(static_cast<__half2_raw>(v[0]));
        asm volatile("red.global.add.noftz.f16x2 [%0], %1;"
          :
          : "l"(addr), "r"(v0)
          : "memory");
      } else if constexpr (VectorWidth::value == 2) {
        auto v0 = cuda::std::bit_cast<uint32_t>(static_cast<__half2_raw>(v[0]));
        auto v1 = cuda::std::bit_cast<uint32_t>(static_cast<__half2_raw>(v[1]));
        asm volatile("red.global.v2.f16x2.add.noftz [%0], {%1, %2};"
          :
          : "l"(addr), "r"(v0), "r"(v1)
          : "memory");
      } else if constexpr (VectorWidth::value == 4) {
        auto v0 = cuda::std::bit_cast<uint32_t>(static_cast<__half2_raw>(v[0]));
        auto v1 = cuda::std::bit_cast<uint32_t>(static_cast<__half2_raw>(v[1]));
        auto v2 = cuda::std::bit_cast<uint32_t>(static_cast<__half2_raw>(v[2]));
        auto v3 = cuda::std::bit_cast<uint32_t>(static_cast<__half2_raw>(v[3]));
        asm volatile("red.global.v4.f16x2.add.noftz [%0], {%1, %2, %3, %4};"
          :
          : "l"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3)
          : "memory");
      }
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<900, __nv_bfloat16, MaxVectorWidth> {
    using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 8)>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __nv_bfloat16> &&
        (VectorWidth::value == 1 || VectorWidth::value == 2 || VectorWidth::value == 4 || VectorWidth::value == 8))
    __device__ __forceinline__
    void operator()(__nv_bfloat16 *__restrict__ const&addr, const T &v) const {
      if constexpr (VectorWidth::value == 1) {
        asm volatile("red.global.add.noftz.bf16 [%0], %1;"
          :
          : "l"(addr), "h"(v[0])
          : "memory");
      } else if constexpr (VectorWidth::value == 2) {
        asm volatile("red.global.v2.bf16.add.noftz [%0], {%1, %2};"
          :
          : "l"(addr), "h"(v[0]), "h"(v[1])
          : "memory");
      } else if constexpr (VectorWidth::value == 4) {
        asm volatile("red.global.v4.bf16.add.noftz [%0], {%1, %2, %3, %4};"
          :
          : "l"(addr), "h"(v[0]), "h"(v[1]), "h"(v[2]), "h"(v[3])
          : "memory");
      } else if constexpr (VectorWidth::value == 8) {
        asm volatile("red.global.v8.bf16.add.noftz [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
          :
          : "l"(addr),
          "h"(v[0]), "h"(v[1]), "h"(v[2]), "h"(v[3]),
          "h"(v[4]), "h"(v[5]), "h"(v[6]), "h"(v[7])
          : "memory");
      }
    }
  };

  template<int MaxVectorWidth>
  struct RedAdd<900, __nv_bfloat162, MaxVectorWidth> {
    using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 4)>;

    template<typename T>
      requires(cuda::std::is_same_v<typename T::value_type, __nv_bfloat162> &&
        (VectorWidth::value == 1 || VectorWidth::value == 2 || VectorWidth::value == 4))
    __device__ __forceinline__
    void operator()(__nv_bfloat162 *__restrict__ const&addr, const T &v) const {
      if constexpr (VectorWidth::value == 1) {
        auto v0 = cuda::std::bit_cast<uint32_t>(static_cast<__nv_bfloat162_raw>(v[0]));
        asm volatile("red.global.add.noftz.bf16x2 [%0], %1;"
          :
          : "l"(addr), "r"(v0)
          : "memory");
      } else if constexpr (VectorWidth::value == 2) {
        auto v0 = cuda::std::bit_cast<uint32_t>(static_cast<__nv_bfloat162_raw>(v[0]));
        auto v1 = cuda::std::bit_cast<uint32_t>(static_cast<__nv_bfloat162_raw>(v[1]));
        asm volatile("red.global.v2.bf16x2.add.noftz [%0], {%1, %2};"
          :
          : "l"(addr), "r"(v0), "r"(v1)
          : "memory");
      } else if constexpr (VectorWidth::value == 4) {
        auto v0 = cuda::std::bit_cast<uint32_t>(static_cast<__nv_bfloat162_raw>(v[0]));
        auto v1 = cuda::std::bit_cast<uint32_t>(static_cast<__nv_bfloat162_raw>(v[1]));
        auto v2 = cuda::std::bit_cast<uint32_t>(static_cast<__nv_bfloat162_raw>(v[2]));
        auto v3 = cuda::std::bit_cast<uint32_t>(static_cast<__nv_bfloat162_raw>(v[3]));
        asm volatile("red.global.v4.bf16x2.add.noftz [%0], {%1, %2, %3, %4};"
          :
          : "l"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3)
          : "memory");
      }
    }
  };

#endif // FLASHMOE_PLATFORM_HIP

}
#endif //FLASHMOE_RVT_CUH
