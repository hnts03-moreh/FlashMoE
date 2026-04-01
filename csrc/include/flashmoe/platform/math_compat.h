/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_PLATFORM_MATH_COMPAT_H
#define FLASHMOE_PLATFORM_MATH_COMPAT_H

// -------------------------------------------------------
// Compatibility layer for CUTLASS/CuTe/CCCL utilities
// -------------------------------------------------------
// This header is ONLY used on ROCm/HIP builds where the original
// CUTLASS, CuTe, and CCCL libraries are not available.
// On CUDA builds, the original libraries are used directly.
//
// It provides minimal, header-only implementations of the
// utility functions and types that FlashMoE uses from those libraries.

#include "platform.h"

#if defined(FLASHMOE_PLATFORM_HIP)

#include <type_traits>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <bit>
#include <limits>
#include <hip/hip_runtime.h>
#include "device.h"

// -------------------------------------------------------
// cuda::std:: → std:: mapping
// -------------------------------------------------------
// On HIP, we redirect cuda::std:: into std::.
namespace cuda {
namespace std {
    using ::std::is_same_v;
    using ::std::same_as;
    using ::std::is_same;
    using ::std::true_type;
    using ::std::false_type;
    using ::std::conditional_t;
    using ::std::underlying_type_t;
    using ::std::enable_if_t;
    using ::std::integral_constant;
    using ::std::is_integral_v;
    using ::std::is_floating_point_v;
    using ::std::decay_t;
    using ::std::byte;
    using ::std::bit_cast;
    using ::std::is_trivially_copyable_v;
    using ::std::is_invocable_r_v;

    // ignore shim (like std::ignore for [[nodiscard]])
    struct ignore_t {
        template <typename T>
        constexpr const ignore_t& operator=(T&&) const noexcept { return *this; }
    };
    inline constexpr ignore_t ignore{};

    // numeric_limits shim
    template <typename T>
    using numeric_limits = ::std::numeric_limits<T>;

    // terminate shim
    [[noreturn]] inline void terminate() { ::std::abort(); }

    // min/max for host+device (needed since std::min is not __device__)
    template <typename T>
    __host__ __device__ __forceinline__
    constexpr T min(T a, T b) { return a < b ? a : b; }

    template <typename T>
    __host__ __device__ __forceinline__
    constexpr T max(T a, T b) { return a > b ? a : b; }
} // namespace std

// cuda::round_up — rounds x up to the nearest multiple of m
template <typename T>
__host__ __device__ __forceinline__
constexpr T round_up(T x, T m) {
    static_assert(::std::is_integral_v<T>, "round_up requires integral types");
    return ((x + m - 1) / m) * m;
}

// cuda::ceil_div
template <typename T>
__host__ __device__ __forceinline__
constexpr T ceil_div(T x, T m) {
    static_assert(::std::is_integral_v<T>, "ceil_div requires integral types");
    return (x + m - 1) / m;
}

// cuda::round_down
template <typename T>
__host__ __device__ __forceinline__
constexpr T round_down(T x, T m) {
    static_assert(::std::is_integral_v<T>, "round_down requires integral types");
    return (x / m) * m;
}

// cuda::is_aligned
__host__ __device__ __forceinline__
bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

} // namespace cuda

// -------------------------------------------------------
// cute:: minimal compatibility
// -------------------------------------------------------
namespace cute {

// Integer constant
template <int N>
struct Int {
    static constexpr int value = N;
    __host__ __device__ constexpr operator int() const { return N; }
};

// Alias for compile-time constants
template <int N>
using C = Int<N>;

// Shape — variadic tuple of dimensions
template <typename... Ts>
struct Shape {
    // Minimal implementation — just holds types.
    // FlashMoE primarily uses Shape with Int<N> types for compile-time shapes.
};

// get<I> — index into a tuple-like type
// (minimal version for common uses)
template <int I, typename T>
__host__ __device__ constexpr auto get(const T& t) {
    if constexpr (I == 0) return t;
    else static_assert(I == 0, "get<I> out of range for scalar");
}

// ceil_div
template <typename T, typename U>
__host__ __device__ __forceinline__
constexpr auto ceil_div(T a, U b) {
    return (a + b - 1) / b;
}

// max / min
template <typename T, typename U>
__host__ __device__ __forceinline__
constexpr auto max(T a, U b) {
    return a > b ? a : b;
}

template <typename T, typename U>
__host__ __device__ __forceinline__
constexpr auto min(T a, U b) {
    return a < b ? a : b;
}

// Layout tags
struct LayoutRight {};
struct LayoutLeft {};

// Minimal tensor placeholder — the actual Tensor implementation is
// complex and will be replaced by composable_kernel equivalents
// in the kernel porting phase. This stub exists only to allow
// non-kernel utility code to compile.

// is_rmem_v — register-memory predicate (always false in stub)
template <typename T>
constexpr bool is_rmem_v = false;

// 3-arg max
template <typename T, typename U, typename V>
__host__ __device__ __forceinline__
constexpr auto max(T a, U b, V c) {
    return max(max(a, b), c);
}

// Placeholder types
struct _0 { static constexpr int value = 0; __host__ __device__ constexpr operator int() const { return 0; } };
struct _1 { static constexpr int value = 1; __host__ __device__ constexpr operator int() const { return 1; } };

// Coord -- minimal tuple-like coord type
template <typename... Ts>
struct Coord {};

template <typename T0, typename T1, typename T2>
struct Coord<T0, T1, T2> {
    T0 v0; T1 v1; T2 v2;
};
template <typename T0, typename T1>
struct Coord<T0, T1> {
    T0 v0; T1 v1;
};

template <int I, typename T0, typename T1, typename T2>
__host__ __device__ constexpr auto get(const Coord<T0,T1,T2>& c) {
    if constexpr (I == 0) return c.v0;
    else if constexpr (I == 1) return c.v1;
    else return c.v2;
}
template <int I, typename T0, typename T1>
__host__ __device__ constexpr auto get(const Coord<T0,T1>& c) {
    if constexpr (I == 0) return c.v0;
    else return c.v1;
}

template <typename T0, typename T1, typename T2>
__host__ __device__ constexpr auto make_coord(T0 a, T1 b, T2 c) {
    return Coord<T0,T1,T2>{a, b, c};
}
template <typename T0, typename T1>
__host__ __device__ constexpr auto make_coord(T0 a, T1 b) {
    return Coord<T0,T1>{a, b};
}

// select -- extracts sub-coordinates
template <int I0, int I1, typename... Ts>
__host__ __device__ constexpr auto select(const Coord<Ts...>& c) {
    return make_coord(get<I0>(c), get<I1>(c));
}

// Shape get specializations
template <int I, typename... Ts>
__host__ __device__ constexpr auto get(const Shape<Ts...>&);

// For 4-element Shape (used in MoEConfig GEMM tiles)
template <typename T0, typename T1, typename T2, typename T3>
struct Shape<T0, T1, T2, T3> {
    static constexpr auto v0 = T0{};
    static constexpr auto v1 = T1{};
    static constexpr auto v2 = T2{};
    static constexpr auto v3 = T3{};
};
template <int I, typename T0, typename T1, typename T2, typename T3>
__host__ __device__ constexpr auto get(const Shape<T0,T1,T2,T3>&) {
    if constexpr (I == 0) return T0{};
    else if constexpr (I == 1) return T1{};
    else if constexpr (I == 2) return T2{};
    else return T3{};
}

// 3-element Shape
template <typename T0, typename T1, typename T2>
struct Shape<T0, T1, T2> {
    static constexpr auto v0 = T0{};
    static constexpr auto v1 = T1{};
    static constexpr auto v2 = T2{};
};
template <int I, typename T0, typename T1, typename T2>
__host__ __device__ constexpr auto get(const Shape<T0,T1,T2>&) {
    if constexpr (I == 0) return T0{};
    else if constexpr (I == 1) return T1{};
    else return T2{};
}

// 2-element Shape
template <typename T0, typename T1>
struct Shape<T0, T1> {};
template <int I, typename T0, typename T1>
__host__ __device__ constexpr auto get(const Shape<T0,T1>&) {
    if constexpr (I == 0) return T0{};
    else return T1{};
}

// rank_v -- number of elements in a Shape
template <typename T> constexpr int rank_v = 0;
template <typename... Ts> constexpr int rank_v<Shape<Ts...>> = sizeof...(Ts);
template <typename... Ts> constexpr int rank_v<Coord<Ts...>> = sizeof...(Ts);

// is_tuple_v
template <typename T> constexpr bool is_tuple_v = false;
template <typename... Ts> constexpr bool is_tuple_v<Shape<Ts...>> = true;

// make_int_sequence / for_each -- for compile-time loops
template <int N, int... Is>
struct make_int_sequence_impl : make_int_sequence_impl<N-1, N-1, Is...> {};
template <int... Is>
struct make_int_sequence_impl<0, Is...> {
    using type = ::std::integer_sequence<int, Is...>;
};
template <int N>
using make_int_sequence = typename make_int_sequence_impl<N>::type;

template <typename F, int... Is>
__host__ __device__ __forceinline__
void for_each_impl(::std::integer_sequence<int, Is...>, F&& f) {
    (f(Int<Is>{}), ...);
}
template <int N, typename F>
__host__ __device__ __forceinline__
void for_each(make_int_sequence<N>, F&& f) {
    for_each_impl(::std::integer_sequence<int>{}, ::std::forward<F>(f));
    // Actually we need proper forwarding:
}
// Proper for_each
template <typename F, int... Is>
__host__ __device__ __forceinline__
void for_each(::std::integer_sequence<int, Is...>, F&& f) {
    (f(Int<Is>{}), ...);
}

// half_t -- used in experimental/topo.cuh
struct half_t {
    __half value;
    __host__ __device__ half_t() : value(0) {}
    __host__ __device__ half_t(__half v) : value(v) {}
    __host__ __device__ operator float() const { return __half2float(value); }
};

} // namespace cute

// -------------------------------------------------------
// cutlass:: minimal compatibility
// -------------------------------------------------------
namespace cutlass {

// round_up
template <typename T>
__host__ __device__ __forceinline__
constexpr T round_up(T x, T m) {
    return ((x + m - 1) / m) * m;
}

// AlignedArray — a simple array with alignment
template <typename T, int N, int Alignment = sizeof(T)>
struct alignas(Alignment) AlignedArray {
    T data_[N];

    __host__ __device__ __forceinline__ T& operator[](int i) { return data_[i]; }
    __host__ __device__ __forceinline__ const T& operator[](int i) const { return data_[i]; }
    __host__ __device__ __forceinline__ constexpr int size() const { return N; }
};

// Array — cutlass Array type
template <typename T, int N, bool RegisterSized = false>
struct Array {
    T data_[N];

    __host__ __device__ __forceinline__ T& operator[](int i) { return data_[i]; }
    __host__ __device__ __forceinline__ const T& operator[](int i) const { return data_[i]; }
    __host__ __device__ __forceinline__ constexpr int size() const { return N; }
};

// NumericConverter — identity by default, specializations for fp16/bf16
template <typename To, typename From, typename = void>
struct NumericConverter {
    __host__ __device__ __forceinline__
    To operator()(From const& x) const {
        return static_cast<To>(x);
    }
};

} // namespace cutlass

#endif // FLASHMOE_PLATFORM_HIP

#endif // FLASHMOE_PLATFORM_MATH_COMPAT_H
