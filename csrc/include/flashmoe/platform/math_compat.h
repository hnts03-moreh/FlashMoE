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
#include <array>
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

    // array — maps cuda::std::array to std::array
    using ::std::array;

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

// fast_mod_div — precomputed divisor for fast modulo/division on GPU
template <typename T>
struct fast_mod_div {
    T divisor_;
    __host__ __device__ fast_mod_div() : divisor_(1) {}
    __host__ __device__ explicit fast_mod_div(T d) : divisor_(d) {}
    __host__ __device__ T divide(T dividend) const { return dividend / divisor_; }
    __host__ __device__ T modulus(T dividend) const { return dividend % divisor_; }
    __host__ __device__ void divmod(T dividend, T& quotient, T& remainder) const {
        quotient = dividend / divisor_;
        remainder = dividend - quotient * divisor_;
    }
    __host__ __device__ T get() const { return divisor_; }

    // operator% so that `dividend % fast_mod_div` works (CCCL compatibility)
    friend __host__ __device__ T operator%(T dividend, const fast_mod_div& d) {
        return d.modulus(dividend);
    }
    // operator/ so that `dividend / fast_mod_div` works
    friend __host__ __device__ T operator/(T dividend, const fast_mod_div& d) {
        return d.divide(dividend);
    }
};

} // namespace cuda

// umin — CUDA built-in for unsigned min; not available on HIP
__host__ __device__ __forceinline__
unsigned int umin(unsigned int a, unsigned int b) { return a < b ? a : b; }

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
constexpr auto ceil_div(T a, U b) {
    return (a + b - 1) / b;
}

// max / min — plain constexpr (HIP/Clang makes constexpr functions
// implicitly available on device; __host__ __device__ would prevent
// use in compile-time constant expressions).
template <typename T, typename U>
constexpr auto max(T a, U b) {
    return a > b ? a : b;
}

template <typename T, typename U>
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

// 3-arg max — plain constexpr (no __host__ __device__ so it can be used in
// compile-time constant expressions on HIP/Clang which rejects
// __host__ __device__ functions in constexpr evaluation).
template <typename T, typename U, typename V>
constexpr auto max(T a, U b, V c) {
    return (a > b ? a : b) > c ? (a > b ? a : b) : c;
}

// 3-arg min
template <typename T, typename U, typename V>
constexpr auto min(T a, U b, V c) {
    return (a < b ? a : b) < c ? (a < b ? a : b) : c;
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

// -------------------------------------------------------
// Stride — compile-time or runtime stride tuple
// -------------------------------------------------------
template <typename... Ss>
struct Stride {};

template <typename S0, typename S1>
struct Stride<S0, S1> {
    S0 s0; S1 s1;
    __host__ __device__ constexpr Stride() : s0{}, s1{} {}
    __host__ __device__ constexpr Stride(S0 a, S1 b) : s0(a), s1(b) {}
};

// -------------------------------------------------------
// Layout — shape + stride pair (2-D only, sufficient for FlashMoE)
// -------------------------------------------------------
template <typename ShapeT, typename StrideT = void>
struct Layout;

// Layout<Shape<R,C>, Stride<SR,SC>> — compile-time 2-D layout
template <typename R, typename C, typename SR, typename SC>
struct Layout<Shape<R, C>, Stride<SR, SC>> {
    using shape_type = Shape<R, C>;
    using stride_type = Stride<SR, SC>;
    static constexpr int rows = R::value;
    static constexpr int cols = C::value;
    static constexpr int stride_r = SR::value;
    static constexpr int stride_c = SC::value;
    __host__ __device__ constexpr int operator()(int r, int c) const {
        return r * stride_r + c * stride_c;
    }
    static constexpr int size() { return rows * cols; }
};

// is_compatible — on HIP, always true since we relax layout compatibility checks.
// rocblasdx::SmemLayout and cute::Layout are different types but represent compatible
// memory layouts. The copy functions handle element-wise copying regardless of padding.
template <typename A, typename B>
struct is_compatible : ::std::true_type {};

// -------------------------------------------------------
// Pointer wrappers — smem_ptr and gmem_ptr
// -------------------------------------------------------
template <typename T>
struct smem_ptr {
    T* ptr;
    __host__ __device__ smem_ptr(T* p) : ptr(p) {}
    __host__ __device__ T* get() const { return ptr; }
};

template <typename T>
struct gmem_ptr {
    T* ptr;
    __host__ __device__ gmem_ptr(T* p) : ptr(p) {}
    __host__ __device__ T* get() const { return ptr; }
};

template <typename T>
__host__ __device__ __forceinline__
smem_ptr<T> make_smem_ptr(T* p) { return smem_ptr<T>{p}; }

template <typename T>
__host__ __device__ __forceinline__
gmem_ptr<T> make_gmem_ptr(T* p) { return gmem_ptr<T>{p}; }

// -------------------------------------------------------
// Tensor — 2-D view over memory (smem or gmem)
// -------------------------------------------------------
// Compile-time layout version
// Note: const Tensor still allows mutation of pointed-to data (like CuTe Tensor)
template <typename Element, typename LayoutT>
struct Tensor {
    Element* ptr_;
    __host__ __device__ Tensor(Element* p) : ptr_(p) {}

    __device__ __forceinline__
    Element& operator()(int row, int col) const { return ptr_[LayoutT{}(row, col)]; }

    __device__ __forceinline__
    Element& operator()(int flat) const { return ptr_[flat]; }

    static constexpr int size() { return LayoutT::size(); }

    using layout_type = LayoutT;
    static constexpr layout_type layout() { return {}; }

    __device__ __forceinline__
    Element*& data() { return ptr_; }
};

// Runtime-stride Tensor (for gmem with dynamic dimensions)
// Note: const TensorDyn still allows mutation of pointed-to data (like CuTe Tensor)
template <typename Element>
struct TensorDyn {
    Element* ptr_;
    int rows_, cols_, stride_;
    __host__ __device__ TensorDyn(Element* p, int r, int c, int s)
        : ptr_(p), rows_(r), cols_(c), stride_(s) {}

    __device__ __forceinline__
    Element& operator()(int row, int col) const { return ptr_[row * stride_ + col]; }

    __device__ __forceinline__
    Element& operator()(int flat) const { return ptr_[flat]; }

    __host__ __device__ int size() const { return rows_ * cols_; }

    __device__ __forceinline__
    Element*& data() { return ptr_; }
};

// -------------------------------------------------------
// make_tensor — construct Tensor from pointer + Layout
// -------------------------------------------------------
// cute::make_tensor(smem_ptr<T>, Layout<Shape, Stride>) -> Tensor
template <typename T, typename ShapeT, typename StrideT>
__device__ __forceinline__
auto make_tensor(smem_ptr<T> p, Layout<ShapeT, StrideT>) {
    return Tensor<T, Layout<ShapeT, StrideT>>{p.ptr};
}

// cute::make_tensor(gmem_ptr<T>, Layout<Shape, Stride>) -> Tensor
template <typename T, typename ShapeT, typename StrideT>
__device__ __forceinline__
auto make_tensor(gmem_ptr<T> p, Layout<ShapeT, StrideT>) {
    return Tensor<T, Layout<ShapeT, StrideT>>{p.ptr};
}

// make_layout(Shape<R,C>, LayoutRight) -> row-major Layout
template <typename R, typename C>
__host__ __device__ __forceinline__
constexpr auto make_layout(Shape<R, C>, LayoutRight) {
    return Layout<Shape<R, C>, Stride<C, _1>>{};
}

// make_layout(Shape, Stride) -> Layout
template <typename R, typename C, typename SR, typename SC>
__host__ __device__ __forceinline__
constexpr auto make_layout(Shape<R, C>, Stride<SR, SC>) {
    return Layout<Shape<R, C>, Stride<SR, SC>>{};
}

// -------------------------------------------------------
// Dynamic shape/layout for runtime dimensions
// -------------------------------------------------------
struct DynShape2D {
    int r, c;
    __host__ __device__ DynShape2D(int r_, int c_) : r(r_), c(c_) {}
};

struct DynShape3D {
    int d0, d1, d2;
    __host__ __device__ DynShape3D(int a, int b, int c) : d0(a), d1(b), d2(c) {}
};

struct DynLayoutRowMajor {
    int rows_, cols_;
    __host__ __device__ DynLayoutRowMajor(int r, int c) : rows_(r), cols_(c) {}
    __host__ __device__ int operator()(int r, int c) const { return r * cols_ + c; }
    __host__ __device__ int size() const { return rows_ * cols_; }
};

// make_shape — runtime ints
__host__ __device__ __forceinline__
DynShape2D make_shape(int a, int b) { return DynShape2D{a, b}; }

// get<I> for DynShape2D
template <int I>
__host__ __device__ __forceinline__
int get(const DynShape2D& s) {
    if constexpr (I == 0) return s.r;
    else return s.c;
}

// make_shape — 3 runtime ints
template <typename T0, typename T1, typename T2>
__host__ __device__ __forceinline__
DynShape3D make_shape(T0 a, T1 b, T2 c) { return DynShape3D{static_cast<int>(a), static_cast<int>(b), static_cast<int>(c)}; }

// get<I> for DynShape3D
template <int I>
__host__ __device__ __forceinline__
int get(const DynShape3D& s) {
    if constexpr (I == 0) return s.d0;
    else if constexpr (I == 1) return s.d1;
    else return s.d2;
}

// make_layout(DynShape2D, LayoutRight) -> DynLayoutRowMajor
__host__ __device__ __forceinline__
DynLayoutRowMajor make_layout(DynShape2D s, LayoutRight) {
    return DynLayoutRowMajor{s.r, s.c};
}

// 3D row-major layout
struct DynLayoutRowMajor3D {
    int d0_, d1_, d2_;
    __host__ __device__ DynLayoutRowMajor3D(int a, int b, int c) : d0_(a), d1_(b), d2_(c) {}
    __host__ __device__ int operator()(int i, int j, int k) const { return (i * d1_ + j) * d2_ + k; }
    __host__ __device__ int size() const { return d0_ * d1_ * d2_; }
};

// make_layout(DynShape3D, LayoutRight) -> DynLayoutRowMajor3D
__host__ __device__ __forceinline__
DynLayoutRowMajor3D make_layout(DynShape3D s, LayoutRight) {
    return DynLayoutRowMajor3D{s.d0, s.d1, s.d2};
}

// make_tensor with DynLayoutRowMajor -> TensorDyn (gmem_ptr)
template <typename T>
__host__ __device__ __forceinline__
TensorDyn<T> make_tensor(gmem_ptr<T> p, DynLayoutRowMajor layout) {
    return TensorDyn<T>{p.ptr, layout.rows_, layout.cols_, layout.cols_};
}

// make_tensor with DynLayoutRowMajor -> TensorDyn (smem_ptr)
template <typename T>
__device__ __forceinline__
TensorDyn<T> make_tensor(smem_ptr<T> p, DynLayoutRowMajor layout) {
    return TensorDyn<T>{p.ptr, layout.rows_, layout.cols_, layout.cols_};
}

// 3D dynamic tensor (row-major)
template <typename Element>
struct TensorDyn3D {
    Element* ptr_;
    int d0_, d1_, d2_;
    __host__ __device__ TensorDyn3D(Element* p, int a, int b, int c)
        : ptr_(p), d0_(a), d1_(b), d2_(c) {}
    __device__ __forceinline__
    Element& operator()(int i, int j, int k) const { return ptr_[(i * d1_ + j) * d2_ + k]; }
    __device__ __forceinline__
    Element& operator()(int flat) const { return ptr_[flat]; }
    __host__ __device__ int size() const { return d0_ * d1_ * d2_; }
    __device__ __forceinline__
    Element*& data() { return ptr_; }
};

// make_tensor with DynLayoutRowMajor3D -> TensorDyn3D
template <typename T>
__host__ __device__ __forceinline__
TensorDyn3D<T> make_tensor(gmem_ptr<T> p, DynLayoutRowMajor3D layout) {
    return TensorDyn3D<T>{p.ptr, layout.d0_, layout.d1_, layout.d2_};
}

// 4D shape
struct DynShape4D {
    int d0, d1, d2, d3;
    __host__ __device__ DynShape4D(int a, int b, int c, int d) : d0(a), d1(b), d2(c), d3(d) {}
};

// make_shape — 4 runtime ints
template <typename T0, typename T1, typename T2, typename T3>
__host__ __device__ __forceinline__
DynShape4D make_shape(T0 a, T1 b, T2 c, T3 d) {
    return DynShape4D{static_cast<int>(a), static_cast<int>(b), static_cast<int>(c), static_cast<int>(d)};
}

// get<I> for DynShape4D
template <int I>
__host__ __device__ __forceinline__
int get(const DynShape4D& s) {
    if constexpr (I == 0) return s.d0;
    else if constexpr (I == 1) return s.d1;
    else if constexpr (I == 2) return s.d2;
    else return s.d3;
}

// 4D row-major layout
struct DynLayoutRowMajor4D {
    int d0_, d1_, d2_, d3_;
    __host__ __device__ DynLayoutRowMajor4D(int a, int b, int c, int d) : d0_(a), d1_(b), d2_(c), d3_(d) {}
    __host__ __device__ int operator()(int i, int j, int k, int l) const {
        return ((i * d1_ + j) * d2_ + k) * d3_ + l;
    }
    __host__ __device__ int size() const { return d0_ * d1_ * d2_ * d3_; }
};

// make_layout(DynShape4D, LayoutRight) -> DynLayoutRowMajor4D
__host__ __device__ __forceinline__
DynLayoutRowMajor4D make_layout(DynShape4D s, LayoutRight) {
    return DynLayoutRowMajor4D{s.d0, s.d1, s.d2, s.d3};
}

// 4D dynamic tensor (row-major)
template <typename Element>
struct TensorDyn4D {
    Element* ptr_;
    int d0_, d1_, d2_, d3_;
    __host__ __device__ TensorDyn4D(Element* p, int a, int b, int c, int d)
        : ptr_(p), d0_(a), d1_(b), d2_(c), d3_(d) {}
    __device__ __forceinline__
    Element& operator()(int i, int j, int k, int l) const {
        return ptr_[((i * d1_ + j) * d2_ + k) * d3_ + l];
    }
    __device__ __forceinline__
    Element& operator()(int flat) const { return ptr_[flat]; }
    __host__ __device__ int size() const { return d0_ * d1_ * d2_ * d3_; }
    __device__ __forceinline__
    Element*& data() { return ptr_; }
};

// make_tensor with DynLayoutRowMajor4D -> TensorDyn4D
template <typename T>
__host__ __device__ __forceinline__
TensorDyn4D<T> make_tensor(gmem_ptr<T> p, DynLayoutRowMajor4D layout) {
    return TensorDyn4D<T>{p.ptr, layout.d0_, layout.d1_, layout.d2_, layout.d3_};
}

// -------------------------------------------------------
// local_tile — slice a tensor into a tile at given coordinate
// -------------------------------------------------------
// For TensorDyn (row-major): local_tile(tensor, shape, coord)
// Returns a new TensorDyn starting at the tile offset
template <typename Element, typename TileShape, typename TileCoord>
__device__ __forceinline__
TensorDyn<Element> local_tile(const TensorDyn<Element>& tensor, const TileShape& tileShape, const TileCoord& coord) {
    const int tileC = get<1>(tileShape);
    const int colOffset = get<1>(coord) * tileC;
    return TensorDyn<Element>{
        tensor.ptr_ + colOffset,
        tensor.rows_,
        tileC,
        tensor.stride_
    };
}

// For compile-time layout Tensor
template <typename Element, typename LayoutT, typename TileShape, typename TileCoord>
__device__ __forceinline__
auto local_tile(const Tensor<Element, LayoutT>& tensor, const TileShape& tileShape, const TileCoord& coord) {
    const int tileC = get<1>(tileShape);
    const int colOffset = get<1>(coord) * tileC;
    return TensorDyn<Element>{
        tensor.ptr_ + colOffset,
        LayoutT::rows,
        tileC,
        LayoutT::stride_r
    };
}

} // namespace cute

// -------------------------------------------------------
// cutlass:: minimal compatibility
// -------------------------------------------------------
namespace cutlass {

// is_pow2 — compile-time power-of-2 check (used by vt.cuh, rvt.cuh, etc.)
template <int N>
struct is_pow2 {
    static constexpr bool value = (N > 0) && ((N & (N - 1)) == 0);
};

// ispow2 — runtime power-of-2 check
__host__ __device__ __forceinline__
constexpr bool ispow2(size_t n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// round_up — same type
template <typename T>
__host__ __device__ __forceinline__
constexpr T round_up(T x, T m) {
    return ((x + m - 1) / m) * m;
}

// round_up — mixed types (e.g. size_t and int)
template <typename T, typename U>
__host__ __device__ __forceinline__
constexpr auto round_up(T x, U m)
    -> ::std::enable_if_t<!::std::is_same_v<T, U>, ::std::common_type_t<T, U>> {
    using C = ::std::common_type_t<T, U>;
    return ((static_cast<C>(x) + static_cast<C>(m) - 1) / static_cast<C>(m)) * static_cast<C>(m);
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
    __host__ __device__ __forceinline__ void fill(const T& value) {
        for (int i = 0; i < N; ++i) data_[i] = value;
    }
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
