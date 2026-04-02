/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by osayamen on 12/25/25.
//

#ifndef FLASHMOE_TILE_CUH
#define FLASHMOE_TILE_CUH

#include "flashmoe/platform/platform.h"
#include "flashmoe/platform/device.h"
#include "flashmoe/platform/math_compat.h"

#if defined(FLASHMOE_PLATFORM_HIP)
#include <rocblasdx/rocblasdx.hpp>
// On HIP, cutlass::NumericConverter is provided by math_compat.h
#else
#include <cublasdx.hpp>
#include <cutlass/numeric_conversion.h>
#endif

// Namespace alias so that code below can use a single namespace for the BLAS descriptor API
#if defined(FLASHMOE_PLATFORM_HIP)
namespace flashmoe_blas = rocblasdx;
#else
namespace flashmoe_blas = cublasdx;
#endif

namespace flashmoe::heuristics {
  template<int M, int Arch>
  consteval int getTileM() {
    static_assert(M > 0 && (M == 1 || M % 2 == 0)); // allows for decode lengths, where M is typically small
    constexpr int cap = (Arch >= 900 && M >= 4096) ? 256 : 128;

    for (const int t : {128, 96, 64, 48, 32, 24, 16, 8}) {
      if (t <= cap && t <= M && (M % t == 0)) {
        return t;
      }
    }
    return M < 128 ? M : 1;
  }
  template<int M, int Arch>
  consteval int getMoETileM() {
    // not a requirement for M to be a multiple of 2; however, I have observed absurd codegen for "unfriendly" shapes.
    // "Absurd" here refers to register spilling.
    if constexpr (Arch >= 900 && M >= 4096 && M % 2 == 0) {
      return cute::max(cute::min(M, 256), 16);
    }
    return cute::max(cute::min(M, 128), 16);
  }
  template<int N, typename Element>
  consteval int getTileN() {
    static_assert(N > 0 && N % 8 == 0);
    constexpr int cap = (sizeof(Element) <= 2) ? 128 : 64;

    for (const int t : {128, 96, 64, 48, 32, 24, 16, 8}) {
      if (t <= cap && t <= N && (N % t == 0)) {
        return t;
      }
    }
    return 8;
  }
  template<int N, int cap>
  consteval int getGateTileN() {
    static_assert(N > 0 && N % 8 == 0);

    for (const int t : {128, 96, 64, 48, 32, 24, 16, 8}) {
      if (t <= cap && t <= N && (N % t == 0)) {
        return t;
      }
    }
    return 8;
  }

  template<int K, typename Element>
  consteval int getGateTileK() {
    static_assert(K > 0 && K % 16 == 0);
    constexpr int cap = cuda::std::is_same_v<Element, double> ? 32 : 64;
    for (const int t : {128, 96, 64, 48, 32, 24, 16}) {
      if (t <= cap && t <= K && (K % t == 0)) {
        return t;
      }
    }
    return 16;
  }

  template<int K, int cap>
  consteval int getTileK() {
    static_assert(K > 0 && K % 16 == 0);
    for (const int t : {128, 96, 64, 48, 32, 24, 16}) {
      if (t <= cap && t <= K && (K % t == 0)) {
        return t;
      }
    }
    return 16;
  }

  template<int K, int bK, int Arch>
  consteval int getPipeStages() {
    static_assert(K % bK == 0);
    constexpr int cap = (Arch>= 900 ? 3 : Arch >= 800 ? 2 : 1);
    constexpr int stages = K / bK;
    return stages >= cap ? cap : stages;
  }
}

namespace flashmoe
{
  enum class CombineMode {
    single, // top k = 1
    plural // top k > 1
  };
  enum class MLPMatmulType {
    gated = 0,
    vanilla = 1
  };
  template<int m>
  consteval MLPMatmulType defineMLPType() {
    static_assert(m == 0 || m == 1, "Invalid MLPMatmulType enum value");
    if constexpr(m == 0) {
      return MLPMatmulType::gated;
    }
    else {
      return MLPMatmulType::vanilla;
    }
  }
  template<int t>
  struct DataType {
    static_assert(t >= 0 && t < 4, "Invalid datatype constant");
  };

  template<>
  struct DataType<0> {
    using Type = __nv_bfloat16;
  };
  template<>
  struct DataType<1> {
    using Type = __half;
  };
  template<>
  struct DataType<2> {
    using Type = float;
  };
  template<>
  struct DataType<3> {
    using Type = double;
  };

  template <typename T, typename S>
  struct Converter {
    __device__ auto operator()(const S& x) const {
      return static_cast<T>(x);
    }
  };

  template <>
  struct Converter<__half, float> {
    __device__ auto operator()(const float& x) const {
      return __float2half(x);
    }
  };

  template <>
  struct Converter<float, __half> {
    __device__ auto operator()(const __half& x) const {
      return __half2float(x);
    }
  };

  template <>
  struct Converter<__nv_bfloat16, float> {
    __device__ auto operator()(const float& x) const {
#if defined(FLASHMOE_PLATFORM_HIP)
      return hip_bfloat16(x);
#else
      return __float2bfloat16(x);
#endif
    }
  };

  template <>
  struct Converter<float, __nv_bfloat16> {
    __device__ auto operator()(const __nv_bfloat16& x) const {
#if defined(FLASHMOE_PLATFORM_HIP)
      return static_cast<float>(x);
#else
      return __bfloat162float(x);
#endif
    }
  };

  template <>
  struct Converter<float2, __half2> {
    __device__ auto operator()(const __half2& x) const {
      return __half22float2(x);
    }
  };

  template <>
  struct Converter<__half2, float2> {
    __device__ auto operator()(const float2& x) const {
      return __float22half2_rn(x);
    }
  };

  template <>
  struct Converter<float2, __nv_bfloat162> {
    __device__ auto operator()(const __nv_bfloat162& x) const {
      return __bfloat1622float2(x);
    }
  };

  template <>
  struct Converter<__nv_bfloat162, float2> {
    __device__ auto operator()(const float2& x) const {
      return __float22bfloat162_rn(x);
    }
  };

  template <>
  struct Converter<float, double> {
    __device__ auto operator()(const double& x) const {
      return __double2float_rn(x);
    }
  };

  template <>
  struct Converter<flashmoe_blas::tfloat32_t, float> {
    __device__ auto operator()(const float& x) const {
#if defined(FLASHMOE_PLATFORM_HIP)
      // MI300X has no native TF32; tfloat32_t is a float wrapper in rocBLASDx
      return flashmoe_blas::tfloat32_t{x};
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      uint32_t storage = cuda::std::bit_cast<uint32_t>(x);
      asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(storage) : "r"(storage));
      return cuda::std::bit_cast<flashmoe_blas::tfloat32_t>(storage);
#else
      constexpr cutlass::NumericConverter<cutlass::tfloat32_t, float> c{};
      return cuda::std::bit_cast<flashmoe_blas::tfloat32_t>(c(x));
#endif
    }
  };


  template <class T>
  struct isTensor : cuda::std::false_type {
  };

#if !defined(FLASHMOE_PLATFORM_HIP)
  template <class Engine, class Layout>
  struct isTensor<cute::Tensor<Engine, Layout>> : cuda::std::true_type {
  };

  template <class Engine, class Layout>
  struct isTensor<const cute::Tensor<Engine, Layout>> : cuda::std::true_type {
  };
#endif
}

namespace flashmoe::tile
{
  constexpr int MAX_ALIGN = 16;

  template <int N>
  __device__ __forceinline__
  void cpWait() {
#if defined(FLASHMOE_PLATFORM_HIP)
    // ROCm: no cp.async; synchronous loads used instead. Just barrier.
    __syncthreads();
#else
    cute::cp_async_wait<N>();
    __syncthreads();
#endif
  }

  __device__ __forceinline__
  void cpFence() {
#if defined(FLASHMOE_PLATFORM_HIP)
    // ROCm: no cp.async fence needed; synchronous loads.
    // No-op -- the __syncthreads() in cpWait provides ordering.
#else
    cute::cp_async_fence();
#endif
  }

  __device__ __forceinline__
  constexpr auto idx2Coord(const int& tilesM, const int& tilesN, const int& tileIdx) {
#if defined(FLASHMOE_PLATFORM_HIP)
    // Replicate cute::idx2crd for a 2-D shape with stride = (tilesN, 1)
    const auto row = tileIdx / tilesN;
    const auto col = tileIdx % tilesN;
    return cute::make_coord(row, col, cute::Int<0>{});
#else
    const auto tileCoord = cute::idx2crd(tileIdx, cute::make_shape(tilesM, tilesN),
                                         cute::make_stride(tilesN, cute::_1{}));
    return cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
#endif
  }

#if defined(FLASHMOE_PLATFORM_HIP)
  // HIP tile accessors — simplified versions without deep CuTe tensor operations
  // GmemTileView: a 2D view into a tile of a larger matrix
  template <typename Element, int TileR, int TileC, flashmoe_blas::arrangement ar>
  struct GmemTileView {
      Element* ptr_;
      int global_stride_;  // stride of the full matrix (nCol for row-major, nRow for col-major)

      __device__ __forceinline__
      Element& operator()(int row, int col) {
          if constexpr (ar == flashmoe_blas::row_major)
              return ptr_[row * global_stride_ + col];
          else
              return ptr_[col * global_stride_ + row];
      }

      __device__ __forceinline__
      const Element& operator()(int row, int col) const {
          if constexpr (ar == flashmoe_blas::row_major)
              return ptr_[row * global_stride_ + col];
          else
              return ptr_[col * global_stride_ + row];
      }

      // Flat indexing (row-major within tile)
      __device__ __forceinline__
      Element& operator()(int flat) {
          int row = flat / TileC;
          int col = flat % TileC;
          return (*this)(row, col);
      }

      __device__ __forceinline__
      const Element& operator()(int flat) const {
          int row = flat / TileC;
          int col = flat % TileC;
          return (*this)(row, col);
      }

      static constexpr int size() { return TileR * TileC; }

      using layout_type = typename rocblasdx::SuggestSmemLayout<ar, TileR, TileC, Element>::type;
      static constexpr layout_type layout() { return {}; }
  };

} // namespace flashmoe::tile (temporarily close)

// MFMA-aware copy_fragment overloads for GmemTileView
// These must be in the rocblasdx namespace so flashmoe_blas::copy_fragment finds them.
namespace rocblasdx {

// GmemTileView -> Fragment (MFMA-aware bias/data loading)
template <int Align, typename Element, int TileR, int TileC,
          rocblasdx::arrangement ar, typename FT, int FN>
__device__ __forceinline__
void copy_fragment(const flashmoe::tile::GmemTileView<Element, TileR, TileC, ar>& src,
                   Fragment<FT, FN>& frag, const auto&) {
#if defined(__gfx9__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int row, col;
        if (detail::auto_mfma_frag_to_rc<std::remove_const_t<Element>, TileR, TileC>(i, row, col))
            frag(i) = static_cast<FT>(src(row, col));
    }
#else
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int flat = tid + i * threads;
        frag(i) = static_cast<FT>(src(flat));
    }
#endif
}

// GmemTileView (const Element) -> Fragment
template <int Align, typename Element, int TileR, int TileC,
          rocblasdx::arrangement ar, typename FT, int FN>
__device__ __forceinline__
void copy_fragment(const flashmoe::tile::GmemTileView<const Element, TileR, TileC, ar>& src,
                   Fragment<FT, FN>& frag, const auto&) {
#if defined(__gfx9__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int row, col;
        if (detail::auto_mfma_frag_to_rc<Element, TileR, TileC>(i, row, col))
            frag(i) = static_cast<FT>(src(row, col));
    }
#else
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int flat = tid + i * threads;
        frag(i) = static_cast<FT>(src(flat));
    }
#endif
}

// Fragment -> GmemTileView (MFMA-aware result writing)
template <int Align, typename FT, int FN, typename Element, int TileR, int TileC,
          rocblasdx::arrangement ar>
__device__ __forceinline__
void copy_fragment(const rocblasdx::Fragment<FT, FN>& frag,
                   flashmoe::tile::GmemTileView<Element, TileR, TileC, ar>& dst,
                   const auto&) {
#if defined(__gfx9__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int row, col;
        if (detail::auto_mfma_frag_to_rc<std::remove_const_t<Element>, TileR, TileC>(i, row, col))
            dst(row, col) = static_cast<Element>(frag(i));
    }
#else
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int flat = tid + i * threads;
        dst(flat) = static_cast<Element>(frag(i));
    }
#endif
}

// SmemTensor -> GmemTileView copy (cooperative, uses 2D indexing to handle stride padding)
template <typename BLAS, int Align,
          typename SrcElement, int R, int C, int S,
          typename DstElement, int TileR, int TileC, rocblasdx::arrangement ar>
__device__ __forceinline__
void copy(const rocblasdx::SmemTensor<SrcElement, R, C, S>& src,
          flashmoe::tile::GmemTileView<DstElement, TileR, TileC, ar>& dst) {
    constexpr int threads = BLAS::max_threads_per_block;
    const int tid = threadIdx.x;
    constexpr int total = R * C;
    constexpr int per = (total + threads - 1) / threads;
    for (int i = 0; i < per; ++i) {
        int flat = tid + i * threads;
        if (flat < total) {
            int row = flat / C;
            int col = flat % C;
            dst(row, col) = static_cast<DstElement>(src(row, col));
        }
    }
}

} // namespace rocblasdx

namespace flashmoe::tile {

  template <int tRow, int tCol, int ar_int, typename Element, typename TileCoord>
  __device__ __forceinline__
  auto get(const Element* __restrict__ const& p, const int& nRow, const int& nCol,
                     const TileCoord& tileCoord) {
      constexpr auto ar = static_cast<flashmoe_blas::arrangement>(ar_int);
      const int tileRow = cute::get<0>(tileCoord);
      const int tileCol = cute::get<1>(tileCoord);
      const Element* base = (ar == flashmoe_blas::row_major)
          ? p + tileRow * tRow * nCol + tileCol * tCol
          : p + tileCol * tCol * nRow + tileRow * tRow;
      const int stride = (ar == flashmoe_blas::row_major) ? nCol : nRow;
      return GmemTileView<const Element, tRow, tCol, ar>{const_cast<const Element*>(base), stride};
  }

  template <int tRow, int tCol, int ar_int, typename Element, typename TileCoord>
  __device__ __forceinline__
  auto getC(Element* __restrict__ const& p, const int& nRow, const int& nCol, const TileCoord& tileCoord) {
      constexpr auto ar = static_cast<flashmoe_blas::arrangement>(ar_int);
      const int tileRow = cute::get<0>(tileCoord);
      const int tileCol = cute::get<1>(tileCoord);
      Element* base = (ar == flashmoe_blas::row_major)
          ? p + tileRow * tRow * nCol + tileCol * tCol
          : p + tileCol * tCol * nRow + tileRow * tRow;
      const int stride = (ar == flashmoe_blas::row_major) ? nCol : nRow;
      return GmemTileView<Element, tRow, tCol, ar>{base, stride};
  }

  template <int bM, int bN, typename Element, typename TileCoord>
  __device__ __forceinline__
  auto getBias(const Element* __restrict__ const& bias, const int& M, const int& N,
                         const TileCoord& tileCoord) {
      // broadcast from {1, N} -> {M, N}: stride for rows is 0
      const int tileCol = cute::get<1>(tileCoord);
      const Element* base = bias + tileCol * bN;
      // For bias broadcast, row stride is 0 — each row reads the same data
      return GmemTileView<const Element, bM, bN, flashmoe_blas::row_major>{const_cast<const Element*>(base), 0};
  }
#else
  // These tile accessors use deep CuTe tensor operations only available on CUDA
  template <int tRow, int tCol, flashmoe_blas::arrangement ar, typename Element, typename TileCoord>
  __device__ __forceinline__
  constexpr auto get(const Element* __restrict__ const& p, const int& nRow, const int& nCol,
                     const TileCoord& tileCoord) {
    const auto stride = cute::conditional_return<ar == flashmoe_blas::row_major>
      (cute::make_stride(nCol, cute::_1{}), cute::make_stride(cute::_1{}, nRow));
    const auto m = cute::make_tensor(cute::make_gmem_ptr(p),
                                     cute::make_layout(cute::make_shape(nRow, nCol), stride));
    return cute::local_tile(m, cute::Shape<cute::Int<tRow>, cute::Int<tCol>>{}, tileCoord);
  }

  template <int tRow, int tCol, flashmoe_blas::arrangement ar, typename Element, typename TileCoord>
  __device__ __forceinline__
  constexpr auto getC(Element* __restrict__ const& p, const int& nRow, const int& nCol, const TileCoord& tileCoord) {
    const auto stride = cute::conditional_return<ar == flashmoe_blas::row_major>
      (cute::make_stride(nCol, cute::_1{}), cute::make_stride(cute::_1{}, nRow));
    auto m = cute::make_tensor(cute::make_gmem_ptr(p),
                               cute::make_layout(cute::make_shape(nRow, nCol), stride));
    return cute::local_tile(m, cute::Shape<cute::Int<tRow>, cute::Int<tCol>>{}, tileCoord);
  }

  template <int bM, int bN, typename Element, typename TileCoord>
  __device__ __forceinline__
  constexpr auto getBias(const Element* __restrict__ const& bias, const int& M, const int& N,
                         const TileCoord& tileCoord) {
    // broadcast from {1, N} -> {M, N}
    const auto mD = cute::make_tensor(cute::make_gmem_ptr(bias),
                                      cute::make_layout(cute::make_shape(M, N), cute::Stride<cute::_0, cute::_1>{}));
    return cute::local_tile(mD, cute::Shape<cute::Int<bM>, cute::Int<bN>>{}, cute::select<0, 1>(tileCoord));
  }

  template <int offset, typename Tensor, typename Element>
  __device__ __forceinline__
  void update_buffer(Tensor& tensor, Element* __restrict__ const& base_ptr, const int& stage) {
    tensor.data() = base_ptr + (stage * offset);
  }

  template <int stage, int offset, typename Tensor, typename Element>
  __device__ __forceinline__
  void update_buffer(Tensor& tensor, Element* __restrict__ const& base_ptr) {
    tensor.data() = base_ptr + (stage * offset);
  }
#endif

  template <flashmoe_blas::arrangement ar, int bM, int bK>
  constexpr int ldA = ar == flashmoe_blas::row_major ? bK : bM;
  template <flashmoe_blas::arrangement br, int bK, int bN>
  constexpr int ldB = br == flashmoe_blas::col_major ? bK : bN;
  template <flashmoe_blas::arrangement cr, int bM, int bN>
  constexpr int ldC = cr == flashmoe_blas::row_major ? bN : bM;

  template <
    int bM, int bN, int bK, // tile shape
    int Arch, // compute capability
    typename Element, // type for A and B
    typename MMA_C, // compute type
    flashmoe_blas::arrangement ar = flashmoe_blas::row_major,
    flashmoe_blas::arrangement br = flashmoe_blas::col_major,
    flashmoe_blas::arrangement cr = flashmoe_blas::row_major,
    int aAlignment = MAX_ALIGN,
    int bAlignment = MAX_ALIGN,
    int cAlignment = MAX_ALIGN
  >
  constexpr int suggest_thread_count() {
    using GhostBLAS = decltype(
      flashmoe_blas::Size<bM, bN, bK>() +
      flashmoe_blas::Precision<Element, Element, MMA_C>() +
      flashmoe_blas::Type<flashmoe_blas::type::real>() +
      flashmoe_blas::Function<flashmoe_blas::function::MM>() +
      flashmoe_blas::Arrangement<ar, br, cr>() +
      flashmoe_blas::Block() +
      flashmoe_blas::Alignment<aAlignment, bAlignment, cAlignment>() +
      flashmoe_blas::StaticBlockDim() +
      flashmoe_blas::EnableInputStreaming() +
      flashmoe_blas::SM<Arch, Arch >= 900 ? flashmoe_blas::sm_modifier::arch_specific : flashmoe_blas::sm_modifier::generic>());
    return GhostBLAS::max_threads_per_block;
  }

  enum class TF32Compute {
    yes,
    no
  };

  template <
    int bM, int bN, int bK, // tile shape
    int Arch, // compute capability
    typename Element, // type for A and B
    typename MMA_C, // compute type
    int threads,
    int pipeStages = 1, // pipeline stages
    TF32Compute tfc = TF32Compute::yes,
    flashmoe_blas::arrangement ar = flashmoe_blas::row_major,
    flashmoe_blas::arrangement br = flashmoe_blas::col_major,
    flashmoe_blas::arrangement cr = flashmoe_blas::row_major,
    int aAlignment = MAX_ALIGN,
    int bAlignment = MAX_ALIGN,
    int cAlignment = MAX_ALIGN
  >
    requires(pipeStages > 0 && Arch >= 700)
  struct CollectiveMainloop {
    using TranslatedElement = cuda::std::conditional_t<
      tfc == TF32Compute::yes && cuda::std::is_same_v<Element, float>, flashmoe_blas::tfloat32_t, Element>;
    using BLAS = cuda::std::conditional_t<cuda::std::is_same_v<TranslatedElement, Element>,
      decltype(
        flashmoe_blas::Size<bM, bN, bK>() +
        flashmoe_blas::Precision<Element, Element, MMA_C>() +
        flashmoe_blas::Type<flashmoe_blas::type::real>() +
        flashmoe_blas::Function<flashmoe_blas::function::MM>() +
        flashmoe_blas::Arrangement<ar, br, cr>() +
        flashmoe_blas::Block() +
        flashmoe_blas::Alignment<aAlignment, bAlignment, cAlignment>() +
        flashmoe_blas::BlockDim<threads>() +
        flashmoe_blas::StaticBlockDim() +
        flashmoe_blas::EnableInputStreaming() +
        flashmoe_blas::SM<Arch, Arch >= 900 ? flashmoe_blas::sm_modifier::arch_specific : flashmoe_blas::sm_modifier::generic>()),
      decltype(
        flashmoe_blas::Size<bM, bN, bK>() +
        flashmoe_blas::Precision<TranslatedElement, TranslatedElement, MMA_C>() +
        flashmoe_blas::Type<flashmoe_blas::type::real>() +
        flashmoe_blas::Function<flashmoe_blas::function::MM>() +
        flashmoe_blas::Arrangement<ar, br, cr>() +
        flashmoe_blas::Block() +
        flashmoe_blas::Alignment<aAlignment, bAlignment, cAlignment>() +
        flashmoe_blas::BlockDim<threads>() +
        flashmoe_blas::StaticBlockDim() +
        flashmoe_blas::SM<Arch, Arch >= 900 ? flashmoe_blas::sm_modifier::arch_specific : flashmoe_blas::sm_modifier::generic>())>;
    using TileArch = cute::Int<Arch>;
    using Threads = cute::Int<threads>;
    using TileShape = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;
    using GeneralAlignment = cute::Int<cute::max(aAlignment, bAlignment, cAlignment)>;
    // SharedSizeAB must account for LDS bank-conflict padding in SmemLayout strides
    using SharedSizeAB = cute::Int<cutlass::round_up(
        (flashmoe_blas::cosize(BLAS::suggest_layout_smem_a()) +
         flashmoe_blas::cosize(BLAS::suggest_layout_smem_b())) * pipeStages * sizeof(Element),
        GeneralAlignment::value)>;
    using SharedSizeC = cute::Int<cutlass::round_up(bM * bN * sizeof(Element), GeneralAlignment::value)>;
    using CArr = cute::C<cr>;
    using AAlign = cute::Int<aAlignment>;
    using BAlign = cute::Int<bAlignment>;
    using CAlign = cute::Int<cAlignment>;
    using AccumType = MMA_C;
    using PipeStages = cute::Int<pipeStages>;

    template <typename Accumulator, typename TileCoord>
#if !defined(FLASHMOE_PLATFORM_HIP)
      requires(cute::rank_v<TileCoord> == 3 && flashmoe_blas::is_blas_execution_v<BLAS>)
#endif
    __device__ __forceinline__
    void operator()(void* __restrict__ const& workspace,
                    const Element* __restrict__ const& a,
                    const Element* __restrict__ const& b,
                    Accumulator& accumulator,
                    const int& M, const int& N, const int& K, const TileCoord& tileCoord) const {
      using TransformType = cuda::std::conditional_t<cuda::std::is_same_v<TranslatedElement, Element>,
                                                     flashmoe_blas::identity, Converter<TranslatedElement, Element>>;
      constexpr TransformType transformOp{};
      // assert(__isShared(workspace));
      accumulator.clear();
      const int tilesK = K / bK;
#if defined(FLASHMOE_PLATFORM_HIP)
      // ROCm path: use rocBLASDx API (same interface as cuBLASDx via the compatibility layer)
      constexpr auto sASS = flashmoe_blas::cosize(BLAS::suggest_layout_smem_a());
      constexpr auto sBSS = flashmoe_blas::cosize(BLAS::suggest_layout_smem_b());
      auto* __restrict__ sAP = static_cast<Element*>(workspace);
      auto* __restrict__ sBP = sAP + (sASS * pipeStages);
      auto sA = flashmoe_blas::make_tensor(sAP, BLAS::suggest_layout_smem_a());
      auto sB = flashmoe_blas::make_tensor(sBP, BLAS::suggest_layout_smem_b());
      // ROCm: no async copy pipeline; synchronous loads
      // Simple mainloop without pipelining
      for (int kStage = 0; kStage < tilesK; ++kStage) {
        // Copy A tile [bM, bK] from global memory to shared memory
        // A is [M, K] row-major; shared A is row_major with padded stride
        // load_a reads: sA.ptr[m * sA_stride + k]
        {
          const auto* __restrict__ gA_ptr = a + cute::get<0>(tileCoord) * bM * K + kStage * bK;
          constexpr int sA_stride = BLAS::SmemLayoutA::stride;
          constexpr int threads_val = BLAS::max_threads_per_block;
          constexpr int total_a = bM * bK;
          constexpr int per_a = (total_a + threads_val - 1) / threads_val;
          for (int ii = 0; ii < per_a; ++ii) {
            int flat = threadIdx.x + ii * threads_val;
            if (flat < total_a) {
              int m = flat / bK;
              int k = flat % bK;
              // Global: row-major with stride K; Shared: row-major with stride sA_stride
              sA.ptr[m * sA_stride + k] = gA_ptr[m * K + k];
            }
          }
        }
        // Copy B tile [bK, bN] from global memory to shared memory
        // B is [K, N] row-major; shared B is col_major with padded stride
        // load_b reads: sB.ptr[n * sB_stride + k]
        {
          const auto* __restrict__ gB_ptr = b + kStage * bK * N + cute::get<1>(tileCoord) * bN;
          constexpr int sB_stride = BLAS::SmemLayoutB::stride;
          constexpr int threads_val = BLAS::max_threads_per_block;
          constexpr int total_b = bK * bN;
          constexpr int per_b = (total_b + threads_val - 1) / threads_val;
          for (int ii = 0; ii < per_b; ++ii) {
            int flat = threadIdx.x + ii * threads_val;
            if (flat < total_b) {
              int k = flat / bN;
              int n = flat % bN;
              // Row-major copy: global stride=N, shared stride=sB_stride
              // (load_b col_major access transposes implicitly)
              sB.ptr[k * sB_stride + n] = gB_ptr[k * N + n];
            }
          }
        }
        __syncthreads();
        BLAS().execute(sA, sB, accumulator, transformOp, transformOp);
        __syncthreads();
      }
#else
      // CUDA path: original CuTe-based implementation with cp.async pipelining
      const auto gA = tile::get<bM, bK, ar>(a, M, K, cute::select<0, 2>(tileCoord)); //  M, K
      const auto gB = tile::get<bK, bN, br>(b, K, N, cute::select<2, 1>(tileCoord)); // K, N
      constexpr auto sASS = flashmoe_blas::cosize(BLAS::suggest_layout_smem_a());
      constexpr auto sBSS = flashmoe_blas::cosize(BLAS::suggest_layout_smem_b());
      auto* __restrict__ sAP = static_cast<Element*>(workspace);
      auto* __restrict__ sBP = sAP + (sASS * pipeStages);
      auto sA = flashmoe_blas::make_tensor(sAP, BLAS::suggest_layout_smem_a());
      auto sB = flashmoe_blas::make_tensor(sBP, BLAS::suggest_layout_smem_b());
      // prime pipeline
      cute::for_each(cute::make_int_sequence<pipeStages>{}, [&](auto stage) {
        update_buffer<stage, sASS>(sA, sAP);
        update_buffer<stage, sBSS>(sB, sBP);
        flashmoe_blas::copy<BLAS, aAlignment>(gA(cute::_, cute::_, stage), sA);
        flashmoe_blas::copy<BLAS, bAlignment>(gB(cute::_, cute::_, stage), sB);
        cpFence();
      });
      // mainloop
      for (int kStage = pipeStages; kStage < tilesK; ++kStage) {
        const int ps = kStage % pipeStages;
        update_buffer<sASS>(sA, sAP, ps);
        update_buffer<sBSS>(sB, sBP, ps);
        cpWait<pipeStages - 1>();
        BLAS().execute(sA, sB, accumulator, transformOp, transformOp);
        __syncthreads();
        flashmoe_blas::copy<BLAS, aAlignment>(gA(cute::_, cute::_, kStage), sA);
        flashmoe_blas::copy<BLAS, bAlignment>(gB(cute::_, cute::_, kStage), sB);
        cpFence();
      }
      // tail
      cute::for_each(cute::make_int_sequence<pipeStages>{}, [&](auto stage) {
        const int ps = (tilesK + stage) % pipeStages;
        update_buffer<sASS>(sA, sAP, ps);
        update_buffer<sBSS>(sB, sBP, ps);
        cpWait<(pipeStages - 1) - stage>();
        BLAS().execute(sA, sB, accumulator, transformOp, transformOp);
      });
#endif
    }
  };
}
#endif //FLASHMOE_TILE_CUH
