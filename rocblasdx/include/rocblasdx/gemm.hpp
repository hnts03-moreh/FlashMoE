/*
 * Copyright (c) 2025, FlashMoE ROCm Porting Project
 * All rights reserved.
 * BSD 3-Clause License — see LICENSE in the project root.
 *
 * rocBLASDx — Device-side GEMM library for AMD MI300X (gfx942)
 * gemm.hpp — BLAS descriptor composition, device-side tile GEMM,
 *            copy utilities, accumulator management, tensor views
 */

#ifndef ROCBLASDX_GEMM_HPP
#define ROCBLASDX_GEMM_HPP

#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

#include "types.hpp"
#include "layout.hpp"
#include "fragment.hpp"
#include "mfma.hpp"

namespace rocblasdx {

// =====================================================================
// Forward declarations for API compatibility traits
// =====================================================================
template <typename BLAS> struct size_of;
template <typename BLAS> struct alignment_of;
template <typename BLAS> constexpr arrangement arrangement_of_v_c = BLAS::c_arrangement;
template <typename BLAS> constexpr bool is_blas_execution_v = BLAS::is_execution;

// =====================================================================
// cosize — total element count needed for a SmemLayout
// =====================================================================
template <int R, int C, int S>
__host__ __device__ __forceinline__
constexpr int cosize(const SmemLayout<R, C, S>&) {
    return R * S;  // rows * stride (stride >= cols, includes padding)
}

// =====================================================================
// SmemTensor — lightweight 2D view over shared memory
// =====================================================================
template <typename Element, int Rows, int Cols, int Stride>
struct SmemTensor {
    Element* ptr;

    __device__ __forceinline__
    SmemTensor(Element* p) : ptr(p) {}

    // 2D indexing
    __device__ __forceinline__
    Element& operator()(int row, int col) { return ptr[row * Stride + col]; }

    __device__ __forceinline__
    const Element& operator()(int row, int col) const { return ptr[row * Stride + col]; }

    // Flat indexing (for copy utilities)
    __device__ __forceinline__
    Element& operator()(int flat) { return ptr[flat]; }

    __device__ __forceinline__
    const Element& operator()(int flat) const { return ptr[flat]; }

    __device__ __forceinline__
    Element*& data() { return ptr; }

    static constexpr int size() { return Rows * Cols; }

    using layout_type = SmemLayout<Rows, Cols, Stride>;
    static constexpr layout_type layout() { return {}; }
};

// =====================================================================
// make_tensor — SmemLayout version
// =====================================================================
template <typename Element, int R, int C, int S>
__device__ __forceinline__
auto make_tensor(Element* ptr, SmemLayout<R, C, S>) {
    return SmemTensor<Element, R, C, S>{ptr};
}

// size() overload for SmemTensor — enables rocblasdx::size(tensor)
template <typename Element, int R, int C, int S>
__device__ __host__ __forceinline__
constexpr int size(const SmemTensor<Element, R, C, S>&) {
    return R * C;
}

// Generic size() for types with .size() member
template <typename T>
__device__ __host__ __forceinline__
auto size(const T& t) -> decltype(t.size()) { return t.size(); }

// CuTe layout passthrough overloads — only available when math_compat.h (cute:: namespace) is included
#if defined(FLASHMOE_USE_MATH_COMPAT)
// make_tensor — raw pointer + CuTe Layout -> wrap in smem_ptr and forward
template <typename Element, typename CuteLayout>
__device__ __forceinline__
auto make_tensor(Element* ptr, CuteLayout layout)
    -> decltype(cute::make_tensor(cute::make_smem_ptr(ptr), layout))
{
    return cute::make_tensor(cute::make_smem_ptr(ptr), layout);
}

// make_tensor — cute::smem_ptr passthrough
template <typename SmemPtr, typename CuteLayout>
__device__ __forceinline__
auto make_tensor(SmemPtr ptr, CuteLayout layout)
    -> decltype(cute::make_tensor(ptr, layout))
{
    return cute::make_tensor(ptr, layout);
}
#endif

// =====================================================================
// copy_wait — synchronize outstanding copies
// =====================================================================
__device__ __forceinline__
void copy_wait() {
    __syncthreads();
}

// =====================================================================
// BlasExecution — the resolved BLAS type with all parameters
// =====================================================================
template <
    int M_, int N_, int K_,
    typename ElementA_, typename ElementB_, typename ElementC_,
    arrangement ArA_, arrangement ArB_, arrangement ArC_,
    int AlignA_, int AlignB_, int AlignC_,
    int Threads_,
    int Arch_
>
struct BlasExecution {
    // ---- Traits ----
    static constexpr bool is_execution = true;

    // ---- Tile dimensions ----
    static constexpr int tile_m = M_;
    static constexpr int tile_n = N_;
    static constexpr int tile_k = K_;

    // ---- Element types ----
    using element_a = ElementA_;
    using element_b = ElementB_;
    using element_c = ElementC_;  // accumulator type

    // ---- Arrangement ----
    static constexpr arrangement a_arrangement = ArA_;
    static constexpr arrangement b_arrangement = ArB_;
    static constexpr arrangement c_arrangement = ArC_;

    // ---- Alignment ----
    static constexpr int a_alignment = AlignA_;
    static constexpr int b_alignment = AlignB_;
    static constexpr int c_alignment = AlignC_;

    // ---- Threading ----
    static constexpr int max_threads_per_block = Threads_;
    static constexpr int arch = Arch_;

    // ---- MFMA selection ----
    using MfmaInstr = mfma::select_mfma_t<ElementA_, M_, N_>;
    static constexpr int mfma_m = MfmaInstr::M;
    static constexpr int mfma_n = MfmaInstr::N;
    static constexpr int mfma_k = MfmaInstr::K;
    static constexpr int mfma_tiles_m = M_ / mfma_m;
    static constexpr int mfma_tiles_n = N_ / mfma_n;
    static constexpr int mfma_iters_k = K_ / mfma_k;

    // ---- Accumulator ----
    static constexpr int wavefront_size = 64;
    static constexpr int num_waves = Threads_ / wavefront_size;
    static constexpr int total_mfma_tiles = mfma_tiles_m * mfma_tiles_n;
    // Tiles per wave: distribute evenly, rounding up
    static constexpr int tiles_per_wave = (total_mfma_tiles + num_waves - 1) / num_waves;
    static constexpr int accum_per_thread = tiles_per_wave * MfmaInstr::c_per_thread;

    // ---- Shared memory layouts ----
    using SmemLayoutA = typename SuggestSmemLayout<ArA_, M_, K_, ElementA_>::type;
    using SmemLayoutB = typename SuggestSmemLayout<ArB_, K_, N_, ElementB_>::type;
    using SmemLayoutC = typename SuggestSmemLayout<ArC_, M_, N_, element_c>::type;

    static constexpr auto suggest_layout_smem_a() { return SmemLayoutA{}; }
    static constexpr auto suggest_layout_smem_b() { return SmemLayoutB{}; }
    static constexpr auto suggest_layout_smem_c() { return SmemLayoutC{}; }
    static constexpr auto get_layout_smem_c()     { return SmemLayoutC{}; }

    // ---- suggest_accumulator ----
    __device__ __forceinline__
    static auto suggest_accumulator() {
        Accumulator<element_c, accum_per_thread> acc;
        acc.clear();
        return acc;
    }

    // ---- execute: tile GEMM C += op_a(A) * op_b(B) from shared memory ----
    template <typename SmemA, typename SmemB, typename Accum, typename TransformA, typename TransformB>
    __device__ __forceinline__
    void execute(const SmemA& sA, const SmemB& sB, Accum& accumulator,
                 TransformA transformA, TransformB transformB) const {
#if defined(__gfx9__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
        execute_mfma(sA, sB, accumulator, transformA, transformB);
#else
        execute_scalar(sA, sB, accumulator, transformA, transformB);
#endif
    }

    template <typename SmemA, typename SmemB, typename Accum>
    __device__ __forceinline__
    void execute(const SmemA& sA, const SmemB& sB, Accum& accumulator) const {
        identity id;
        execute(sA, sB, accumulator, id, id);
    }

private:
    // ---- Shared memory element access ----
    template <typename SmemA>
    __device__ __forceinline__
    static element_a load_a(const SmemA& sA, int row, int col) {
        if constexpr (ArA_ == row_major)
            return sA.ptr[row * SmemLayoutA::stride + col];
        else
            return sA.ptr[col * SmemLayoutA::stride + row];
    }

    template <typename SmemB>
    __device__ __forceinline__
    static element_b load_b(const SmemB& sB, int row, int col) {
        if constexpr (ArB_ == col_major)
            return sB.ptr[col * SmemLayoutB::stride + row];
        else
            return sB.ptr[row * SmemLayoutB::stride + col];
    }

    // ---- Scalar fallback (host-side testing / non-gfx9 targets) ----
    template <typename SmemA, typename SmemB, typename Accum, typename TA, typename TB>
    __device__ __forceinline__
    void execute_scalar(const SmemA& sA, const SmemB& sB, Accum& accumulator,
                        TA tA, TB tB) const {
        const int tid = threadIdx.x;
        constexpr int total = M_ * N_;
        constexpr int per_thread = (total + Threads_ - 1) / Threads_;

        for (int idx = 0; idx < per_thread; ++idx) {
            int flat = tid + idx * Threads_;
            if (flat >= total) break;
            int i = (ArC_ == row_major) ? flat / N_ : flat % M_;
            int j = (ArC_ == row_major) ? flat % N_ : flat / M_;

            element_c sum{0};
            for (int kk = 0; kk < K_; ++kk) {
                sum += static_cast<element_c>(tA(load_a(sA, i, kk))) *
                       static_cast<element_c>(tB(load_b(sB, kk, j)));
            }
            if (idx < accum_per_thread)
                accumulator(idx) += sum;
        }
    }

    // ---- MFMA implementation ----
    template <typename SmemA, typename SmemB, typename Accum, typename TA, typename TB>
    __device__ __forceinline__
    void execute_mfma(const SmemA& sA, const SmemB& sB, Accum& accumulator,
                      TA transformA, TB transformB) const {
        const int lane_id = threadIdx.x % wavefront_size;
        const int wave_id = threadIdx.x / wavefront_size;

        // Each wave processes a subset of MFMA sub-tiles
        for (int t = wave_id; t < total_mfma_tiles; t += num_waves) {
            const int tm = t / mfma_tiles_n;
            const int tn = t % mfma_tiles_n;
            const int m_base = tm * mfma_m;
            const int n_base = tn * mfma_n;

            // accum_offset for this tile in this wave
            int tile_local_idx = t / num_waves;
            if (t % num_waves < wave_id) {} // correction not needed for striped
            // Simple: map tile t to accum position
            int accum_base = (t / num_waves) * MfmaInstr::c_per_thread;
            if (t % num_waves != wave_id) continue;  // shouldn't happen in stride loop
            // Correct mapping: t maps to (t - wave_id) / num_waves-th slot
            accum_base = ((t - wave_id) / num_waves) * MfmaInstr::c_per_thread;

            // Load accumulator into MFMA vector
            typename MfmaInstr::CVec c_vec;
            #pragma unroll
            for (int i = 0; i < MfmaInstr::c_per_thread; ++i) {
                reinterpret_cast<element_c*>(&c_vec)[i] = accumulator(accum_base + i);
            }

            // Iterate along K
            for (int ki = 0; ki < mfma_iters_k; ++ki) {
                const int k_base = ki * mfma_k;
                run_one_mfma(sA, sB, c_vec, lane_id, m_base, n_base, k_base,
                             transformA, transformB);
            }

            // Write back
            #pragma unroll
            for (int i = 0; i < MfmaInstr::c_per_thread; ++i) {
                accumulator(accum_base + i) = reinterpret_cast<element_c*>(&c_vec)[i];
            }
        }
    }

    // ---- Single MFMA instruction dispatch ----
    template <typename SmemA, typename SmemB, typename TA, typename TB>
    __device__ __forceinline__
    void run_one_mfma(const SmemA& sA, const SmemB& sB,
                      typename MfmaInstr::CVec& c_vec,
                      int lane_id, int m_base, int n_base, int k_base,
                      TA tA, TB tB) const {
        run_one_mfma_dispatch(sA, sB, c_vec, lane_id, m_base, n_base, k_base, tA, tB,
            std::bool_constant<std::is_same_v<element_a, __half> ||
                               std::is_same_v<element_a, hip_bfloat16>>{});
    }

    // Half / BF16 path: pack 4 elements into vector registers
    template <typename SmemA, typename SmemB, typename TA, typename TB>
    __device__ __forceinline__
    void run_one_mfma_dispatch(const SmemA& sA, const SmemB& sB,
                               typename MfmaInstr::CVec& c_vec,
                               int lane_id, int m_base, int n_base, int k_base,
                               TA tA, TB tB, std::true_type) const {
        typename MfmaInstr::AVec a_vec;
        typename MfmaInstr::BVec b_vec;

        const int tpg = mfma_m;  // threads per group: 32 for 32x32, 16 for 16x16
        const int group = lane_id / tpg;
        const int tid = lane_id % tpg;
        const int k_off = group * 4;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            reinterpret_cast<element_a*>(&a_vec)[j] = tA(load_a(sA, m_base + tid, k_base + k_off + j));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            reinterpret_cast<element_b*>(&b_vec)[j] = tB(load_b(sB, k_base + k_off + j, n_base + tid));
        }

        MfmaInstr::run(c_vec, a_vec, b_vec);
    }

    // FP32 / FP64 path: scalar input
    template <typename SmemA, typename SmemB, typename TA, typename TB>
    __device__ __forceinline__
    void run_one_mfma_dispatch(const SmemA& sA, const SmemB& sB,
                               typename MfmaInstr::CVec& c_vec,
                               int lane_id, int m_base, int n_base, int k_base,
                               TA tA, TB tB, std::false_type) const {
        const int tpg = mfma_m;
        const int tid = lane_id % tpg;

        // FP32 MFMA: mfma_f32_32x32x2f32 takes scalar a, scalar b
        // Internally it reads from different lanes to get K=2 values.
        // We provide one scalar per lane. The instruction does K steps internally.
        typename MfmaInstr::AVec a_val = static_cast<typename MfmaInstr::AVec>(
            tA(load_a(sA, m_base + tid, k_base)));
        typename MfmaInstr::BVec b_val = static_cast<typename MfmaInstr::BVec>(
            tB(load_b(sB, k_base, n_base + tid)));

        MfmaInstr::run(c_vec, a_val, b_val);
    }
};

// =====================================================================
// size_of — extract tile dimensions
// =====================================================================
template <typename BLAS>
struct size_of {
    static constexpr int m = BLAS::tile_m;
    static constexpr int n = BLAS::tile_n;
    static constexpr int k = BLAS::tile_k;
};

// =====================================================================
// alignment_of — extract alignment
// =====================================================================
template <typename BLAS>
struct alignment_of {
    static constexpr int a = BLAS::a_alignment;
    static constexpr int b = BLAS::b_alignment;
    static constexpr int c = BLAS::c_alignment;
};

// =====================================================================
// Descriptor composition via operator+
// =====================================================================
// cuBLASDx builds a BLAS type via:
//   decltype(Size<M,N,K>() + Precision<A,B,C>() + Type<real>() + ... + SM<Arch,mod>())
//
// We replicate this pattern with a BlasBag that accumulates parameters.

namespace detail {

template <
    int M = 0, int N = 0, int K = 0,
    typename ElemA = void, typename ElemB = void, typename ElemC = void,
    arrangement ArA = row_major, arrangement ArB = col_major, arrangement ArC = row_major,
    int AlignA = 16, int AlignB = 16, int AlignC = 16,
    int Threads = 0,
    int Arch = 0
>
struct BlasBag {
    // ---- Stored parameters ----
    static constexpr int m = M;
    static constexpr int n = N;
    static constexpr int k = K;
    using elem_a = ElemA;
    using elem_b = ElemB;
    using elem_c = ElemC;
    static constexpr arrangement ar_a = ArA;
    static constexpr arrangement ar_b = ArB;
    static constexpr arrangement ar_c = ArC;
    static constexpr int align_a = AlignA;
    static constexpr int align_b = AlignB;
    static constexpr int align_c = AlignC;
    static constexpr int threads = Threads;
    static constexpr int arch = Arch;

    // ---- Completeness check ----
    static constexpr bool is_complete = !std::is_void_v<ElemA> && !std::is_void_v<ElemB>
                                        && !std::is_void_v<ElemC> && M > 0 && N > 0 && K > 0;

    // ---- Safe element type for sizeof (avoids sizeof(void)) ----
    using SafeElemA = std::conditional_t<std::is_void_v<ElemA>, char, ElemA>;
    using SafeElemB = std::conditional_t<std::is_void_v<ElemB>, char, ElemB>;
    using SafeElemC = std::conditional_t<std::is_void_v<ElemC>, char, ElemC>;
    static constexpr int SafeM = (M > 0) ? M : 1;
    static constexpr int SafeN = (N > 0) ? N : 1;
    static constexpr int SafeK = (K > 0) ? K : 1;

    // ---- Resolve to BlasExecution ----
    using resolved = BlasExecution<SafeM, SafeN, SafeK, SafeElemA, SafeElemB, SafeElemC,
                                   ArA, ArB, ArC,
                                   AlignA, AlignB, AlignC,
                                   (Threads > 0 ? Threads : 256), Arch>;

    // ---- Forward all BlasExecution members ----
    static constexpr bool is_execution = true;
    static constexpr int tile_m = M;
    static constexpr int tile_n = N;
    static constexpr int tile_k = K;
    using element_a = ElemA;
    using element_b = ElemB;
    using element_c = ElemC;
    static constexpr arrangement a_arrangement = ArA;
    static constexpr arrangement b_arrangement = ArB;
    static constexpr arrangement c_arrangement = ArC;
    static constexpr int a_alignment = AlignA;
    static constexpr int b_alignment = AlignB;
    static constexpr int c_alignment = AlignC;
    static constexpr int max_threads_per_block = (Threads > 0 ? Threads : 256);

    // SmemLayouts (use safe types to avoid sizeof(void))
    using SmemLayoutA = typename SuggestSmemLayout<ArA, SafeM, SafeK, SafeElemA>::type;
    using SmemLayoutB = typename SuggestSmemLayout<ArB, SafeK, SafeN, SafeElemB>::type;
    using SmemLayoutC = typename SuggestSmemLayout<ArC, SafeM, SafeN, SafeElemC>::type;

    static constexpr auto suggest_layout_smem_a() { return SmemLayoutA{}; }
    static constexpr auto suggest_layout_smem_b() { return SmemLayoutB{}; }
    static constexpr auto suggest_layout_smem_c() { return SmemLayoutC{}; }
    static constexpr auto get_layout_smem_c()     { return SmemLayoutC{}; }

    // suggest_accumulator
    __device__ __forceinline__
    static auto suggest_accumulator() {
        return resolved::suggest_accumulator();
    }

    // execute — forward to resolved type
    template <typename... Args>
    __device__ __forceinline__
    void execute(Args&&... args) const {
        resolved{}.execute(static_cast<Args&&>(args)...);
    }

    __host__ __device__ constexpr BlasBag() = default;
};

} // namespace detail

// ---- operator+ overloads ----
// cuBLASDx uses left-to-right associativity:
//   ((((Size() + Precision()) + Type()) + Function()) + ... + SM())
// So after the first two descriptors produce a BlasBag, all subsequent
// additions are: BlasBag + Descriptor -> BlasBag

// Initial: Size<M,N,K> + Precision<A,B,C> -> BlasBag (the first two descriptors)
template <int M, int N, int K, typename PA, typename PB, typename PC>
__host__ __device__ constexpr auto operator+(Size<M, N, K>, Precision<PA, PB, PC>) {
    return detail::BlasBag<M, N, K, PA, PB, PC>{};
}

// bag + Type<V> -> no-op (we only support real)
template <int M, int N, int K, typename A, typename B, typename C,
          arrangement ArA, arrangement ArB, arrangement ArC,
          int AlA, int AlB, int AlC, int T, int Ar,
          type::value V>
__host__ __device__ constexpr auto operator+(
    detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>,
    Type<V>) {
    return detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>{};
}

// bag + Function<V> -> no-op (we only support MM)
template <int M, int N, int K, typename A, typename B, typename C,
          arrangement ArA, arrangement ArB, arrangement ArC,
          int AlA, int AlB, int AlC, int T, int Ar,
          function::value V>
__host__ __device__ constexpr auto operator+(
    detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>,
    Function<V>) {
    return detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>{};
}

// bag + Arrangement<a,b,c> -> set arrangements
template <int M, int N, int K, typename A, typename B, typename C,
          arrangement ArA, arrangement ArB, arrangement ArC,
          int AlA, int AlB, int AlC, int T, int Ar,
          arrangement nArA, arrangement nArB, arrangement nArC>
__host__ __device__ constexpr auto operator+(
    detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>,
    Arrangement<nArA, nArB, nArC>) {
    return detail::BlasBag<M, N, K, A, B, C, nArA, nArB, nArC, AlA, AlB, AlC, T, Ar>{};
}

// bag + Block -> no-op (marker)
template <int M, int N, int K, typename A, typename B, typename C,
          arrangement ArA, arrangement ArB, arrangement ArC,
          int AlA, int AlB, int AlC, int T, int Ar>
__host__ __device__ constexpr auto operator+(
    detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>,
    Block) {
    return detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>{};
}

// bag + Alignment<a,b,c> -> set alignments
template <int M, int N, int K, typename A, typename B, typename C,
          arrangement ArA, arrangement ArB, arrangement ArC,
          int AlA, int AlB, int AlC, int T, int Ar,
          int nAlA, int nAlB, int nAlC>
__host__ __device__ constexpr auto operator+(
    detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>,
    Alignment<nAlA, nAlB, nAlC>) {
    return detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, nAlA, nAlB, nAlC, T, Ar>{};
}

// bag + BlockDim<threads> -> set thread count
template <int M, int N, int K, typename A, typename B, typename C,
          arrangement ArA, arrangement ArB, arrangement ArC,
          int AlA, int AlB, int AlC, int T, int Ar,
          int nT>
__host__ __device__ constexpr auto operator+(
    detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>,
    BlockDim<nT>) {
    return detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, nT, Ar>{};
}

// bag + StaticBlockDim -> no-op (marker)
template <int M, int N, int K, typename A, typename B, typename C,
          arrangement ArA, arrangement ArB, arrangement ArC,
          int AlA, int AlB, int AlC, int T, int Ar>
__host__ __device__ constexpr auto operator+(
    detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>,
    StaticBlockDim) {
    return detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>{};
}

// bag + EnableInputStreaming -> no-op (marker)
template <int M, int N, int K, typename A, typename B, typename C,
          arrangement ArA, arrangement ArB, arrangement ArC,
          int AlA, int AlB, int AlC, int T, int Ar>
__host__ __device__ constexpr auto operator+(
    detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>,
    EnableInputStreaming) {
    return detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>{};
}

// bag + SM<Arch,Mod> -> set arch
template <int M, int N, int K, typename A, typename B, typename C,
          arrangement ArA, arrangement ArB, arrangement ArC,
          int AlA, int AlB, int AlC, int T, int Ar,
          int nAr, sm_modifier Mod>
__host__ __device__ constexpr auto operator+(
    detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, Ar>,
    SM<nAr, Mod>) {
    return detail::BlasBag<M, N, K, A, B, C, ArA, ArB, ArC, AlA, AlB, AlC, T, nAr>{};
}

// =====================================================================
// copy — cooperative smem <-> gmem copy (BLAS-parameterized)
// =====================================================================
// For SmemTensor types
template <typename BLAS, int Align, typename SrcElement, int R, int C, int S,
          typename DstElement, int R2, int C2, int S2>
__device__ __forceinline__
void copy(const SmemTensor<SrcElement, R, C, S>& src,
          SmemTensor<DstElement, R2, C2, S2>& dst) {
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

// For CuTe tensor types — forward to cooperative copy pattern
template <typename BLAS, int Align, typename SrcTensor, typename DstTensor>
__device__ __forceinline__
void copy(const SrcTensor& src, DstTensor& dst) {
    // Cooperative thread-strided copy
    constexpr int threads = BLAS::max_threads_per_block;
    const int tid = threadIdx.x;
    const int total = rocblasdx::size(src);
    const int per = (total + threads - 1) / threads;
    for (int i = 0; i < per; ++i) {
        int flat = tid + i * threads;
        if (flat < total)
            dst(flat) = src(flat);
    }
}

// Thread-count copy (used in combine.cuh): copy<threads, align>(tid, src, dst)
template <int Threads, int Align, typename SrcTensor, typename DstTensor>
__device__ __forceinline__
void copy(int tid, const SrcTensor& src, DstTensor& dst) {
    const int total = rocblasdx::size(src);
    for (int flat = tid; flat < total; flat += Threads) {
        dst(flat) = src(flat);
    }
}

// =====================================================================
// copy_fragment — fragment <-> smem/gmem copy (MFMA-layout-aware)
// =====================================================================
//
// On MFMA-capable architectures (gfx9xx), the accumulator fragment is
// stored in MFMA register layout, NOT simple striped layout.
// Each thread's fragment elements map to specific (row, col) positions
// determined by the MFMA instruction's output register distribution.
//
// Fragment element i maps to:
//   tile_local = i / c_per_thread (which MFMA sub-tile within this wave)
//   vgpr       = i % c_per_thread (which register within that sub-tile)
//   tile_global = wave_id + tile_local * num_waves (striped tile assignment)
//   tm = tile_global / mfma_tiles_n
//   tn = tile_global % mfma_tiles_n
//   row = tm * mfma_m + MfmaLayout::row(lane_id, vgpr)
//   col = tn * mfma_n + MfmaLayout::col(lane_id)

namespace detail {

// Resolve BLAS type: BlasBag → BlasExecution via ::resolved, or pass through
template <typename T, typename = void>
struct resolve_blas { using type = T; };
template <typename T>
struct resolve_blas<T, std::void_t<typename T::resolved>> { using type = typename T::resolved; };
template <typename T>
using resolve_blas_t = typename resolve_blas<T>::type;

// Helper: given BLAS parameters, compute the (row, col) for fragment element i
// Returns true if the position is valid (within the tile), false for inactive threads/tiles
template <typename BLAS_>
__device__ __forceinline__
bool mfma_frag_to_rc(int i, int& row, int& col) {
    using BLAS = resolve_blas_t<BLAS_>;
    using MfmaInstr = typename BLAS::MfmaInstr;
    using Layout = mfma::MfmaOutputLayout<MfmaInstr>;
    constexpr int c_per_thread = MfmaInstr::c_per_thread;
    constexpr int mfma_m = MfmaInstr::M;
    constexpr int mfma_n = MfmaInstr::N;
    constexpr int mfma_tiles_m = BLAS::mfma_tiles_m;
    constexpr int mfma_tiles_n = BLAS::mfma_tiles_n;
    constexpr int total_mfma_tiles = mfma_tiles_m * mfma_tiles_n;
    constexpr int num_waves = BLAS::num_waves;
    constexpr int wavefront_size = 64;

    const int lane_id = threadIdx.x % wavefront_size;
    const int wave_id = threadIdx.x / wavefront_size;

    int tile_local = i / c_per_thread;
    int vgpr = i % c_per_thread;
    int tile_global = wave_id + tile_local * num_waves;

    // Check if this tile exists (some waves may have fewer tiles)
    if (tile_global >= total_mfma_tiles) {
        row = -1; col = -1;
        return false;
    }

    int tm = tile_global / mfma_tiles_n;
    int tn = tile_global % mfma_tiles_n;

    row = tm * mfma_m + Layout::row(lane_id, vgpr);
    col = tn * mfma_n + Layout::col(lane_id);
    return true;
}

// Check if BLAS type uses MFMA (has MfmaInstr member)
template <typename T, typename = void>
struct has_mfma : std::false_type {};
template <typename T>
struct has_mfma<T, std::void_t<typename T::MfmaInstr>> : std::true_type {};

// Check if a type is a BlasExecution (for SFINAE)
template <typename T>
concept IsBlasExecution = requires { typename T::MfmaInstr; T::mfma_tiles_n; T::num_waves; };
} // namespace detail

// -------------------------------------------------------------------
// Fragment -> SmemTensor (rmem -> smem) — MFMA-aware
// -------------------------------------------------------------------
// The accumulator carries BLAS type info via its size which encodes
// the MFMA tiling. We use a BLAS tag overload for MFMA-aware copy.

// MFMA-aware overload: uses BLAS type to compute correct (row, col)
template <int Align, typename BLAS, typename FT, int FN, typename ST, int R, int C, int S>
__device__ __forceinline__
void copy_fragment_mfma(const Fragment<FT, FN>& frag,
                        SmemTensor<ST, R, C, S>& smem) {
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int row, col;
        detail::mfma_frag_to_rc<BLAS>(i, row, col);
        if (row < R && col < C)
            smem(row, col) = static_cast<ST>(frag(i));
    }
}

// MFMA-aware overload: SmemTensor -> Fragment
template <int Align, typename BLAS, typename ST, int R, int C, int S, typename FT, int FN>
__device__ __forceinline__
void copy_fragment_mfma(const SmemTensor<ST, R, C, S>& smem,
                        Fragment<FT, FN>& frag) {
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int row, col;
        detail::mfma_frag_to_rc<BLAS>(i, row, col);
        if (row < R && col < C)
            frag(i) = static_cast<FT>(smem(row, col));
    }
}

// MFMA-aware overload: CuTe/GmemTileView Tensor -> Fragment
template <int Align, typename BLAS, typename SrcTensor, typename FT, int FN>
__device__ __forceinline__
void copy_fragment_mfma(const SrcTensor& src, Fragment<FT, FN>& frag) {
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int row, col;
        if (detail::mfma_frag_to_rc<BLAS>(i, row, col))
            frag(i) = static_cast<FT>(src(row, col));
    }
}

// MFMA-aware overload: Fragment -> CuTe/GmemTileView Tensor
template <int Align, typename BLAS, typename FT, int FN, typename DstTensor>
__device__ __forceinline__
void copy_fragment_mfma(const Fragment<FT, FN>& frag, DstTensor& dst) {
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int row, col;
        if (detail::mfma_frag_to_rc<BLAS>(i, row, col))
            dst(row, col) = static_cast<std::decay_t<decltype(dst(0, 0))>>(frag(i));
    }
}

// -------------------------------------------------------------------
// Unified copy_fragment — auto-detects MFMA on gfx9 architectures
// -------------------------------------------------------------------
// On MFMA-capable HW, uses MFMA register layout mapping.
// The BLAS type is derived from the accumulator (which is a Fragment
// whose size encodes the tiling). The third parameter (accumulator)
// carries the layout info implicitly.
//
// Strategy: We derive the MFMA parameters from the SmemTensor dimensions
// (R=tile_m, C=tile_n) and the thread count. This avoids changing call sites.

namespace detail {
// Select MFMA instruction from element type and tile dims
template <typename Element, int TileM, int TileN>
using auto_mfma_t = mfma::select_mfma_t<Element, TileM, TileN>;

// Auto-derive MFMA tile counts and wave count
template <typename MfmaInstr, int TileM, int TileN, int Threads>
struct MfmaParams {
    static constexpr int mfma_m = MfmaInstr::M;
    static constexpr int mfma_n = MfmaInstr::N;
    static constexpr int c_per_thread = MfmaInstr::c_per_thread;
    static constexpr int mfma_tiles_m = TileM / mfma_m;
    static constexpr int mfma_tiles_n = TileN / mfma_n;
    static constexpr int num_waves = Threads / 64;
};

// Compute (row, col) from fragment index using MFMA layout
// TileM, TileN = overall tile dims; returns false for invalid positions
template <typename Element, int TileM, int TileN>
__device__ __forceinline__
bool auto_mfma_frag_to_rc(int i, int& row, int& col) {
    using MfmaInstr = auto_mfma_t<Element, TileM, TileN>;
    using Layout = mfma::MfmaOutputLayout<MfmaInstr>;
    constexpr int c_per_thread = MfmaInstr::c_per_thread;
    constexpr int mfma_m = MfmaInstr::M;
    constexpr int mfma_n = MfmaInstr::N;
    constexpr int mfma_tiles_m = TileM / mfma_m;
    constexpr int mfma_tiles_n = TileN / mfma_n;
    constexpr int total_mfma_tiles = mfma_tiles_m * mfma_tiles_n;

    const int lane_id = threadIdx.x % 64;
    const int wave_id = threadIdx.x / 64;
    const int num_waves = static_cast<int>(blockDim.x) / 64;

    int tile_local = i / c_per_thread;
    int vgpr = i % c_per_thread;
    int tile_global = wave_id + tile_local * num_waves;

    if (tile_global >= total_mfma_tiles) {
        row = -1; col = -1;
        return false;
    }

    int tm = tile_global / mfma_tiles_n;
    int tn = tile_global % mfma_tiles_n;

    row = tm * mfma_m + Layout::row(lane_id, vgpr);
    col = tn * mfma_n + Layout::col(lane_id);
    return true;
}
} // namespace detail

// Fragment -> SmemTensor (rmem -> smem)
template <int Align, typename FT, int FN, typename ST, int R, int C, int S>
__device__ __forceinline__
void copy_fragment(const Fragment<FT, FN>& frag,
                   SmemTensor<ST, R, C, S>& smem,
                   const auto& /* accumulator */) {
#if defined(__gfx9__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    // MFMA path: use hardware register layout
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int row, col;
        detail::auto_mfma_frag_to_rc<ST, R, C>(i, row, col);
        if (row < R && col < C)
            smem(row, col) = static_cast<ST>(frag(i));
    }
#else
    // Scalar path: striped layout
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    constexpr int total = R * C;
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int flat = tid + i * threads;
        if (flat < total) {
            smem(flat / C, flat % C) = static_cast<ST>(frag(i));
        }
    }
#endif
}

// SmemTensor -> Fragment (smem -> rmem)
template <int Align, typename ST, int R, int C, int S, typename FT, int FN>
__device__ __forceinline__
void copy_fragment(const SmemTensor<ST, R, C, S>& smem,
                   Fragment<FT, FN>& frag,
                   const auto& /* accumulator */) {
#if defined(__gfx9__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int row, col;
        detail::auto_mfma_frag_to_rc<ST, R, C>(i, row, col);
        if (row < R && col < C)
            frag(i) = static_cast<FT>(smem(row, col));
    }
#else
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    constexpr int total = R * C;
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int flat = tid + i * threads;
        if (flat < total) {
            frag(i) = static_cast<FT>(smem(flat / C, flat % C));
        }
    }
#endif
}

// Generic Tensor -> Fragment (gmem/smem -> rmem) — uses flat/striped indexing
template <int Align, typename SrcTensor, typename FT, int FN>
__device__ __forceinline__
void copy_fragment(const SrcTensor& src, Fragment<FT, FN>& frag, const auto&)
    requires(!std::is_same_v<std::decay_t<SrcTensor>, Fragment<FT, FN>>)
{
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int flat = tid + i * threads;
        frag(i) = static_cast<FT>(src(flat));
    }
}

// Generic Fragment -> Tensor (rmem -> smem/gmem) — uses flat/striped indexing
template <int Align, typename FT, int FN, typename DstTensor>
__device__ __forceinline__
void copy_fragment(const Fragment<FT, FN>& frag, DstTensor& dst, const auto&)
    requires(!std::is_same_v<std::decay_t<DstTensor>, Fragment<FT, FN>>)
{
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    #pragma unroll
    for (int i = 0; i < FN; ++i) {
        int flat = tid + i * threads;
        dst(flat) = static_cast<std::decay_t<decltype(dst(0))>>(frag(i));
    }
}

// =====================================================================
// Convenience: suggest_threads (standalone)
// =====================================================================
template <int M, int N, int K, typename Element>
constexpr int suggest_threads() {
    using Mfma = mfma::select_mfma_t<Element, M, N>;
    constexpr int tiles = (M / Mfma::M) * (N / Mfma::N);
    constexpr int t = tiles * 64;
    return t < 64 ? 64 : (t > 1024 ? 1024 : t);
}

} // namespace rocblasdx

#endif // ROCBLASDX_GEMM_HPP
