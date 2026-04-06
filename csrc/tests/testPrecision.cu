//
// Phase 3 precision validation: FP16, BF16, FP32 MFMA correctness test.
// Tests multiple tile sizes with Identity activation to isolate MFMA accuracy.
// Outputs structured results for automated analysis.
//
// Usage: ./testPrecision [seed]
//   Defaults: seed=42
//
// Output format: CSV with columns:
//   precision, M, N, K, bM, bN, bK, threads, error_pct, max_abs_err, status
//

#include <random>
#include <cstring>
#include <cmath>

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/tile.cuh"

#if !defined(FLASHMOE_PLATFORM_HIP)
#error "testPrecision is a HIP-only test"
#endif

#include "../include/flashmoe/infra/activation.cuh"
#include "host_reference.cuh"

struct IdentityAct {
    template<typename T>
    __device__ __forceinline__ T operator()(const T& x) const { return x; }
};

template<typename TileGEMM, typename Activation, typename ElementC, typename Element>
__device__ __forceinline__
void gemmMainloop(void* __restrict__ const& workspace,
    const Element* __restrict__ const& a,
    const Element* __restrict__ const& b,
    ElementC* __restrict__ const& c,
    const ElementC* __restrict__ const& bias,
    const int& M, const int& N, const int& K, const int& tileIdx) {
    using BLAS = TileGEMM::BLAS;
    auto accumulator = BLAS::suggest_accumulator();
    using BM = cute::Int<flashmoe_blas::size_of<BLAS>::m>;
    using BN = cute::Int<flashmoe_blas::size_of<BLAS>::n>;
    const auto tileCoord = flashmoe::tile::idx2Coord(M / BM{}, N / BN{}, tileIdx);
    constexpr TileGEMM tileMainloop{};
    tileMainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
    const auto gD = flashmoe::tile::getBias<BM{}, BN{}>(bias, M, N, cute::select<0, 1>(tileCoord));
    auto d_frag = flashmoe_blas::make_fragment_like<ElementC>(accumulator.get_results());
    flashmoe_blas::copy_fragment_mfma<flashmoe_blas::alignment_of<BLAS>::c, BLAS>(gD, d_frag);
    constexpr Activation act{};
    using AccumType = decltype(accumulator)::value_type;
    constexpr flashmoe::Converter<AccumType, ElementC> loadConv{};
    constexpr flashmoe::Converter<ElementC, AccumType> storeConv{};
    const auto c_frag = accumulator.get_results();
    constexpr int accum_size = flashmoe_blas::size(c_frag);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&](auto i) {
        d_frag(i) = storeConv(act(c_frag(i) + loadConv(d_frag(i))));
    });
    auto gC = flashmoe::tile::getC<BM{}, BN{}, flashmoe_blas::arrangement_of_v_c<BLAS>>(c, M, N,
        cute::select<0, 1>(tileCoord));
    flashmoe_blas::copy_fragment_mfma<flashmoe_blas::alignment_of<BLAS>::c, BLAS>(d_frag, gC);
}

#define SC(T, v) static_cast<T>(v)

template<typename TileGEMM, typename Activation, typename Element, typename ElementC>
requires(flashmoe_blas::is_blas_execution_v<typename TileGEMM::BLAS>)
__launch_bounds__(TileGEMM::BLAS::max_threads_per_block, 1)
__global__ void gk(const Element* __restrict__ a, const Element* __restrict__ b,
    ElementC* __restrict__ c, const ElementC* __restrict__ bias,
    const __grid_constant__ int M, const __grid_constant__ int N, const int __grid_constant__ K) {
    using BLAS = TileGEMM::BLAS;
    constexpr int bM = flashmoe_blas::size_of<BLAS>::m;
    constexpr int bN = flashmoe_blas::size_of<BLAS>::n;
    const int nTiles = (M / bM) * (N / bN);
    extern __shared__ __align__(TileGEMM::GeneralAlignment::value) cuda::std::byte gemmWorkspace[];
    for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; tileIdx += SC(int, gridDim.x)) {
        gemmMainloop<TileGEMM, Activation>(gemmWorkspace, a, b, c, bias, M, N, K, tileIdx);
    }
}

// Run a single GEMM test case, return (error_pct, max_abs_err)
template<int bM, int bN, int bK, int pipeStages, typename AccumType, typename Element, typename ElementC>
__host__ __forceinline__
std::pair<double, float> run_test(
    const int M, const int N, const int K, const long seed,
    const char* prec_label, const float rtol, const float atol) {
    Element* a = nullptr;
    Element* b = nullptr;
    ElementC* c = nullptr;
    ElementC* bias = nullptr;
    gpuStream_t stream;
    gpuStreamCreate(&stream);

    CHECK_CUDA(gpuMallocAsync(&a, M * K * sizeof(Element), stream));
    CHECK_CUDA(gpuMallocAsync(&b, N * K * sizeof(Element), stream));
    CHECK_CUDA(gpuMallocAsync(&c, M * N * sizeof(ElementC), stream));
    CHECK_CUDA(gpuMallocAsync(&bias, N * sizeof(ElementC), stream));

    using Act = IdentityAct;
    constexpr int threads = flashmoe::tile::suggest_thread_count<bM, bN, bK, FLASHMOE_ARCH, Element, AccumType>();
    using TileGEMM = flashmoe::tile::CollectiveMainloop<
            bM, bN, bK, FLASHMOE_ARCH, Element, AccumType, threads, pipeStages
    >;
    auto kernel = gk<TileGEMM, Act, Element, ElementC>;
    int bps = 0;
    constexpr auto sharedSize = TileGEMM::SharedSizeAB::value;
    // Only request extended shared memory if > 48KB default
    if constexpr (sharedSize > 49152) {
        CHECK_CUDA(gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, sharedSize));
    }
    CHECK_CUDA(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, sharedSize));
    const int blocks = cute::min((M / bM) * (N / bN), bps * NUM_SMS);

    constexpr auto min_v = -1.f;
    constexpr auto max_v = 1.f;
    randUniform<FLASHMOE_ARCH>(a, M * K, seed, min_v, max_v, stream);
    randUniform<FLASHMOE_ARCH>(b, N * K, seed + 1, min_v, max_v, stream);
    randUniform<FLASHMOE_ARCH>(bias, N, seed + 2, min_v, max_v, stream);

    gk<TileGEMM, Act><<<blocks, threads, sharedSize, stream>>>(a, b, c, bias, M, N, K);
    CHECK_CUDA(gpuStreamSynchronize(stream));
    CHECK_CUDA(gpuPeekAtLastError());

    // Copy to host
    const size_t sz_a = static_cast<size_t>(M) * K;
    const size_t sz_b = static_cast<size_t>(N) * K;
    const size_t sz_c = static_cast<size_t>(M) * N;
    std::vector<Element> h_a(sz_a), h_b(sz_b);
    std::vector<ElementC> h_bias(N), h_c(sz_c), h_c_ref(sz_c);
    CHECK_CUDA(hipMemcpy(h_a.data(), a, sz_a * sizeof(Element), hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(h_b.data(), b, sz_b * sizeof(Element), hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(h_bias.data(), bias, N * sizeof(ElementC), hipMemcpyDeviceToHost));
    CHECK_CUDA(hipMemcpy(h_c.data(), c, sz_c * sizeof(ElementC), hipMemcpyDeviceToHost));

    // CPU reference: Identity(A @ B.T + bias)
    host_ref::gemm_bias_act<host_ref::HostIdentity>(h_a.data(), h_b.data(), h_bias.data(), h_c_ref.data(), M, N, K);

    // Compute error stats
    long matches = 0;
    float max_abs_err = 0.f;
    for (size_t i = 0; i < sz_c; ++i) {
        float act_v = host_ref::to_float(h_c[i]);
        float exp_v = host_ref::to_float(h_c_ref[i]);
        float diff = std::fabs(act_v - exp_v);
        if (diff <= atol + rtol * std::fabs(exp_v)) ++matches;
        if (diff > max_abs_err) max_abs_err = diff;
    }
    double error_pct = (1.0 - static_cast<double>(matches) / static_cast<double>(sz_c)) * 100.0;

    const char* status = (error_pct <= 0.01) ? "PASS" : "FAIL";
    printf("%s, %d, %d, %d, %d, %d, %d, %d, %.4f, %.6e, %s\n",
           prec_label, M, N, K, bM, bN, bK, threads, error_pct, max_abs_err, status);

    gpuFreeAsync(a, stream);
    gpuFreeAsync(b, stream);
    gpuFreeAsync(c, stream);
    gpuFreeAsync(bias, stream);
    gpuStreamSynchronize(stream);
    gpuStreamDestroy(stream);

    return {error_pct, max_abs_err};
}

// Test sweep for FP16/BF16 (bK=64, sizeof(Element)=2 → fits in LDS)
template<typename Element, typename ElementC, typename AccumType>
void run_precision_suite_half(const char* label, const long seed, const float rtol, const float atol) {
    int pass = 0, fail = 0;

    auto check = [&](auto result) {
        if (result.first <= 0.01) ++pass; else ++fail;
    };

    // Small tiles
    check(run_test<16, 16, 16, 1, AccumType, Element, ElementC>(16, 16, 16, seed, label, rtol, atol));
    check(run_test<32, 32, 32, 1, AccumType, Element, ElementC>(32, 32, 32, seed, label, rtol, atol));

    // Standard tiles (64x64)
    check(run_test<64, 64, 64, 1, AccumType, Element, ElementC>(64, 64, 64, seed, label, rtol, atol));
    check(run_test<64, 64, 64, 1, AccumType, Element, ElementC>(128, 128, 64, seed, label, rtol, atol));

    // Large tiles (128x64, 128x128)
    check(run_test<128, 64, 64, 1, AccumType, Element, ElementC>(128, 64, 64, seed, label, rtol, atol));
    check(run_test<128, 128, 64, 1, AccumType, Element, ElementC>(128, 128, 64, seed, label, rtol, atol));

    // Non-square
    check(run_test<64, 64, 64, 1, AccumType, Element, ElementC>(256, 128, 64, seed, label, rtol, atol));
    check(run_test<128, 64, 64, 1, AccumType, Element, ElementC>(256, 128, 128, seed, label, rtol, atol));

    // Larger K
    check(run_test<64, 64, 64, 1, AccumType, Element, ElementC>(128, 128, 256, seed, label, rtol, atol));

    // Production-like
    check(run_test<128, 128, 64, 1, AccumType, Element, ElementC>(512, 256, 128, seed, label, rtol, atol));

    printf("# %s summary: %d PASS, %d FAIL (total %d)\n\n", label, pass, fail, pass + fail);
}

// Test sweep for FP32 (bK=32 to fit LDS: float is 4B, so halve K-tile vs FP16)
// MI300X LDS = 64KB. FP32 bM=128,bN=128,bK=32 → ~37KB (fits).
template<typename AccumType>
void run_precision_suite_fp32(const char* label, const long seed, const float rtol, const float atol) {
    using Element = float;
    using ElementC = float;
    int pass = 0, fail = 0;

    auto check = [&](auto result) {
        if (result.first <= 0.01) ++pass; else ++fail;
    };

    // Small tiles
    check(run_test<16, 16, 16, 1, AccumType, Element, ElementC>(16, 16, 16, seed, label, rtol, atol));
    check(run_test<32, 32, 32, 1, AccumType, Element, ElementC>(32, 32, 32, seed, label, rtol, atol));

    // Standard tiles (64x64, bK=32 for FP32 LDS budget)
    check(run_test<64, 64, 32, 1, AccumType, Element, ElementC>(64, 64, 32, seed, label, rtol, atol));
    check(run_test<64, 64, 32, 1, AccumType, Element, ElementC>(128, 128, 64, seed, label, rtol, atol));

    // Large tiles (bK=32)
    check(run_test<128, 64, 32, 1, AccumType, Element, ElementC>(128, 64, 64, seed, label, rtol, atol));
    check(run_test<128, 128, 32, 1, AccumType, Element, ElementC>(128, 128, 64, seed, label, rtol, atol));

    // Non-square
    check(run_test<64, 64, 32, 1, AccumType, Element, ElementC>(256, 128, 64, seed, label, rtol, atol));
    check(run_test<128, 64, 32, 1, AccumType, Element, ElementC>(256, 128, 128, seed, label, rtol, atol));

    // Larger K (still bK=32, more K-stages)
    check(run_test<64, 64, 32, 1, AccumType, Element, ElementC>(128, 128, 256, seed, label, rtol, atol));

    // Production-like
    check(run_test<128, 128, 32, 1, AccumType, Element, ElementC>(512, 256, 128, seed, label, rtol, atol));

    printf("# %s summary: %d PASS, %d FAIL (total %d)\n\n", label, pass, fail, pass + fail);
}

int main(const int argc, char** argv) {
    long seed = 42;
    if (argc > 1) seed = std::stol(argv[1]);

    gpuSetDevice(0);

    printf("############################################################\n");
    printf("# testPrecision: FP16/BF16/FP32 MFMA accuracy validation\n");
    printf("# seed=%ld, GPU=MI300X (gfx942)\n", seed);
    printf("############################################################\n\n");

    printf("precision, M, N, K, bM, bN, bK, threads, error_pct, max_abs_err, status\n");

    // FP16: rtol=2e-2, atol=2e-3
    run_precision_suite_half<__half, __half, float>("fp16", seed, 2e-2f, 2e-3f);

    // BF16: rtol=2e-2, atol=2e-3 (same tolerance as FP16 — BF16 has less mantissa)
    run_precision_suite_half<__nv_bfloat16, __nv_bfloat16, float>("bf16", seed, 2e-2f, 2e-3f);

    // FP32: rtol=1e-4, atol=1e-5, bK=32 (LDS budget: float=4B, halve K-tile)
    // MFMA FMA ordering differs from sequential CPU reference; ~1e-6 rounding expected
    run_precision_suite_fp32<float>("fp32", seed, 1e-4f, 1e-5f);

    printf("############################################################\n");
    printf("# Done. Results saved above.\n");
    printf("############################################################\n");
}
