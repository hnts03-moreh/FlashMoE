//
// Diagnostic test for 128+ tile MFMA multi-tile correctness.
// Compares bM=64 (known good, tiles_per_wave=1) vs bM=128 (broken, tiles_per_wave=2)
// with Identity activation to isolate the MFMA copy_fragment issue.
//
// Usage: ./diagGEMM [M] [N] [K] [seed]
//   Defaults: M=256, N=128, K=64, seed=42
//

#include <random>
#include <cstring>

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/tile.cuh"

#if defined(FLASHMOE_PLATFORM_HIP)
#include "../include/flashmoe/infra/activation.cuh"
#include "host_reference.cuh"
#else
#error "diagGEMM is a HIP-only diagnostic test"
#endif

// Identity activation — no nonlinearity to confound the comparison
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

template<int bM, int bN, int bK, int pipeStages, typename AccumType, typename Element, typename ElementC>
__host__ __forceinline__
void driver(const int& M, const int& N, const int& K, const long seed, const char* label) {
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
    CHECK_CUDA(gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, sharedSize));
    CHECK_CUDA(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, sharedSize));
    const int blocks = cute::min((M / bM) * (N / bN), bps * NUM_SMS);

    // Deterministic init with fixed seed
    constexpr auto min_v = -1.f;
    constexpr auto max_v = 1.f;
    randUniform<FLASHMOE_ARCH>(a, M * K, seed, min_v, max_v, stream);
    randUniform<FLASHMOE_ARCH>(b, N * K, seed + 1, min_v, max_v, stream);
    randUniform<FLASHMOE_ARCH>(bias, N, seed + 2, min_v, max_v, stream);

    // Run kernel
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

    // CPU reference: Act(A @ B.T + bias) with Identity
    host_ref::gemm_bias_act<host_ref::HostIdentity>(h_a.data(), h_b.data(), h_bias.data(), h_c_ref.data(), M, N, K);

    printf("=== %s: bM=%d, bN=%d, bK=%d, threads=%d, M=%d, N=%d, K=%d ===\n",
           label, bM, bN, bK, threads, M, N, K);
    printf("    tiles: %dx%d = %d, blocks=%d, sharedSize=%d\n",
           M / bM, N / bN, (M / bM) * (N / bN), blocks, sharedSize);

    const float rtol = 2e-2f;
    const float atol = 2e-3f;
    host_ref::compare_isclose_verbose(h_c.data(), h_c_ref.data(), M, N, rtol, atol, bM, bN, 30);

    gpuFreeAsync(a, stream);
    gpuFreeAsync(b, stream);
    gpuFreeAsync(c, stream);
    gpuFreeAsync(bias, stream);
    gpuStreamSynchronize(stream);
    gpuStreamDestroy(stream);
}

int main(const int argc, char** argv) {
    int M = 256;
    int N = 128;
    int K = 64;
    long seed = 42;

    if (argc > 1) M = std::stoi(argv[1]);
    if (argc > 2) N = std::stoi(argv[2]);
    if (argc > 3) K = std::stoi(argv[3]);
    if (argc > 4) seed = std::stol(argv[4]);

    gpuSetDevice(0);

    using Element = __half;
    using ElementC = Element;
    using MMA_C = float;

    printf("############################################################\n");
    printf("# diagGEMM: 128+ tile MFMA diagnostic (Identity activation)\n");
    printf("# M=%d, N=%d, K=%d, seed=%ld\n", M, N, K, seed);
    printf("############################################################\n\n");

    // Test 1: bM=64, bN=64 (known good — tiles_per_wave=1)
    if (M >= 64 && N >= 64) {
        driver<64, 64, 64, 1, MMA_C, Element, ElementC>(M, N, K, seed, "CONTROL-64x64");
    }

    // Test 2: bM=128, bN=64 (broken — tiles_per_wave=2 with 256 threads)
    if (M >= 128 && N >= 64) {
        driver<128, 64, 64, 1, MMA_C, Element, ElementC>(M, N, K, seed, "TEST-128x64");
    }

    // Test 3: bM=64, bN=128 (check if bN=128 also fails)
    if (M >= 64 && N >= 128) {
        driver<64, 128, 64, 1, MMA_C, Element, ElementC>(M, N, K, seed, "TEST-64x128");
    }

    // Test 4: bM=128, bN=128 (both large)
    if (M >= 128 && N >= 128) {
        driver<128, 128, 64, 1, MMA_C, Element, ElementC>(M, N, K, seed, "TEST-128x128");
    }

    // Test 5: bM=32, bN=32 (small, baseline)
    if (M >= 32 && N >= 32) {
        driver<32, 32, 32, 1, MMA_C, Element, ElementC>(M, N, K, seed, "CONTROL-32x32");
    }

    printf("\n############################################################\n");
    printf("# Done. If CONTROL tests show 0%% error but TEST tests show >0%%,\n");
    printf("# the bug is in multi-tile MFMA (tiles_per_wave > 1).\n");
    printf("# Check per-tile breakdown to see which tiles are affected.\n");
    printf("############################################################\n");
}
