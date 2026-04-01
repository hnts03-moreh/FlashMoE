//
// Created by osayamen on 1/14/26.
//
// unit tests for combine
#include <stdexcept>

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/combine.cuh"

// Kernel Under Test (KUT)
template<int Arch, int bM, int bN, int threads, flashmoe::CombineMode c, typename Element>
__launch_bounds__(threads, 1)
__global__ void combineKernel(const __grid_constant__ size_t EC,
    const __grid_constant__ size_t S,
    const __grid_constant__ uint E,
    const __grid_constant__ uint H,
    const int* __restrict__ expertCounts,
    const Element* __restrict__ tokens,
    Element* __restrict__ output,
    const flashmoe::TPS* __restrict__ tokenIndices
    ) {
    constexpr int Alignment = flashmoe::ElementAlignment<Element, bN>;
    extern __shared__ __align__(Alignment) cuda::std::byte combineWorkspace[];
    const uint tilesM = EC / bM;
    const uint tilesN = H / bN;
    const auto tilesPerExpert = tilesM * tilesN;
    const auto numTiles = E * tilesM * tilesN;
    const auto tokTensor = cute::make_tensor(cute::make_gmem_ptr(tokens),
        cute::make_layout(cute::make_shape(E, EC, static_cast<size_t>(H)), cute::LayoutRight{}));
    const auto tokenIds = cute::make_tensor(cute::make_gmem_ptr(tokenIndices),
        cute::make_layout(cute::make_shape(E, EC), cute::LayoutRight{}));
    for (int globalIdx = blockIdx.x; globalIdx < numTiles; globalIdx += gridDim.x) {
        const auto expertIdx = globalIdx / tilesPerExpert;
        const auto tileIdx = globalIdx % tilesPerExpert;
        const auto coord = flashmoe::tile::idx2Coord(tilesM, tilesN, tileIdx);
        const auto tileCoord = cute::make_coord(cute::_0{}, cute::get<1>(coord));
        const auto tileM = cute::get<0>(coord);
        const auto expertCount  = expertCounts[expertIdx];
        const auto actualTiles = cute::ceil_div(expertCount, bM) * tilesN;
        if (tileIdx < actualTiles) {
            const auto numFullMTiles = expertCount / bM;
            auto* __restrict__ tP = &tokTensor(static_cast<size_t>(expertIdx), tileM * bM, 0);
            const auto tileSize = tileM < numFullMTiles ? bM : (expertCount - numFullMTiles * bM);
            const auto* __restrict__ tI = &tokenIds(expertIdx, tileM * bM);
            flashmoe::combine<bM, bN, Arch, threads, c>(S, H, combineWorkspace, tI, output, tP, tileSize, tileCoord);
        }
    }
}

template<typename AccumType, typename Element>
__global__ void combineReference(const __grid_constant__ int E, const __grid_constant__ int S,
    const __grid_constant__ int H, const __grid_constant__ size_t EC, const __grid_constant__ int topK,
    const Element* __restrict__ tokens,
    const flashmoe::TPS* __restrict__ tokenIds,
    const int* __restrict__ expertCounts,
    Element* __restrict__ result,
    float* __restrict__ oracleResult
    ) {
    static_assert(cuda::std::is_same_v<AccumType, Element> || cuda::std::is_same_v<AccumType, float>);
    if (blockIdx.x > 0) return;
    const auto tIds = cute::make_tensor(cute::make_gmem_ptr(tokenIds),
        cute::make_layout(cute::make_shape(E, EC), cute::LayoutRight{}));
    const auto tokTensor = cute::make_tensor(cute::make_gmem_ptr(tokens),
        cute::make_layout(cute::make_shape(E, EC, static_cast<size_t>(H)), cute::LayoutRight{}));
    auto resultTensor = cute::make_tensor(cute::make_gmem_ptr(result),
        cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
    auto oracleT = cute::make_tensor(cute::make_gmem_ptr(oracleResult),
        cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
    for (int i = 0; i < E; ++i) {
        const auto expertCount = expertCounts[i];
        for (int j = 0; j < expertCount; ++j) {
            const auto tokenId = tIds(i, j);
            if (topK == 1) {
                for (auto k = threadIdx.x; k < H; k += blockDim.x) {
                    resultTensor(tokenId.tokenIdx, k) = tokTensor(i, j, k);
                }
            }
            else {
                constexpr flashmoe::Converter<float, Element> loadOp{};
                constexpr flashmoe::Converter<AccumType, Element> convOp1{};
                constexpr flashmoe::Converter<AccumType, float> convOp2{};
                constexpr flashmoe::Converter<Element, AccumType> storeOp{};
                for (auto k = threadIdx.x; k < H; k += blockDim.x) {
                    const auto v = loadOp(tokTensor(i, j, k));
                    const float scaledV = v * tokenId.probability;
                    oracleT(tokenId.tokenIdx, k) += scaledV;
                    const auto rv = resultTensor(tokenId.tokenIdx, k);
                    const auto c = convOp1(rv);
                    const AccumType res = c + convOp2(scaledV);
                    resultTensor(tokenId.tokenIdx, k) = storeOp(res);
                }
            }
        }
    }
}

template<int Arch, int H, int bM, int bN, typename Element>
__host__ __forceinline__
void kickStart(const int& S, const int& E, const int& k, const float& rtol, const float& atol) {
    const size_t EC = S;
    const size_t roundEC = cute::ceil_div(EC, bM) * bM;
    const auto [counts, indices] = generate_token_ids_and_expert_counts(S, E, EC, roundEC,
        k, 0.f, 0.001f);
    CHECK_CUDA(gpuSetDevice(0));
    gpuStream_t stream;
    CHECK_CUDA(gpuStreamCreate(&stream));
    constexpr int sharedSize = bM * bN * sizeof(Element);
    int maxSharedMemory = 0;
    CHECK_CUDA(gpuDeviceGetAttribute(&maxSharedMemory, gpuDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (sharedSize > maxSharedMemory) {
        printf("Insufficient shared memory for bM: %d, bN: %d -> %d should be <= %d\n",
            bM, bN, sharedSize, maxSharedMemory);
        gpuStreamDestroy(stream);
        return;
    }
    Element* tokens = nullptr;
    flashmoe::TPS* tIds = nullptr;
    int* expertCounts = nullptr;
    Element* kut_result = nullptr;
    Element* ref_result = nullptr;
    float* oracleResult = nullptr;

    CHECK_CUDA(gpuMallocAsync(&tokens, sizeof(Element) * E * EC * H, stream));
    CHECK_CUDA(gpuMallocAsync(&tIds, sizeof(flashmoe::TPS) * E * EC, stream));
    CHECK_CUDA(gpuMallocAsync(&expertCounts, sizeof(int) * E, stream));
    CHECK_CUDA(gpuMallocAsync(&ref_result, sizeof(Element) * S * H, stream));
    CHECK_CUDA(gpuMemsetAsync(ref_result, 0, sizeof(Element) * S * H, stream));

    std::random_device rd;
    CHECK_CUDA(gpuMallocAsync(&kut_result, sizeof(Element) * S * H, stream));
    if (k > 1) {
        CHECK_CUDA(gpuMemsetAsync(kut_result, 0, sizeof(Element) * S * H, stream));
        CHECK_CUDA(gpuMallocAsync(&oracleResult, sizeof(float) * S * H, stream));
        CHECK_CUDA(gpuMemsetAsync(oracleResult, 0, sizeof(float) * S * H, stream));
    }
    CHECK_CUDA(gpuPeekAtLastError());
    randUniform<Arch>(tokens, static_cast<size_t>(E * EC) * H, rd(), -1.0f, 1.0f, stream);
    CHECK_CUDA(gpuMemcpyAsync(expertCounts, counts.data(), sizeof(int) * E, gpuMemcpyHostToDevice, stream));
    CHECK_CUDA(gpuMemcpyAsync(tIds, indices.data(), sizeof(flashmoe::TPS) * E * EC, gpuMemcpyHostToDevice, stream));
    using AccumType = Element;
    constexpr int threads = cute::max(cute::min(H, 1024), 32);
    combineReference<AccumType><<<1, threads, 0, stream>>>(E, S, H, EC, k, tokens, tIds, expertCounts, ref_result, oracleResult);
    constexpr int cThreads = cute::max(cute::min(bM, 128), 32);
    int bps = 0;
    int blocks = 0;
    if (k > 1) {
        auto kernel = combineKernel<Arch, bM, bN, cThreads, flashmoe::CombineMode::plural, Element>;
        CHECK_CUDA(gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, sharedSize));
        CHECK_CUDA(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, cThreads, sharedSize));
        blocks = cute::min(((E * EC) / bM) * (H / bN), bps * NUM_SMS);
        kernel<<<blocks, cThreads, sharedSize, stream>>>(EC, S, E, H, expertCounts, tokens, kut_result, tIds);
    }
    else {
        auto kernel = combineKernel<Arch, bM, bN, cThreads, flashmoe::CombineMode::single, Element>;
        CHECK_CUDA(gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, sharedSize));
        CHECK_CUDA(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, cThreads, sharedSize));
        blocks = cute::min(((E * EC) / bM) * (H / bN), bps * NUM_SMS);
        kernel<<<blocks, cThreads, sharedSize, stream>>>(EC, S, E, H, expertCounts, tokens, kut_result, tIds);
    }

#if !defined(FLASHMOE_PLATFORM_HIP)
    // MatX comparison — CUDA only
    using MatXType = MXE<Element>;
    auto tKUT = matx::make_tensor<MatXType>(reinterpret_cast<MatXType*>(kut_result), {S, H});
    auto tRef = matx::make_tensor<MatXType>(reinterpret_cast<MatXType*>(ref_result), {S, H});
    matx::cudaExecutor exec{stream};
    auto num_matches_ref = matx::make_tensor<long int>({});
    auto num_matches_oracle = matx::make_tensor<long int>({});
    (num_matches_ref = matx::sum(matx::isclose(tKUT, tRef, rtol, atol))).run(exec);
    if (k > 1) {
        auto tOracle = matx::make_tensor<float>(oracleResult, {S, H});
        (num_matches_oracle = matx::sum(matx::isclose(tKUT, tOracle, rtol, atol))).run(exec);
    }
    exec.sync();
    const auto error_ref = (1.0 - (static_cast<double>(num_matches_ref()) / static_cast<double>(tKUT.TotalSize())))*100.0;
    double error_oracle = -0.0;
    if (k > 1) {
        error_oracle = (1.0 - (static_cast<double>(num_matches_oracle()) / static_cast<double>(tKUT.TotalSize())))*100.0;
    }
    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %lf\n",
        S, H, E, k, bM, bN, cThreads, bps, NUM_SMS, blocks, rtol, atol, error_ref, error_oracle);
#else
    // HIP: no MatX, just run and check no errors
    CHECK_CUDA(gpuStreamSynchronize(stream));
    CHECK_CUDA(gpuPeekAtLastError());
    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, ran(no_matx), ran(no_matx)\n",
        S, H, E, k, bM, bN, cThreads, bps, NUM_SMS, blocks, rtol, atol);
#endif

    CHECK_CUDA(gpuPeekAtLastError());
    CHECK_CUDA(gpuFreeAsync(tokens, stream));
    CHECK_CUDA(gpuFreeAsync(tIds, stream));
    CHECK_CUDA(gpuFreeAsync(expertCounts, stream));
    CHECK_CUDA(gpuFreeAsync(ref_result, stream));
    if (oracleResult) CHECK_CUDA(gpuFreeAsync(oracleResult, stream));
    CHECK_CUDA(gpuFreeAsync(kut_result, stream));
    CHECK_CUDA(gpuStreamSynchronize(stream));
    CHECK_CUDA(gpuStreamDestroy(stream));
}
// ./testCombine <S> <E> <topK> <atol> <rtol>
__host__ __forceinline__
void doTest(const int argc, char** argv) {
    int S = 8192; int E = 16; int k = 2;
    float rtol = 2e-2f; float atol = 2e-3f;
    if (argc > 1) S = std::stoi(argv[1]);
    if (argc > 2) E = std::stoi(argv[2]);
    if (argc > 3) k = std::stoi(argv[3]);
    if (argc > 4) rtol = std::stof(argv[4]);
    if (argc > 5) atol = std::stof(argv[5]);
    if (k > E) throw std::runtime_error("k must be at most E");
    if (!cutlass::ispow2(S)) throw std::runtime_error("S must be a power of 2");
    using Element = __half;
    constexpr int H = 2048;
    constexpr int bN = cute::min(H, 64 * (sizeof(Element) == 2 ? 2 : 1));
    static_assert(H % bN == 0);
    constexpr int Arch = FLASHMOE_ARCH;
    printf("S,H,E,k,bM,bN,threads,blocks/SM,SMs,blocks,rtol,atol, error_ref(%%),error_oracle(%%)\n");
    switch (S) {
    case 1:  kickStart<Arch, H, 1, bN, Element>(S, E, k, rtol, atol); break;
    case 2:  kickStart<Arch, H, 2, bN, Element>(S, E, k, rtol, atol); break;
    case 4:  kickStart<Arch, H, 4, bN, Element>(S, E, k, rtol, atol); break;
    case 8:  kickStart<Arch, H, 8, bN, Element>(S, E, k, rtol, atol); break;
    case 16: kickStart<Arch, H, 16, bN, Element>(S, E, k, rtol, atol); break;
    case 32: kickStart<Arch, H, 32, bN, Element>(S, E, k, rtol, atol); break;
    case 64: kickStart<Arch, H, 64, bN, Element>(S, E, k, rtol, atol); break;
    default:
        if (S >= 128) kickStart<Arch, H, 128, bN, Element>(S, E, k, rtol, atol);
    }
}
int main(const int argc, char** argv) {
    doTest(argc, argv);
}
