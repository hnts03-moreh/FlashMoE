//
// Created by osayamen on 12/28/25.
//
// Benchmark and unit tests for the fused gate
#include <random>
#include <tuple>
#include <vector>
#include <cstdio>
#include <string>

#include "../include/flashmoe/platform/profiling.h"

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/gate.cuh"

#if !defined(FLASHMOE_PLATFORM_HIP)
// MatX-based reference — CUDA only

struct IndexSanitizer {
  const matx::index_t y;

  explicit IndexSanitizer(const matx::index_t& _y) : y(_y) {
  }

  __host__ __device__
  auto operator()(const matx::index_t& x) const {
    return x / y;
  }
};

struct SplitFunctor {
  __host__ __device__
  auto operator()(const flashmoe::TPS& t) const {
    return t.tokenIdx;
  }
};

struct __align__(16) GateArgs {
  void* tokens;
  void* gateWeights;
  void* routing;
  void* routing_ref;
  void* routing_interim;
  flashmoe::TPS* tokenIds_packed;
  uint* tokenIds;
  matx::index_t* tokenIds_ref;
  int* eCounts;
  int* eCGuards;
  matx::index_t* eCounts_ref;
  matx::index_t* topK;
  flashmoe::SoftmaxStatePacked* rSp;
  flashmoe::RingTopKPayload* rTp;
  int S;
  int H;
  int E;
  int k;
  int EC;
  int roundEC;
  float rtol;
  float atol;
};

template <int warmup, int runs, typename RE, typename REC>
__host__ __forceinline__
auto reference(matx::cudaExecutor& exec, const GateArgs& gArgs, cudaEvent_t start, cudaEvent_t stop) {
  using Element = MXE<RE>;
  using ElementC = MXE<REC>;
  auto tA = matx::make_tensor<Element>(static_cast<Element*>(gArgs.tokens),
                                       {gArgs.S, gArgs.H});
  auto tB = matx::make_tensor<Element>(static_cast<Element*>(gArgs.gateWeights),
                                       {gArgs.E, gArgs.H});
  auto tC = matx::make_tensor<ElementC>(static_cast<ElementC*>(gArgs.routing_ref),
                                        {gArgs.S, gArgs.E});
  auto tC_interim = matx::make_tensor<flashmoe::gate::SoftType>(
    static_cast<flashmoe::gate::SoftType*>(gArgs.routing_interim), {gArgs.S, gArgs.E});
  auto tCx = matx::make_tensor<ElementC>(static_cast<ElementC*>(gArgs.routing),
                                         {gArgs.S, gArgs.E});

  auto tokenIds = matx::make_tensor<matx::index_t>(gArgs.tokenIds_ref,
                                                   {gArgs.E, gArgs.S});
  auto eCounts = matx::make_tensor<matx::index_t>(gArgs.eCounts_ref,
                                                  {gArgs.E});
  auto tokenIds_packed = matx::make_tensor<flashmoe::TPS>(gArgs.tokenIds_packed,
                                                          {gArgs.E, gArgs.roundEC});
  auto tokenIds_x = matx::make_tensor<uint>(gArgs.tokenIds, {gArgs.E, gArgs.roundEC});
  (tokenIds_x = matx::apply(SplitFunctor{}, tokenIds_packed)).run(exec);
  auto eCounts_x = matx::make_tensor<int>(gArgs.eCounts, {gArgs.E});

  auto gemm_n_matches = matx::make_tensor<long int>({});
  auto ec_matches = matx::make_tensor<int>({gArgs.E});
  auto s_ec_matches = matx::make_tensor<int>({});
  auto tIds_matches = matx::make_tensor<int>({gArgs.E});
  auto s_tIds_matches = matx::make_tensor<int>({});
  auto st_x = matx::make_tensor<uint>({gArgs.roundEC});
  auto st = matx::make_tensor<matx::index_t>({gArgs.roundEC});
  (tIds_matches = matx::zeros<int>(tIds_matches.Shape())).run(exec);

  (tC_interim = matx::matmul(tA, tB.PermuteMatrix())).run(exec);
  (tC = matx::apply(Converter<ElementC, flashmoe::gate::SoftType>{},
                    matx::softmax(tC_interim, {1}))).run(exec);
  matx::cudaExecutor exec1{exec.getStream()};
  (gemm_n_matches = matx::sum(matx::isclose(tCx, tC, gArgs.rtol, gArgs.atol))).run(exec1);
  auto sIndices = matx::make_tensor<matx::index_t>(gArgs.topK, {gArgs.S, gArgs.E});
  (sIndices = matx::argsort(tCx, matx::SORT_DIR_DESC)).run(exec);
  auto topK_idx = sIndices.Slice<2>({0, 0}, {matx::matxEnd, gArgs.k});
  for (int i = 0; i < gArgs.E; ++i) {
    auto tIdx_row = tokenIds.Slice<1>({i, 0}, {matx::matxDropDim, matx::matxEnd});
    auto eC = eCounts.Slice<0>({i}, {matx::matxDropDim});
    (matx::mtie(tIdx_row, eC) = matx::find_idx(topK_idx, matx::EQ<matx::index_t>{i})).run(exec);
    (tIdx_row = matx::apply(IndexSanitizer{gArgs.k}, tIdx_row)).run(exec);
  }
  exec.sync();
  (ec_matches = (eCounts_x == eCounts)).run(exec);
  (s_ec_matches = matx::sum(ec_matches)).run(exec);
  std::vector<matx::index_t> hec(eCounts.TotalSize());
  cudaMemcpyAsync(hec.data(), gArgs.eCounts_ref, sizeof(matx::index_t) * eCounts.TotalSize(),
                  cudaMemcpyDeviceToHost, exec.getStream());
  exec.sync();
  matx::index_t totalRoutedTokens = 0;
  for (int i = 0; i < gArgs.E; ++i) {
    if (ec_matches(i)) {
      const auto eCount = hec[i];
      totalRoutedTokens += eCount;
      auto tIdsRow_x = tokenIds_x.Slice<1>({i, 0}, {matx::matxDropDim, eCount});
      auto st_x_s = st_x.Slice<1>({0}, {eCount});
      (st_x_s = matx::sort(tIdsRow_x, matx::SORT_DIR_DESC)).run(exec);
      auto tIdsRow = tokenIds.Slice<1>({i, 0}, {matx::matxDropDim, eCount});
      auto st_s = st.Slice<1>({0}, {eCount});
      (st_s = matx::sort(tIdsRow, matx::SORT_DIR_DESC)).run(exec);
      auto tIm_r = tIds_matches.Slice<0>({i}, {matx::matxDropDim});
      exec.sync();
      (tIm_r = matx::sum(st_x_s == st_s)).run(exec);
    }
  }
  (s_tIds_matches = matx::sum(tIds_matches)).run(exec);
  CHECK_CUDA(cudaPeekAtLastError());
  exec.sync();
  const auto ep_gs = (1.0 - (static_cast<double>(gemm_n_matches()) /
    static_cast<double>(tC.TotalSize()))) * 100;
  const auto ep_ec = (1.0 - (static_cast<double>(s_ec_matches()) /
    static_cast<double>(eCounts.TotalSize()))) * 100;
  const auto ep_tIds = (1.0 - (static_cast<double>(s_tIds_matches()) /
    static_cast<double>(totalRoutedTokens))) * 100;

  auto gate_via_matx = [&]() {
    flashmoe::flashmoeRange matxRange{"Gate"};
    (tC = matx::softmax(matx::matmul(tA, tB.PermuteMatrix()), {1})).run(exec);
    (sIndices = matx::argsort(tCx, matx::SORT_DIR_DESC)).run(exec);
    for (int i = 0; i < gArgs.E; ++i) {
      auto tIdx_row = tokenIds.Slice<1>({i, 0}, {matx::matxDropDim, matx::matxEnd});
      auto eC = eCounts.Slice<0>({i}, {matx::matxDropDim});
      (matx::mtie(tIdx_row, eC) = matx::find_idx(topK_idx, matx::EQ<matx::index_t>{i})).run(exec);
      (tIdx_row = matx::apply(IndexSanitizer{gArgs.k}, tIdx_row)).run(exec);
    }
  };
  for (int i = 0; i < warmup; ++i) gate_via_matx();
  exec.sync();
  cudaEventRecord(start, exec.getStream());
  for (int i = 0; i < runs; ++i) gate_via_matx();
  cudaEventRecord(stop, exec.getStream());
  CHECK_CUDA(cudaEventSynchronize(stop));
  float m_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&m_ms, start, stop));
  const float m_time_ms = m_ms / static_cast<float>(runs);
  return std::make_tuple(m_time_ms, ep_gs, ep_ec, ep_tIds);
}
#endif // !FLASHMOE_PLATFORM_HIP

// --- Gate kernel runner (shared between CUDA and HIP) ---
template <int Arch, int sharedSize,
          typename TileShape,
          flashmoe::GateReductionLevel grl = flashmoe::GateReductionLevel::singleBlock,
          flashmoe::SoftMaxOptimizationLevel sro = flashmoe::SoftMaxOptimizationLevel::none,
          int threads, typename AccumType, typename Element, typename ElementC
>
__host__ __forceinline__
auto gk_run(gpuStream_t stream,
#if !defined(FLASHMOE_PLATFORM_HIP)
            matx::cudaExecutor& exec,
#endif
            const int& S, const int& H, const int& E, const int& k, const int& EC,
            const int& roundEC, const int& blocks,
            const flashmoe::TPS* tokenIds_packed, int* eCounts, int* eCGuards,
            const void* tokens, const void* gateWeights, void* routing,
            flashmoe::SoftmaxStatePacked* rSp, flashmoe::RingTopKPayload* rTp,
            const int& checkCorrectness
#if !defined(FLASHMOE_PLATFORM_HIP)
            , void* routing_ref, void* routing_interim, uint* tokenIds_idx,
            matx::index_t* tokenIds_ref, matx::index_t* eCounts_ref, matx::index_t* topK,
            float rtol, float atol
#endif
            ) {
  constexpr auto runs = 128;
  constexpr auto warmup = 32;
  gpuEvent_t start, stop;
  CHECK_CUDA(gpuEventCreate(&start));
  CHECK_CUDA(gpuEventCreate(&stop));
  constexpr auto bM = cute::get<0>(TileShape{});
  const flashmoe::gate::GateKernelArgs kArgs{
    .tokens = static_cast<const cuda::std::byte*>(tokens),
    .weights = static_cast<const cuda::std::byte*>(gateWeights),
    .routing = static_cast<cuda::std::byte*>(routing),
    .expertCounts = eCounts,
    .tokenIds = const_cast<flashmoe::TPS*>(tokenIds_packed),
    .S = S, .H = H, .E = E, .k = k, .EC = EC,
    .roundEC = cute::ceil_div(EC, bM) * bM
  };
  const flashmoe::GateContext ctx {
    .ecGuards = eCGuards, .ssp = rSp, .rtp = rTp,
  };
  auto kernel = [&]() {
    flashmoe::gate::forwardKernel<TileShape, Arch, threads, grl, sro, flashmoe::gate::ReturnLogits::yes, AccumType, Element, ElementC>
      <<<blocks, threads, sharedSize, stream>>>(kArgs, ctx);
  };
  kernel();
  CHECK_CUDA(gpuPeekAtLastError());

#if !defined(FLASHMOE_PLATFORM_HIP)
  auto ref_result = std::make_tuple(0.f, -0.0, -0.0, -0.0);
  if (checkCorrectness) {
    const GateArgs gArgs{
      const_cast<void*>(tokens), const_cast<void*>(gateWeights), routing,
      routing_ref, routing_interim, const_cast<flashmoe::TPS*>(tokenIds_packed),
      tokenIds_idx, tokenIds_ref, eCounts, eCGuards, eCounts_ref, topK, rSp, rTp,
      S, H, E, k, EC, roundEC, rtol, atol
    };
    ref_result = reference<warmup, runs, Element, ElementC>(exec, gArgs, start, stop);
  }
#endif

  for (int i = 0; i < warmup; ++i) kernel();
  CHECK_CUDA(gpuStreamSynchronize(stream));
  CHECK_CUDA(gpuEventRecord(start, stream));
  for (int i = 0; i < runs; ++i) kernel();
  CHECK_CUDA(gpuEventRecord(stop, stream));
  float k_ms = 0;
  CHECK_CUDA(gpuEventSynchronize(stop));
  CHECK_CUDA(gpuEventElapsedTime(&k_ms, start, stop));
  CHECK_CUDA(gpuPeekAtLastError());
  const auto k_time_ms = k_ms / static_cast<float>(runs);
  CHECK_CUDA(gpuEventDestroy(start));
  CHECK_CUDA(gpuEventDestroy(stop));

#if !defined(FLASHMOE_PLATFORM_HIP)
  return std::make_tuple(k_time_ms, ref_result);
#else
  return k_time_ms;
#endif
}

template <
  int Arch,
  int bM, int bN, int bK, int pipeStages,
  flashmoe::GateReductionLevel grl = flashmoe::GateReductionLevel::singleBlock,
  flashmoe::SoftMaxOptimizationLevel sro = flashmoe::SoftMaxOptimizationLevel::none,
  typename Element, typename ElementC
>
__host__ __forceinline__
void driver(const size_t& S, const size_t& E, const size_t& H, const int& k, const float& rtol, const float& atol,
            const int& checkCorrectness,
#if !defined(FLASHMOE_PLATFORM_HIP)
            matx::cudaExecutor& exec
#else
            gpuStream_t stream
#endif
            ) {
  flashmoe::flashmoeRange driverRange{
    std::string("S: ")
    .append(std::to_string(S)).append(", E: ")
    .append(std::to_string(E)).append(", H: ")
    .append(std::to_string(H).append(", topK: ")
                             .append(std::to_string(k)))
  };
  const auto M = S;
  const auto N = E;
  const auto K = H;
  const auto eCap = S;
  Element* tokens = nullptr;
  Element* gateWeights = nullptr;
  ElementC* routing = nullptr;
  flashmoe::TPS* tokenIds = nullptr;
  int* eCounts = nullptr;
  int* eCGuards = nullptr;
  flashmoe::SoftmaxStatePacked* rSp = nullptr;
  flashmoe::RingTopKPayload* rTp = nullptr;

#if !defined(FLASHMOE_PLATFORM_HIP)
  auto stream = exec.getStream();
  ElementC* routing_ref = nullptr;
  flashmoe::gate::SoftType* routing_interim = nullptr;
  uint* tokenIds_idx = nullptr;
  matx::index_t* tokenIds_ref = nullptr;
  matx::index_t* eCounts_ref = nullptr;
  matx::index_t* topK_alloc = nullptr;
#endif

  const auto roundM = static_cast<size_t>(cute::max(M, bM));
  const auto roundEC = static_cast<size_t>(cute::max(eCap, bM));
  CHECK_CUDA(gpuMallocAsync(&eCounts, E * sizeof(int), stream));
  CHECK_CUDA(gpuMallocAsync(&eCGuards, E * sizeof(int), stream));
  CHECK_CUDA(gpuMemsetAsync(eCGuards, flashmoe::STALE_AS_BYTE, E * sizeof(int), stream));
  CHECK_CUDA(gpuMallocAsync(&tokens, roundM * K * sizeof(Element), stream));
  CHECK_CUDA(gpuMallocAsync(&gateWeights, K * N * sizeof(Element), stream));
  CHECK_CUDA(gpuMallocAsync(&routing, roundM * N * sizeof(ElementC), stream));
  CHECK_CUDA(gpuMallocManaged(&tokenIds, E * roundEC * sizeof(flashmoe::TPS)));
  if (checkCorrectness) {
#if !defined(FLASHMOE_PLATFORM_HIP)
    CHECK_CUDA(gpuMallocAsync(&routing_ref, M * N * sizeof(ElementC), stream));
    CHECK_CUDA(gpuMallocAsync(&routing_interim, M * N * sizeof(ElementC), stream));
    CHECK_CUDA(gpuMallocManaged(&tokenIds_idx, E * roundEC * sizeof(uint)));
    CHECK_CUDA(gpuMallocManaged(&tokenIds_ref, E * S * sizeof(matx::index_t)));
    CHECK_CUDA(gpuMallocAsync(&topK_alloc, S * E * sizeof(matx::index_t), stream));
    CHECK_CUDA(gpuMallocAsync(&eCounts_ref, E * sizeof(matx::index_t), stream));
#endif
  }
  if (E > bN) {
    const auto sspSize = S * cute::ceil_div(E, bN) * sizeof(flashmoe::SoftmaxStatePacked);
    CHECK_CUDA(gpuMallocAsync(&rSp, sspSize, stream));
    CHECK_CUDA(gpuMemsetAsync(rSp, 0, sspSize, stream));
    const auto rtpSize = 2 * S * cute::ceil_div(E, bN) * sizeof(flashmoe::RingTopKPayload);
    CHECK_CUDA(gpuMallocAsync(&rTp, rtpSize, stream));
    CHECK_CUDA(gpuMemsetAsync(rTp, 0, rtpSize, stream));
  }

  using AccumType = float;
  using TileShape = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>, cute::Int<pipeStages>>;
  constexpr int threads = cute::max(flashmoe::tile::suggest_thread_count<bM, bN, bK, Arch, Element, AccumType>(), bM);
  auto kernel = flashmoe::gate::forwardKernel<TileShape, Arch, threads, grl, sro, flashmoe::gate::ReturnLogits::yes,
  AccumType, Element, ElementC>;
  int bps = 0;
  constexpr auto sharedSize = cute::max(bK * pipeStages * (bM + bN) * sizeof(Element),
                                        bM * bN * sizeof(AccumType));
  int maxSharedMemory = 0;
  CHECK_CUDA(gpuDeviceGetAttribute(&maxSharedMemory, gpuDevAttrMaxSharedMemoryPerBlockOptin, 0));
  if (sharedSize > maxSharedMemory) {
    throw std::runtime_error(std::string("Required shared memory ").append(std::to_string(sharedSize))
                             .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory))
                             .append(" Reduce tile shapes or input sizes."));
  }
  CHECK_CUDA(gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, sharedSize));
  CHECK_CUDA(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, sharedSize));
  const int blocks = cute::min(cute::ceil_div(M, bM) * cute::ceil_div(N, bN),
                               bps * NUM_SMS);
  if (E > blocks * bN) throw std::invalid_argument("E is too big!");
  constexpr auto min_v = -1.f;
  constexpr auto max_v = 1.f;
  std::random_device rd;
  randUniform<Arch, true>(tokens, M * K, rd(), min_v, max_v, stream);
  randUniform<Arch, true>(gateWeights, K * N, rd(), min_v, max_v, stream);

#if !defined(FLASHMOE_PLATFORM_HIP)
  const auto results = gk_run<Arch, sharedSize, TileShape, grl, sro, threads, AccumType, Element, ElementC>(
    stream, exec, S, H, E, k, eCap, roundEC, blocks,
    tokenIds, eCounts, eCGuards, tokens, gateWeights, routing, rSp, rTp,
    checkCorrectness, routing_ref, routing_interim, tokenIds_idx, tokenIds_ref, eCounts_ref, topK_alloc, rtol, atol);
  const float kernel_ms = std::get<0>(results);
  const auto r_tuple = std::get<1>(results);
  const float matx_ms = std::get<0>(r_tuple);
  const double ep_gs = std::get<1>(r_tuple);
  const double ep_ec = std::get<2>(r_tuple);
  const double ep_tIds = std::get<3>(r_tuple);
  printf("%lu, %lu, %lu, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %lf, %lf, %f, %f\n",
         M, N, K, bM, bN, bK, pipeStages, threads, bps, NUM_SMS,
         blocks, rtol, atol, ep_gs, ep_ec, ep_tIds, kernel_ms, matx_ms);
#else
  const auto kernel_ms = gk_run<Arch, sharedSize, TileShape, grl, sro, threads, AccumType, Element, ElementC>(
    stream, S, H, E, k, eCap, roundEC, blocks,
    tokenIds, eCounts, eCGuards, tokens, gateWeights, routing, rSp, rTp, checkCorrectness);
  printf("%lu, %lu, %lu, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, n/a, n/a, n/a, %f, n/a\n",
         M, N, K, bM, bN, bK, pipeStages, threads, bps, NUM_SMS, blocks, rtol, atol, kernel_ms);
#endif

  CHECK_CUDA(gpuFreeAsync(tokens, stream));
  CHECK_CUDA(gpuFreeAsync(gateWeights, stream));
  CHECK_CUDA(gpuFreeAsync(routing, stream));
  CHECK_CUDA(gpuFreeAsync(tokenIds, stream));
  CHECK_CUDA(gpuFreeAsync(eCounts, stream));
  CHECK_CUDA(gpuFreeAsync(eCGuards, stream));
#if !defined(FLASHMOE_PLATFORM_HIP)
  if (checkCorrectness) {
    gpuFreeAsync(routing_ref, stream);
    gpuFreeAsync(tokenIds_idx, stream);
    gpuFreeAsync(tokenIds_ref, stream);
    gpuFreeAsync(routing_interim, stream);
    gpuFreeAsync(topK_alloc, stream);
    gpuFreeAsync(eCounts_ref, stream);
  }
#endif
  if (E > bN) {
    CHECK_CUDA(gpuFreeAsync(rSp, stream));
    CHECK_CUDA(gpuFreeAsync(rTp, stream));
  }
  CHECK_CUDA(gpuStreamSynchronize(stream));
}

// ./testGate <S> <H> <E> <E_max> <k> <checkCorrectness> <rtol> <atol>
__host__ __forceinline__
void kickStart(const int argc, char** argv) {
  using Element = __half;
  using ElementC = float;
  int S = 128; int H = 4096;
  float rtol = 2e-2f; float atol = 2e-3f;
  int E = 8; int E_max = 256; int k = 8; int checkCorrectness = 1;
  constexpr int Arch = FLASHMOE_ARCH;
  printf("S, E, H, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, "
    "blocks, rtol, atol, error_gemm_softmax(%%), error_expert_counts(%%), error_tokenIds(%%), "
    "Kernel_Time(ms), Matx_Time(ms)\n");
  if (argc > 1) S = std::stoi(argv[1]);
  if (argc > 2) H = std::stoi(argv[2]);
  if (argc > 3) E = std::stoi(argv[3]);
  if (argc > 4) E_max = std::stoi(argv[4]);
  if (argc > 5) k = std::stoi(argv[5]);
  if (argc > 6) checkCorrectness = std::stoi(argv[6]);
  if (argc > 7) rtol = std::stof(argv[7]);
  if (argc > 8) atol = std::stof(argv[8]);
  if (k > E) throw std::invalid_argument("k must be at most number of experts");
  CHECK_CUDA(gpuSetDevice(0));
  gpuStream_t stream;
  CHECK_CUDA(gpuStreamCreate(&stream));

#if !defined(FLASHMOE_PLATFORM_HIP)
  matx::cudaExecutor exec{stream};
#endif
  constexpr auto sro = flashmoe::SoftMaxOptimizationLevel::none;
  constexpr int bM = cuda::std::is_same_v<Element, double> ? 64 : 128;
  constexpr int bK = 64;
#if defined(FLASHMOE_PLATFORM_HIP)
  constexpr int pS = 1; // no async copy
#else
  constexpr int pS = FLASHMOE_ARCH >= 800 ? 2 : 1;
#endif
  if (S < 1 || (S > bM && S % bM != 0)) throw std::invalid_argument("S is invalid");
  if (H % bK != 0 || (H / bK) < pS) throw std::invalid_argument("H is invalid");
  for (int i = E; i <= E_max; i *= 2) {
    switch (i) {
      case 8: {
        constexpr int bN = 8;
        driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::singleBlock, sro,
               Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness,
#if !defined(FLASHMOE_PLATFORM_HIP)
               exec
#else
               stream
#endif
               );
      } break;
      case 16: {
        constexpr int bN = 16;
        driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::singleBlock, sro,
               Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness,
#if !defined(FLASHMOE_PLATFORM_HIP)
               exec
#else
               stream
#endif
               );
      } break;
      case 32: {
        constexpr int bN = 32;
        driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::singleBlock, sro,
               Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness,
#if !defined(FLASHMOE_PLATFORM_HIP)
               exec
#else
               stream
#endif
               );
      } break;
      default: {
        if (i > 32) {
          constexpr int bN = 32;
          driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::multiBlock, sro,
                 Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness,
#if !defined(FLASHMOE_PLATFORM_HIP)
                 exec
#else
                 stream
#endif
                 );
        }
      }
    }
  }
  CHECK_CUDA(gpuStreamDestroy(stream));
}

int main(const int argc, char** argv) {
  kickStart(argc, argv);
}
