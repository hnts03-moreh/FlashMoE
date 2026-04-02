//
// Created by osayamen on 2/2/26.
//
// unit test and benchmark for gated GEMM of gated MLP

#include <random>

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/tile.cuh"

#if defined(FLASHMOE_PLATFORM_HIP)
#include "../include/flashmoe/infra/activation.cuh"
#else
#include <cutlass/epilogue/thread/activation.h>
#endif

template<typename TileGEMM, typename Activation, typename ElementC, typename Element>
__device__ __forceinline__
void gemmMainloop(cuda::std::byte *__restrict__ const&workspace,
                  const Element *__restrict__ const&a,
                  const Element *__restrict__ const&b,
                  const Element *__restrict__ const&bV,
                  ElementC *__restrict__ const&c,
                  const ElementC *__restrict__ const&bias,
                  const ElementC *__restrict__ const&biasV,
                  const typename TileGEMM::AccumType &swishAlpha,
                  const typename TileGEMM::AccumType &swishBeta,
                  const int &M, const int &N, const int &K, const int &tileIdx) {
  auto *__restrict__ gateCache = workspace + cutlass::round_up(cute::max(TileGEMM::SharedSizeC::value,
                                                                         TileGEMM::SharedSizeAB::value),
                                                               TileGEMM::GeneralAlignment::value);
  using BLAS = TileGEMM::BLAS;
  auto accumulator = BLAS::suggest_accumulator();
  using BM = cute::Int<flashmoe_blas::size_of<BLAS>::m>;
  using BN = cute::Int<flashmoe_blas::size_of<BLAS>::n>;
  const auto tileCoord = flashmoe::tile::idx2Coord(M / BM{}, N / BN{}, tileIdx);
  constexpr TileGEMM tileMainloop{};
  tileMainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
  __syncthreads();
  const auto gD = flashmoe::tile::getBias<BM{}, BN{}>(bias, M, N, cute::select<0, 1>(tileCoord));
  auto d_frag = flashmoe_blas::make_fragment_like<ElementC>(accumulator.get_results());
  flashmoe_blas::copy_fragment<flashmoe_blas::alignment_of<BLAS>::c>(gD, d_frag, accumulator);
  constexpr Activation act{};
  using AccumType = decltype(accumulator)::value_type;
  constexpr flashmoe::Converter<AccumType, ElementC> loadConv{};
  constexpr flashmoe::Converter<ElementC, AccumType> storeConv{};
  const auto c_frag = accumulator.get_results();
  constexpr int accum_size = flashmoe_blas::size(c_frag);
#pragma unroll
  for (int i = 0; i < accum_size; ++i) {
    const auto g = (c_frag(i) + loadConv(d_frag(i))) * swishBeta;
    d_frag(i) = storeConv(swishAlpha * act(g));
  }
  auto sGate = flashmoe_blas::make_tensor(reinterpret_cast<ElementC *>(gateCache), BLAS::suggest_layout_smem_c());
  flashmoe_blas::copy_fragment<flashmoe_blas::alignment_of<BLAS>::c>(d_frag, sGate, accumulator);
  tileMainloop(workspace, a, bV, accumulator, M, N, K, tileCoord);
  auto cv_frag = accumulator.get_results();
  const auto gV = flashmoe::tile::getBias<BM{}, BN{}>(biasV, M, N, cute::select<0, 1>(tileCoord));
  flashmoe_blas::copy_fragment<flashmoe_blas::alignment_of<BLAS>::c>(gV, d_frag, accumulator);
#pragma unroll
  for (int i = 0; i < accum_size; ++i) {
    cv_frag(i) = cv_frag(i) + loadConv(d_frag(i));
  }
  __syncthreads();
  flashmoe_blas::copy_fragment<flashmoe_blas::alignment_of<BLAS>::c>(sGate, d_frag, accumulator);
#pragma unroll
  for (int i = 0; i < accum_size; ++i) {
    d_frag(i) = storeConv(cv_frag(i) * loadConv(d_frag(i)));
  }
  auto gC = flashmoe::tile::getC<BM{}, BN{}, flashmoe_blas::arrangement_of_v_c<BLAS> >(c, M, N,
    cute::select<0, 1>(tileCoord));
  auto sC = flashmoe_blas::make_tensor(reinterpret_cast<ElementC *>(workspace), BLAS::suggest_layout_smem_c());
  __syncthreads();
  flashmoe_blas::copy_fragment<flashmoe_blas::alignment_of<BLAS>::c>(d_frag, sC, accumulator);
  __syncthreads();
  flashmoe_blas::copy<BLAS, flashmoe_blas::alignment_of<BLAS>::c>(sC, gC);
}

#define SC(T, v) static_cast<T>(v)
template<typename TileGEMM, typename Activation, typename Element, typename ElementC>
  requires(flashmoe_blas::is_blas_execution_v<typename TileGEMM::BLAS>)
__launch_bounds__(TileGEMM::BLAS::max_threads_per_block, 1)
__global__ void gk(const Element *__restrict__ a,
                   const Element *__restrict__ b, const Element *__restrict__ bV,
                   ElementC *__restrict__ c,
                   const ElementC *__restrict__ bias, const ElementC *__restrict__ biasV,
                   const __grid_constant__ typename TileGEMM::AccumType swishAlpha,
                   const __grid_constant__ typename TileGEMM::AccumType swishBeta,
                   const __grid_constant__ int M,
                   const __grid_constant__ int N,
                   const int __grid_constant__ K) {
  using BLAS = TileGEMM::BLAS;
  constexpr int bM = flashmoe_blas::size_of<BLAS>::m;
  constexpr int bN = flashmoe_blas::size_of<BLAS>::n;
  const int nTiles = (M / bM) * (N / bN);
  extern __shared__ __align__(TileGEMM::GeneralAlignment::value) cuda::std::byte gemmWorkspace[];
  for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; tileIdx += SC(int, gridDim.x)) {
    gemmMainloop<TileGEMM, Activation>(gemmWorkspace, a, b, bV, c, bias, biasV, swishAlpha, swishBeta, M, N, K,
                                       tileIdx);
  }
}

#if !defined(FLASHMOE_PLATFORM_HIP)
// CUDA: full reference via MatX
template<int warmup, int runs, typename AccumType, typename Activation, typename Element, typename ElementC>
__host__ __forceinline__
auto reference(void *const&a, void *const&b, void *const&bV,
               void *const&bias, void *const&biasV, void *const&bias_interim,
               void *const&c_ref, void *const&c_interim, void *const&c_final,
               void *const&c_ext, const AccumType &swishAlpha, const AccumType &swishBeta,
               const int &M, const int &N, const int &K, const float &rtol, const float &atol,
               matx::cudaExecutor &exec) {
  auto *mx_a = static_cast<Element *>(a);
  auto *mx_b = static_cast<Element *>(b);
  auto *mx_bv = static_cast<Element *>(bV);
  auto *mx_bias = static_cast<ElementC *>(bias);
  auto *mx_bias_interim = static_cast<AccumType *>(bias_interim);
  auto *mx_bias_v = static_cast<ElementC *>(biasV);
  auto *mx_c_ref = static_cast<ElementC *>(c_ref);
  auto *mx_c_final = static_cast<ElementC *>(c_final);
  auto *mx_c_interim = static_cast<AccumType *>(c_interim);
  auto *mx_c_ext = static_cast<ElementC *>(c_ext);

  auto tA = matx::make_tensor<Element>(mx_a, {M, K});
  auto tB = matx::make_tensor<Element>(mx_b, {N, K});
  auto tBV = matx::make_tensor<Element>(mx_bv, {N, K});
  auto tC = matx::make_tensor<ElementC>(mx_c_ref, {M, N});
  auto tC_interim = matx::make_tensor<AccumType>(mx_c_interim, {M, N});
  auto tC_final = matx::make_tensor<ElementC>(mx_c_final, {M, N});
  auto tCx = matx::make_tensor<ElementC>(mx_c_ext, {M, N});
  auto tBias = matx::make_tensor<ElementC>(mx_bias, {N});
  auto tBias_interim = matx::make_tensor<AccumType>(mx_bias_interim, {N});
  auto tBiasV = matx::make_tensor<ElementC>(mx_bias_v, {N});

  auto swiOut = matx::make_tensor<AccumType>({M, N});
  auto tC_stash = matx::make_tensor<AccumType>({M, N});

  (tC_interim = matx::matmul(tA, tB.PermuteMatrix())).run(exec);
  (tBias_interim = matx::apply(Converter<AccumType, ElementC>{}, tBias)).run(exec);
  (swiOut = matx::apply(Activation{}, swishBeta * (tC_interim + tBias_interim))).run(exec);
  (tC = matx::apply(Converter<ElementC, AccumType>{}, swishAlpha * swiOut)).run(exec);
  (tC_interim = matx::matmul(tA, tBV.PermuteMatrix())).run(exec);
  (tBias_interim = matx::apply(Converter<AccumType, ElementC>{}, tBiasV)).run(exec);
  (tC_stash = matx::apply(Converter<AccumType, Element>{}, tC)).run(exec);
  (tC_final = matx::apply(Converter<ElementC, AccumType>{}, (tC_interim + tBias_interim) * tC_stash)).run(exec);
  exec.sync();
  matx::cudaExecutor exec1{exec.getStream()};
  auto num_matches = matx::make_tensor<long int>({});
  (num_matches = matx::sum(matx::isclose(tCx, tC_final, rtol, atol))).run(exec1);
  exec1.sync();
  const auto ep = (1.0 - (static_cast<double>(num_matches()) / static_cast<double>(M * N))) * 100;
  for (int i = 0; i < warmup; ++i) {
    (tC = swishAlpha * matx::apply(Activation{}, swishBeta * (matx::matmul(tA, tB.PermuteMatrix()) + tBias)) *
          (matx::matmul(tA, tBV.PermuteMatrix()) + tBiasV)).run(exec);
  }
  exec.sync();
  exec.start_timer();
  for (int i = 0; i < runs; ++i) {
    (tC = swishAlpha * matx::apply(Activation{}, swishBeta * (matx::matmul(tA, tB.PermuteMatrix()) + tBias)) *
          (matx::matmul(tA, tBV.PermuteMatrix()) + tBiasV)).run(exec);
  }
  exec.stop_timer();
  exec.sync();
  CHECK_CUDA(cudaPeekAtLastError());
  return std::make_tuple(ep, exec.get_time_ms() / static_cast<float>(runs));
}

template<typename TileGEMM, typename Activation, int threads, int sharedSize, typename ActM,
  typename AccumType, typename Element, typename ElementC>
__host__ __forceinline__
auto gk_run(Element *const&a, Element *const&b, Element *const&bV,
            ElementC *const&c, ElementC *const&c_ref, ElementC *const&bias, ElementC *const&biasV,
            AccumType *const&bias_interim,
            AccumType *const&c_interim, ElementC *const&c_final, const AccumType &swishAlpha,
            const AccumType &swishBeta,
            const int &M, const int &N, const int &K, const int &blocks,
            const float &rtol, const float &atol, matx::cudaExecutor &exec) {
  constexpr auto runs = 128;
  constexpr auto warmup = 128;
  gk<TileGEMM, Activation><<<blocks, threads, sharedSize, exec.getStream()>>>(
    a, b, bV, c, bias, biasV, swishAlpha, swishBeta, M, N, K);
  const auto [error_pct, ref_time_ms] = reference<warmup, runs, AccumType, ActM, MXE<Element>, MXE<ElementC> >(
    a, b, bV, bias, biasV, bias_interim, c_ref, c_interim, c_final, c, swishAlpha, swishBeta,
    M, N, K, rtol, atol, exec);
  for (int i = 0; i < warmup; ++i) {
    gk<TileGEMM, Activation><<<blocks, threads, sharedSize, exec.getStream()>>>(
      a, b, bV, c, bias, biasV, swishAlpha, swishBeta, M, N, K);
  }
  exec.sync();
  exec.start_timer();
  for (int i = 0; i < runs; ++i) {
    gk<TileGEMM, Activation><<<blocks, threads, sharedSize, exec.getStream()>>>(
      a, b, bV, c, bias, biasV, swishAlpha, swishBeta, M, N, K);
  }
  exec.stop_timer();
  exec.sync();
  CHECK_CUDA(cudaPeekAtLastError());
  const auto k_time_ms = exec.get_time_ms() / static_cast<float>(runs);
  return std::make_tuple(k_time_ms, error_pct, ref_time_ms);
}

template<int bM, int bN, int bK, int pipeStages, typename AccumType, typename Element, typename ElementC>
__host__ __forceinline__
void driver(const int &M, const int &N, const int &K, const float &rtol, const float &atol, matx::cudaExecutor &exec) {
  Element *a = nullptr; Element *b = nullptr; Element *bV = nullptr;
  ElementC *c = nullptr; ElementC *c_ref = nullptr; AccumType *c_interim = nullptr;
  ElementC *c_final = nullptr; ElementC *bias = nullptr; ElementC *biasV = nullptr;
  AccumType *bias_interim = nullptr;
  auto stream = exec.getStream();
  CHECK_CUDA(cudaMallocAsync(&a, sizeof(Element)* M * static_cast<size_t>(K), stream));
  CHECK_CUDA(cudaMallocAsync(&b, sizeof(Element) * N * K, stream));
  CHECK_CUDA(cudaMallocAsync(&bV, sizeof(Element) * N * K , stream));
  CHECK_CUDA(cudaMallocAsync(&c, sizeof(ElementC) * M * N, stream));
  CHECK_CUDA(cudaMallocAsync(&c_ref, sizeof(ElementC)* M * N, stream));
  CHECK_CUDA(cudaMallocAsync(&c_final, sizeof(ElementC) * M * N, stream));
  CHECK_CUDA(cudaMallocAsync(&c_interim, sizeof(AccumType) * M * N, stream));
  CHECK_CUDA(cudaMallocAsync(&bias, N * sizeof(ElementC), stream));
  CHECK_CUDA(cudaMallocAsync(&biasV, N * sizeof(ElementC), stream));
  CHECK_CUDA(cudaMallocAsync(&bias_interim, N * sizeof(AccumType), stream));

  using Act = cutlass::epilogue::thread::SiLu<AccumType>;
  using ActM = cutlass::epilogue::thread::SiLu<MXE<AccumType> >;
  constexpr int threads = flashmoe::tile::suggest_thread_count<bM, bN, bK, FLASHMOE_ARCH, Element, AccumType>();
  using TileGEMM = flashmoe::tile::CollectiveMainloop<
    bM, bN, bK, FLASHMOE_ARCH, Element, AccumType, threads, pipeStages
  >;
  auto kernel = gk<TileGEMM, Act, Element, ElementC>;
  int bps = 0;
  constexpr auto totalSharedSize = cutlass::round_up(cute::max(TileGEMM::SharedSizeC::value,
                                                               TileGEMM::SharedSizeAB::value),
                                                     TileGEMM::GeneralAlignment::value) + TileGEMM::SharedSizeC::value;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, totalSharedSize));
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, totalSharedSize));
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory,cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  if (totalSharedSize > maxSharedMemory) {
    throw std::runtime_error(std::string("Required shared memory ").append(std::to_string(totalSharedSize))
        .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory)).append(
          " Reduce tile shapes or input sizes."));
  }
  const int blocks = min((M / bM) * (N / bN), bps * NUM_SMS);
  std::random_device rd;
  constexpr auto min_v = -1.f; constexpr auto max_v = 1.f;
  randUniform<FLASHMOE_ARCH>(a, static_cast<size_t>(M) * K, rd(), min_v, max_v, exec.getStream());
  randUniform<FLASHMOE_ARCH>(b, static_cast<size_t>(N) * K, rd(), min_v, max_v, exec.getStream());
  randUniform<FLASHMOE_ARCH>(bV, static_cast<size_t>(N) * K, rd(), min_v, max_v, exec.getStream());
  randUniform<FLASHMOE_ARCH>(bias, N, rd(), min_v, max_v, exec.getStream());
  randUniform<FLASHMOE_ARCH>(biasV, N, rd(), min_v, max_v, exec.getStream());
  const auto swishAlpha = static_cast<AccumType>(random_float(min_v, max_v, rd()));
  const auto swishBeta = static_cast<AccumType>(random_float(min_v, max_v, rd()));
  const auto [k_ms, e_p, r_ms] = gk_run<TileGEMM, Act, threads, totalSharedSize, ActM>(a, b, bV, c, c_ref, bias, biasV,
    bias_interim, c_interim, c_final, swishAlpha, swishBeta, M, N, K, blocks, rtol, atol, exec);

  printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %f, %f\n",
         M, N, K, bM, bN, bK, pipeStages, threads, bps, NUM_SMS, blocks, rtol, atol, e_p, k_ms, r_ms);

  CHECK_CUDA(cudaPeekAtLastError());
  cudaFreeAsync(a, stream); cudaFreeAsync(b, stream); cudaFreeAsync(bV, stream);
  cudaFreeAsync(c, stream); cudaFreeAsync(c_ref, stream); cudaFreeAsync(c_final, stream);
  cudaFreeAsync(bias, stream); cudaFreeAsync(biasV, stream); cudaFreeAsync(bias_interim, stream);
  cudaFreeAsync(c_interim, stream);
  cudaStreamSynchronize(stream);
}

__host__ __forceinline__
void kickStart(const int argc, char **argv) {
  int MNK = 2; int MNK_max = 8192;
  float rtol = 2e-2f; float atol = 2e-3f;
  using Element = __half; using ElementC = Element; using AccumType = float;
  printf("M, N, K, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, blocks, rtol, atol, error(%%), Kernel_Time(ms), Matx_Time(ms)\n");
  if (argc > 1) MNK = std::stoi(argv[1]);
  if (argc > 2) MNK_max = std::stoi(argv[2]);
  if (argc > 3) rtol = std::stof(argv[3]);
  if (argc > 4) atol = std::stof(argv[4]);
  cudaSetDevice(0);
  cudaStream_t stream; cudaStreamCreate(&stream);
  matx::cudaExecutor exec{stream, true};
  for (int i = MNK; i <= MNK_max; i *= 2) {
    switch (i) {
      case 2: driver<2, 2, 2, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec); break;
      case 4: driver<4, 4, 4, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec); break;
      case 8: driver<8, 8, 8, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec); break;
      case 16: driver<16, 16, 16, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec); break;
      case 32: driver<32, 32, 32, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec); break;
      case 64: driver<64, 64, 64, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec); break;
      default: {
        constexpr int pS = 2; constexpr int bK = 32;
        if (i >= 128 && i <= 2048) {
          driver<128, 64, bK, pS, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec);
        } else if (i > 2048) {
          driver<128, cute::max(64, 64 * (4 / sizeof(Element))), bK, pS, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec);
        }
      }
    }
  }
  cudaStreamDestroy(stream);
}

int main(const int argc, char **argv) {
  kickStart(argc, argv);
}

#else
// HIP: kernel benchmark + host CPU reference for accuracy verification
#include "host_reference.cuh"

template<int bM, int bN, int bK, int pipeStages, typename AccumType, typename Element, typename ElementC>
__host__ __forceinline__
void driver(const int &M, const int &N, const int &K, const float &rtol, const float &atol, gpuStream_t stream) {
  Element *a = nullptr; Element *b = nullptr; Element *bV = nullptr;
  ElementC *c = nullptr; ElementC *bias = nullptr; ElementC *biasV = nullptr;
  CHECK_CUDA(gpuMallocAsync(&a, sizeof(Element) * M * static_cast<size_t>(K), stream));
  CHECK_CUDA(gpuMallocAsync(&b, sizeof(Element) * N * K, stream));
  CHECK_CUDA(gpuMallocAsync(&bV, sizeof(Element) * N * K, stream));
  CHECK_CUDA(gpuMallocAsync(&c, sizeof(ElementC) * M * N, stream));
  CHECK_CUDA(gpuMallocAsync(&bias, N * sizeof(ElementC), stream));
  CHECK_CUDA(gpuMallocAsync(&biasV, N * sizeof(ElementC), stream));

  using Act = flashmoe::hip_compat::SiLu<AccumType>;
  constexpr int threads = flashmoe::tile::suggest_thread_count<bM, bN, bK, FLASHMOE_ARCH, Element, AccumType>();
  using TileGEMM = flashmoe::tile::CollectiveMainloop<
    bM, bN, bK, FLASHMOE_ARCH, Element, AccumType, threads, pipeStages
  >;
  auto kernel = gk<TileGEMM, Act, Element, ElementC>;
  int bps = 0;
  constexpr auto totalSharedSize = cutlass::round_up(cute::max(TileGEMM::SharedSizeC::value,
                                                               TileGEMM::SharedSizeAB::value),
                                                     TileGEMM::GeneralAlignment::value) + TileGEMM::SharedSizeC::value;
  CHECK_CUDA(gpuFuncSetAttribute(kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, totalSharedSize));
  CHECK_CUDA(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, totalSharedSize));
  const int blocks = cute::min((M / bM) * (N / bN), bps * NUM_SMS);
  std::random_device rd;
  constexpr auto min_v = -1.f; constexpr auto max_v = 1.f;
  randUniform<FLASHMOE_ARCH>(a, static_cast<size_t>(M) * K, rd(), min_v, max_v, stream);
  randUniform<FLASHMOE_ARCH>(b, static_cast<size_t>(N) * K, rd(), min_v, max_v, stream);
  randUniform<FLASHMOE_ARCH>(bV, static_cast<size_t>(N) * K, rd(), min_v, max_v, stream);
  randUniform<FLASHMOE_ARCH>(bias, N, rd(), min_v, max_v, stream);
  randUniform<FLASHMOE_ARCH>(biasV, N, rd(), min_v, max_v, stream);
  const auto swishAlpha = static_cast<AccumType>(random_float(min_v, max_v, rd()));
  const auto swishBeta = static_cast<AccumType>(random_float(min_v, max_v, rd()));

  // Run kernel once for accuracy check
  gk<TileGEMM, Act><<<blocks, threads, totalSharedSize, stream>>>(a, b, bV, c, bias, biasV, swishAlpha, swishBeta, M, N, K);
  CHECK_CUDA(gpuStreamSynchronize(stream));

  // Copy to host for CPU reference
  const size_t sz_a = static_cast<size_t>(M) * K;
  const size_t sz_b = static_cast<size_t>(N) * K;
  const size_t sz_c = static_cast<size_t>(M) * N;
  std::vector<Element> h_a(sz_a), h_b(sz_b), h_bV(sz_b);
  std::vector<ElementC> h_bias(N), h_biasV(N), h_c(sz_c), h_c_ref(sz_c);
  CHECK_CUDA(hipMemcpy(h_a.data(), a, sz_a * sizeof(Element), hipMemcpyDeviceToHost));
  CHECK_CUDA(hipMemcpy(h_b.data(), b, sz_b * sizeof(Element), hipMemcpyDeviceToHost));
  CHECK_CUDA(hipMemcpy(h_bV.data(), bV, sz_b * sizeof(Element), hipMemcpyDeviceToHost));
  CHECK_CUDA(hipMemcpy(h_bias.data(), bias, N * sizeof(ElementC), hipMemcpyDeviceToHost));
  CHECK_CUDA(hipMemcpy(h_biasV.data(), biasV, N * sizeof(ElementC), hipMemcpyDeviceToHost));
  CHECK_CUDA(hipMemcpy(h_c.data(), c, sz_c * sizeof(ElementC), hipMemcpyDeviceToHost));

  // CPU reference: swishAlpha * SiLu(swishBeta * (A@B.T + bias)) * (A@BV.T + biasV)
  host_ref::gated_gemm_ref<host_ref::HostSiLu>(
      h_a.data(), h_b.data(), h_bV.data(), h_bias.data(), h_biasV.data(),
      h_c_ref.data(), M, N, K,
      static_cast<float>(swishAlpha), static_cast<float>(swishBeta));
  const double error_pct = host_ref::compare_isclose(h_c.data(), h_c_ref.data(), M * N, rtol, atol);

  // Benchmark
  constexpr auto runs = 128; constexpr auto warmup = 32;
  for (int i = 0; i < warmup; ++i) {
    gk<TileGEMM, Act><<<blocks, threads, totalSharedSize, stream>>>(a, b, bV, c, bias, biasV, swishAlpha, swishBeta, M, N, K);
  }
  CHECK_CUDA(gpuStreamSynchronize(stream));
  gpuEvent_t start, stop;
  CHECK_CUDA(gpuEventCreate(&start)); CHECK_CUDA(gpuEventCreate(&stop));
  CHECK_CUDA(gpuEventRecord(start, stream));
  for (int i = 0; i < runs; ++i) {
    gk<TileGEMM, Act><<<blocks, threads, totalSharedSize, stream>>>(a, b, bV, c, bias, biasV, swishAlpha, swishBeta, M, N, K);
  }
  CHECK_CUDA(gpuEventRecord(stop, stream));
  CHECK_CUDA(gpuEventSynchronize(stop));
  float k_ms = 0;
  CHECK_CUDA(gpuEventElapsedTime(&k_ms, start, stop));
  printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %f, n/a\n",
         M, N, K, bM, bN, bK, pipeStages, threads, bps, NUM_SMS, blocks, rtol, atol,
         error_pct, k_ms / static_cast<float>(runs));
  gpuEventDestroy(start); gpuEventDestroy(stop);
  gpuFreeAsync(a, stream); gpuFreeAsync(b, stream); gpuFreeAsync(bV, stream);
  gpuFreeAsync(c, stream); gpuFreeAsync(bias, stream); gpuFreeAsync(biasV, stream);
  gpuStreamSynchronize(stream);
}

__host__ __forceinline__
void kickStart(const int argc, char **argv) {
  int MNK = 2; int MNK_max = 8192;
  float rtol = 2e-2f; float atol = 2e-3f;
  using Element = __half; using ElementC = Element; using AccumType = float;
  printf("M, N, K, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, blocks, rtol, atol, error(%%), Kernel_Time(ms), Ref_Time(ms)\n");
  if (argc > 1) MNK = std::stoi(argv[1]);
  if (argc > 2) MNK_max = std::stoi(argv[2]);
  if (argc > 3) rtol = std::stof(argv[3]);
  if (argc > 4) atol = std::stof(argv[4]);
  gpuSetDevice(0);
  gpuStream_t stream; gpuStreamCreate(&stream);
  for (int i = MNK; i <= MNK_max; i *= 2) {
    switch (i) {
      case 2: driver<2, 2, 2, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, stream); break;
      case 4: driver<4, 4, 4, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, stream); break;
      case 8: driver<8, 8, 8, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, stream); break;
      case 16: driver<16, 16, 16, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, stream); break;
      case 32: driver<32, 32, 32, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, stream); break;
      case 64: driver<64, 64, 64, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, stream); break;
      default: {
        constexpr int pS = 1; constexpr int bK = 32;
        if (i >= 128 && i <= 2048) driver<128, 64, bK, pS, AccumType, Element, ElementC>(i, i, i, rtol, atol, stream);
        else if (i > 2048) driver<128, cute::max(64, 64 * (4 / sizeof(Element))), bK, pS, AccumType, Element, ElementC>(i, i, i, rtol, atol, stream);
      }
    }
  }
  gpuStreamDestroy(stream);
}

int main(const int argc, char **argv) {
  kickStart(argc, argv);
}
#endif // FLASHMOE_PLATFORM_HIP
