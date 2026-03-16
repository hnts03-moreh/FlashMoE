//
// Created by osayamen on 12/22/25.
//

// place to experiment

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include "../include/flashmoe/tile.cuh"
#include "../include/flashmoe/infra/activation.cuh"
#include "../include/flashmoe/infra/packed.cuh"
#include "../include/flashmoe/infra/vt.cuh"

#if !defined(CHECK_CUDA)
#  define CHECK_CUDA(e)                                      \
do {                                                         \
    cudaError_t code = (e);                                  \
    if (code != cudaSuccess) {                               \
        fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",         \
            __FILE__, __LINE__, #e,                          \
            cudaGetErrorName(code),                          \
            cudaGetErrorString(code));                       \
        fflush(stderr);                                      \
        exit(1);                                             \
    }                                                        \
} while (0)
#endif
namespace py = pybind11;

template<typename TileGEMM, typename Activation, flashmoe::MLPMatmulType mt, typename ElementC, typename Element>
__device__ __forceinline__
void gemmMainloop(cuda::std::byte* __restrict__ const& workspace,
    const Element* __restrict__ const& a,
    const Element* __restrict__ const& b,
    const Element* __restrict__ const& bV,
    ElementC* __restrict__ const& c,
    const ElementC* __restrict__ const& bias,
    const ElementC* __restrict__ const& biasV,
    const typename TileGEMM::AccumType& swishAlpha,
    const typename TileGEMM::AccumType& swishBeta,
    const int& M, const int& N, const int& K, const int& tileIdx) {
  using BLAS = TileGEMM::BLAS;
  auto accumulator = BLAS::suggest_accumulator();
  using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
  using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
  const auto tileCoord = flashmoe::tile::idx2Coord(M / BM{}, N / BN{}, tileIdx);
  // gmem -> rmem: prefetch bias
  const auto gD = flashmoe::tile::getBias<BM{}, BN{}>(bias, M, N, cute::select<0, 1>(tileCoord));
  auto d_frag = cublasdx::make_fragment_like<ElementC>(accumulator.get_results());
  cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gD, d_frag, accumulator);
  // compute Tile
  constexpr TileGEMM tileMainloop{};
  tileMainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
  // Epilogue
  constexpr Activation act{}; // activation function like relu, etc
  using AccumType = decltype(accumulator)::value_type;
  // ElementC -> accum type
  constexpr flashmoe::Converter<AccumType, ElementC> loadConv{};
  // accum type -> ElementC
  constexpr flashmoe::Converter<ElementC, AccumType> storeConv{};
  const auto c_frag = accumulator.get_results();
  constexpr int accum_size = cublasdx::size(c_frag);
  if constexpr (mt == flashmoe::MLPMatmulType::gated) {
    __syncthreads();
    auto* __restrict__ gateCache = workspace + cutlass::round_up(cute::max(TileGEMM::SharedSizeC::value,
        TileGEMM::SharedSizeAB::value), TileGEMM::GeneralAlignment::value);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&](auto i) {
      const auto g = (c_frag(i) + loadConv(d_frag(i))) * swishBeta;
      d_frag(i) = storeConv(swishAlpha * act(g));
    });
    // rmem -> smem, cache gate results
    // holding in registers otherwise would blow up pressure
    auto sGate = cublasdx::make_tensor(reinterpret_cast<ElementC*>(gateCache), BLAS::suggest_layout_smem_c());
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, sGate, accumulator);
    // now, compute v tile
    tileMainloop(workspace, a, bV, accumulator, M, N, K, tileCoord);
    auto cv_frag = accumulator.get_results();
    const auto gV = flashmoe::tile::getBias<BM{}, BN{}>(biasV, M, N, cute::select<0, 1>(tileCoord));
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gV, d_frag, accumulator);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&](auto i) {
      // x = (a @ bV) + biasV
      cv_frag(i) = cv_frag(i) + loadConv(d_frag(i));
    });
    // smem -> rmem, load g
    __syncthreads();
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(sGate, d_frag, accumulator);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&](auto i) {
      // y = x * (act(a @ b))
      d_frag(i) = storeConv(cv_frag(i) * loadConv(d_frag(i)));
    });
  }
  else {
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&c_frag, &d_frag](auto i) {
      d_frag(i) = storeConv(act(c_frag(i) + loadConv(d_frag(i))));
    });
  }
  auto gC = flashmoe::tile::getC<BM{}, BN{}, cublasdx::arrangement_of_v_c<BLAS>>(c, M, N,
      cute::select<0, 1>(tileCoord));
  // rmem -> smem
  auto sC = cublasdx::make_tensor(reinterpret_cast<ElementC*>(workspace), BLAS::suggest_layout_smem_c());
  __syncthreads();
  cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, sC, accumulator);
  __syncthreads();
  // smem -> gmem
  cublasdx::copy<BLAS, cublasdx::alignment_of<BLAS>::c>(sC, gC);
}

template<typename TileGEMM, typename Activation, flashmoe::MLPMatmulType mt, typename Element, typename ElementC>
requires(cublasdx::is_blas_execution_v<typename TileGEMM::BLAS>)
__launch_bounds__(TileGEMM::BLAS::max_threads_per_block, 1)
__global__ void gk(const Element* __restrict__ a, const Element* __restrict__ b,
  const Element* __restrict__ bV, ElementC* __restrict__ c, const ElementC* __restrict__ bias,
  const ElementC* __restrict__ biasV, const __grid_constant__ typename TileGEMM::AccumType swishAlpha,
  const __grid_constant__ typename TileGEMM::AccumType swishBeta,
  const __grid_constant__ int M, const __grid_constant__ int N,
  const int __grid_constant__ K) {
  using BLAS = TileGEMM::BLAS;
  constexpr int bM = cublasdx::size_of<BLAS>::m;
  constexpr int bN = cublasdx::size_of<BLAS>::n;
  const int nTiles = (M / bM) * (N / bN);
  extern __shared__ __align__(TileGEMM::GeneralAlignment::value) cuda::std::byte gemmWorkspace[];
  for (int tileIdx = blockIdx.x; tileIdx < nTiles; tileIdx += gridDim.x) {
    gemmMainloop<TileGEMM, Activation, mt>(gemmWorkspace, a, b, bV, c, bias, biasV, swishAlpha, swishBeta,
      M, N, K, tileIdx);
  }
}

// note this is not an optimal implementation!
template<typename Element>
__global__ void gatherTokens(const flashmoe::TPS* __restrict__ tokenIds,
  const Element* __restrict__ src, Element* __restrict__ dst,
  const __grid_constant__ uint roundEC,
  const __grid_constant__ int count,
  const __grid_constant__ uint S,
  const __grid_constant__ uint H) {
  const auto srcTensor = cute::make_tensor(cute::make_gmem_ptr(src),
    cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
  auto dstTensor = cute::make_tensor(cute::make_gmem_ptr(dst),
    cute::make_layout(cute::make_shape(roundEC, H), cute::LayoutRight{}));
  for (uint idx = blockIdx.x; idx < count; idx += gridDim.x) {
    const auto tIdx = tokenIds[idx].tokenIdx;
    for (uint i = threadIdx.x; i < H; i += blockDim.x) {
      dstTensor(idx, i) = srcTensor(tIdx, i);
    }
  }
}

template<typename Element>
__global__ void combineReference( const __grid_constant__ int S,
    const __grid_constant__ int H, const __grid_constant__ int roundEC,
    const __grid_constant__ int expertCount,
    const __grid_constant__ int topK,
    const Element* __restrict__ tokens, // [roundEC, H]
    const flashmoe::TPS* __restrict__ tokenIds, //[roundEC] metadata
    Element* __restrict__ result // [S, H]
    ) {
  const auto tIds = cute::make_tensor(cute::make_gmem_ptr(tokenIds),
      cute::make_layout(cute::make_shape(roundEC), cute::LayoutRight{}));
  const auto tokTensor = cute::make_tensor(cute::make_gmem_ptr(tokens),
      cute::make_layout(cute::make_shape(roundEC, H), cute::LayoutRight{}));
  auto resultTensor = cute::make_tensor(cute::make_gmem_ptr(result),
      cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
  for (int j = static_cast<int>(blockIdx.x); j < expertCount; j += static_cast<int>(gridDim.x)) {
    const auto tokenId = tIds(j);
    if (topK == 1) {
      for (auto k = threadIdx.x; k < H; k += blockDim.x) {
        resultTensor(tokenId.tokenIdx, k) = tokTensor(j, k);
      }
    }
    else {
      // a token is mapped to an expert at most once, so we can safely accumulate without atomics
      constexpr flashmoe::Converter<float, Element> loadOp{};
      constexpr flashmoe::Converter<Element, float> storeOp{};
      for (auto k = threadIdx.x; k < H; k += blockDim.x) {
        // read token value and convert to float.
        const auto v = loadOp(tokTensor(j, k));
        const float scaledV = v * tokenId.probability;
        // read current result value
        const auto rv = resultTensor(tokenId.tokenIdx, k);
        // store summed result
        resultTensor(tokenId.tokenIdx, k) = storeOp(scaledV) + rv;
      }
    }
  }
}

constexpr int S = 128; // jit value
constexpr int H = 128; // jit value
constexpr int I = 128; // jit value
constexpr int E = 8; // jit value
constexpr int EC = 128; // jit value
constexpr int Arch = 800; // jit value
constexpr int topK = 8; // jit value
constexpr auto mt = flashmoe::defineMLPType<0>();
using Element = flashmoe::DataType<0>::Type; // jit value
using AccumType = cuda::std::conditional_t<cuda::std::is_same_v<Element, double>, double, float>;
using Activation = flashmoe::ActivationType<AccumType, flashmoe::defineAct<1>()>::AT; // jit value
constexpr auto cm = topK > 1 ? flashmoe::CombineMode::plural : flashmoe::CombineMode::single;

// tile shapes
constexpr auto bM = flashmoe::heuristics::getMoETileM<S, Arch>();
constexpr auto tkCap = cuda::std::is_same_v<Element, double> ? 32 : (mt == flashmoe::MLPMatmulType::vanilla ? 64 : (Arch >= 900 ? 64 : 32));
constexpr auto bK0 = flashmoe::heuristics::getTileK<H, tkCap>();
constexpr auto bK1 = flashmoe::heuristics::getTileK<I, tkCap>();
constexpr auto bN0 = flashmoe::heuristics::getTileN<I, Element>();
constexpr auto bN1 = flashmoe::heuristics::getTileN<H, Element>();
constexpr auto pSK0 = flashmoe::heuristics::getPipeStages<H, bK0, Arch>();
constexpr auto pSK1 = flashmoe::heuristics::getPipeStages<I, bK1, Arch>();
constexpr auto roundEC = cute::ceil_div(EC, bM) * bM;

constexpr auto threadsGEMM0 = flashmoe::tile::suggest_thread_count<bM, bN0, bK0, Arch, Element, AccumType>();
constexpr auto threadsGEMM1 = flashmoe::tile::suggest_thread_count<bM, bN1, bK1, Arch, Element, AccumType>();
constexpr auto threads = cute::max(threadsGEMM0, threadsGEMM1, 64);

using TileGEMM0 = flashmoe::tile::CollectiveMainloop<bM, bN0, bK0, Arch, Element, AccumType, threads, pSK0>;
using TileGEMM1 = flashmoe::tile::CollectiveMainloop<bM, bN1, bK1, Arch, Element, AccumType, threads, pSK1>;

constexpr auto GEMM0Sz = cutlass::round_up(cute::max(sizeof(Element) * bK0 * pSK0 * (bM + bN0),
    sizeof(Element) * bM * bN0) + (mt == flashmoe::MLPMatmulType::gated ? sizeof(Element) * bM * bN0 : 0),
    flashmoe::MAX_ALIGNMENT);
constexpr auto GEMM1Sz = cutlass::round_up(cute::max(sizeof(Element) * bK1 * pSK1 * (bM + bN1),
  sizeof(Element) * bM * bN1), flashmoe::MAX_ALIGNMENT);

int blocks0 = -1;
int blocks1 = -1;

static void reference_initialize(const int& devId) {
  CHECK_CUDA(cudaSetDevice(devId));
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));
  auto kernel0 = gk<TileGEMM0, Activation, mt, Element, Element>;
  // set shared memory
  CHECK_CUDA(cudaFuncSetAttribute(kernel0, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM0Sz));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel0, threads, GEMM0Sz));
  blocks0 = cute::min((roundEC / bM) * (I / bN0), bps * num_sms);
  auto kernel1 = gk<TileGEMM1, cublasdx::identity, flashmoe::MLPMatmulType::vanilla, Element, Element>;
  // set shared memory
  CHECK_CUDA(cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM1Sz));
  // get blocks
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel1, threads, GEMM1Sz));
  blocks1 = cute::min((roundEC / bM) * (H / bN1), bps * num_sms);
}

static void reference_forward(
  const std::uintptr_t& tokenIds_,
  const std::uintptr_t& tokens_,
  const std::uintptr_t& ref_input_,
  const std::uintptr_t& expertUp_,
  const std::uintptr_t& expertUpV_,
  const std::uintptr_t& expertDown_,
  const std::uintptr_t& biasUp_,
  const std::uintptr_t& biasUpV_,
  const std::uintptr_t& biasDown_,
  const std::uintptr_t& ref_interim0_,
  const std::uintptr_t& ref_interim1_,
  const std::uintptr_t& expertCounts_,
  const std::uintptr_t& ref_out_,
  const float& swishAlpha,
  const float& swishBeta,
  const std::uintptr_t& stream_ptr) {
  if (blocks0 <= 0 || blocks1 <=0 ) {
    throw std::runtime_error("# of CTAs for reference kernels should be > 0");
  }
  const auto* __restrict__ tokenIds = reinterpret_cast<const flashmoe::TPS*>(tokenIds_);
  const auto* __restrict__ tokens = reinterpret_cast<const Element*>(tokens_);
  auto* __restrict__ ref_input = reinterpret_cast<Element*>(ref_input_);
  const auto* __restrict__ expertUp = reinterpret_cast<const Element*>(expertUp_);
  const auto* __restrict__ expertUpV = reinterpret_cast<const Element*>(expertUpV_);
  const auto* __restrict__ expertDown = reinterpret_cast<const Element*>(expertDown_);
  const auto* __restrict__ biasUp = reinterpret_cast<const Element*>(biasUp_);
  const auto* __restrict__ biasUpV = reinterpret_cast<const Element*>(biasUpV_);
  const auto* __restrict__ biasDown = reinterpret_cast<const Element*>(biasDown_);
  auto* __restrict__ ref_interim0 = reinterpret_cast<Element*>(ref_interim0_);
  auto* __restrict__ ref_interim1 = reinterpret_cast<Element*>(ref_interim1_);
  const auto* __restrict__ expertCounts = reinterpret_cast<const int*>(expertCounts_);
  auto* __restrict__ ref_out = reinterpret_cast<Element*>(ref_out_);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  std::vector<int> hCounts (E);
  cudaMemcpyAsync(hCounts.data(), expertCounts, sizeof(int) * E, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  constexpr auto nonAlpha = static_cast<AccumType>(1.f);
  constexpr auto nonBeta = static_cast<AccumType>(1.f);
  Element* nonV = nullptr;
  for (int i = 0; i < E; ++i) {
    const auto count = cute::min(hCounts[i], EC);
    if (count > 0) {
      // get the tokens routed to expert i
      auto* __restrict__ tIds = tokenIds + i * roundEC;
      gatherTokens<<<count, threads, 0, stream>>>(tIds, tokens, ref_input, roundEC, count, S, H);
      // now do GEMM0 + bias + act
      auto* __restrict__ expertU = expertUp + i * (static_cast<size_t>(H) * I);
      auto* __restrict__ expertUV = (mt == flashmoe::MLPMatmulType::gated ?
        expertUpV + i * (static_cast<size_t>(H) * I) : nullptr);
      auto* __restrict__ biasU = biasUp + i * I;
      auto* __restrict__ biasUV = (mt == flashmoe::MLPMatmulType::gated ? biasUpV + i * I : nullptr);
      gk<TileGEMM0, Activation, mt><<<blocks0, threads, GEMM0Sz, stream>>>
      (ref_input, expertU, expertUV, ref_interim0, biasU, biasUV, swishAlpha, swishBeta,
        roundEC, I, H);
      // do GEMM 1 + bias
      auto* __restrict__ expertD = expertDown + i * (static_cast<size_t>(H) * I);
      auto* __restrict__ biasD = biasDown + i * H;
      gk<TileGEMM1, cublasdx::identity, flashmoe::MLPMatmulType::vanilla><<<blocks1, threads, GEMM1Sz, stream>>>
      (ref_interim0, expertD, nonV, ref_interim1, biasD, nonV,
        nonAlpha, nonBeta, roundEC, H, I);
      // do combine
      combineReference<<<count, threads, 0, stream>>>(S, H, roundEC, count, topK, ref_interim1, tIds, ref_out);
    }
  }
}

PYBIND11_MODULE($mod_name, m) {
  m.def("initialize", &reference_initialize,
    py::arg("device_id"));
  m.def("forward", &reference_forward,
    py::arg("token_ids"),
    py::arg("tokens"),
    py::arg("ref_input"),
    py::arg("expert_up"),
    py::arg("expert_up_v"),
    py::arg("expert_down"),
    py::arg("bias_up"),
    py::arg("bias_up_v"),
    py::arg("bias_down"),
    py::arg("ref_interim0"),
    py::arg("ref_interim1"),
    py::arg("expert_counts"),
    py::arg("ref_out"),
    py::arg("swish_alpha") = 1.f,
    py::arg("swish_beta") = 1.f,
    py::arg("stream_ptr"));
}

struct Foo {
  int a;
  int b;
};

int main() {
  constexpr Foo foo{6, 7};
  const auto p = new Foo(foo);
  p->a += 1;
  auto q = *p;
  printf("a is %d\n", q.a);
  delete p;
}
