//
// Created by osayamen on 1/17/26.
//

#ifndef FLASHMOE_BOOTSTRAP_CUH
#define FLASHMOE_BOOTSTRAP_CUH
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <cstdio>

#include "flashmoe/platform/platform.h"
#include "flashmoe/platform/runtime.h"
#include "flashmoe/platform/math_compat.h"

#if defined(FLASHMOE_PLATFORM_HIP)
#  include <hip/hip_runtime.h>
#else
#  include <cuda_runtime.h>
#  include <cuda/cmath>
#  include <cuda/memory>
#endif

// NVSHMEM / ROCSHMEM
#if defined(FLASHMOE_PLATFORM_HIP)
#  if __has_include("flashmoe/platform/shmem.h")
#    include "flashmoe/platform/shmem.h"
#  endif
#  if __has_include(<mpi.h>)
#    include <mpi.h>
#  endif
#else
#  include <nvshmem.h>
#endif

#if !defined(FLASHMOE_PLATFORM_HIP)
#  include <cute/int_tuple.hpp>
#endif

#include "infra/constants.cuh"
#include "infra/telemetry.cuh"
#include "infra/bitset.cuh"
#include "context.cuh"
#include "infra/atomics.cuh"
#include "infra/signal.cuh"
#include "infra/heap.cuh"

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

__host__ __forceinline__
  auto checkPtrAlignment(const void *const&p, const bool supports32 = false) {
  const auto alignment = supports32 ? 32 : 16;
#if defined(FLASHMOE_PLATFORM_HIP)
  if (p == nullptr || (reinterpret_cast<uintptr_t>(p) % alignment) != 0) {
    printf("Pointer is not %d-byte aligned\n", alignment);
    std::abort();
  }
#else
  if (p == nullptr || !cuda::is_aligned(p, alignment)) {
    printf("Pointer is not %d-byte aligned\n", alignment);
    cuda::std::terminate();
  }
#endif
}

namespace flashmoe {
  struct MoEArgs {
    const size_t elementBytes;
    const uint sequenceLength;
    const size_t EC;
    const uint tokenDim;
    const uint ffnIntermediateSize;
    const uint bM;
    const uint bN0;
    const uint bN1;
    const uint bK0;
    const uint bK1;
    const uint threads;
    const uint blocks; //CTAs
    const uint smemSize;
    const uint16_t epRank;
    const uint16_t epWorld;
    const uint16_t myPE; // NVSHMEM PE
    const size_t numExperts;
    const uint16_t numLocalExperts;
    const Topology topo;

    MoEArgs(const size_t &eb, const uint &S, const uint &H, const uint &I, const size_t &ec,
            const uint &bm, const uint &bn0, const uint &bn1, const uint &bk0, const uint bk1,
            const uint &_threads, const uint &ctas, const uint& sharedSize, const uint16_t &ep_rank, const uint16_t &ep_world,
            const uint16_t &mype, const uint16_t &experts,
            const uint16_t &nlx, const Topology &topo_) : elementBytes(eb), sequenceLength(S),
                                                          EC(ec),
                                                          tokenDim(H), ffnIntermediateSize(I), bM(bm), bN0(bn0),
                                                          bN1(bn1), bK0(bk0), bK1(bk1),
                                                          threads(_threads), blocks(ctas), smemSize(sharedSize), epRank(ep_rank),
                                                          epWorld(ep_world), myPE(mype), numExperts(experts),
                                                          numLocalExperts(nlx), topo(topo_) {
    }
  };

  template<int subscriberWarpSize>
  __host__ __device__ __forceinline__
  constexpr auto subscriberTQLength(const int &world, const uint &numLocalExperts, const uint &ecTilesM,
                                    const uint &E, const uint &tilesN0, const uint &tilesN1,
                                    const uint &subscriberCount) {
    const auto dispatchTaskQL = cute::ceil_div(world * numLocalExperts, subscriberCount / subscriberWarpSize) *
                                (cute::ceil_div(ecTilesM * tilesN0, subscriberWarpSize) + cute::ceil_div(
                                   tilesN0, subscriberWarpSize));
    const auto combineTaskQL = (cute::ceil_div(ecTilesM * E, subscriberCount) * tilesN1) +
                               cute::ceil_div(ecTilesM * E * tilesN1, subscriberCount);
    return static_cast<size_t>(dispatchTaskQL + combineTaskQL) * subscriberCount;
  }

  __host__ __device__ __forceinline__
  auto secondaryTQLength(const int &world, const int &numLocalExperts, const uint &ecTilesM, const uint &tilesN1) {
    return world * numLocalExperts * ecTilesM * tilesN1;
  }

  template<int subscriberCount, int subscriberWarpSize>
  __device__ __forceinline__
  constexpr auto subscriberTQLength(const int &world, const int &numLocalExperts, const uint &ecTilesM,
                                    const uint &E, const uint &tilesN0, const uint &tilesN1) {
    static_assert(subscriberCount % subscriberWarpSize == 0);
    return subscriberTQLength<subscriberWarpSize>(world, numLocalExperts, ecTilesM, E, tilesN0, tilesN1,
                                                  subscriberCount);
  }

#if !defined(FLASHMOE_PLATFORM_HIP)
  // cuda::barrier is not available on HIP
  __global__ void bI(cuda::barrier<cuda::thread_scope_device> *db, const uint blocks) {
    init(db, blocks);
  }
#endif

  __host__ __forceinline__
  void expertParallelBookkeeping(const int *__restrict__ const&expertToEpRank,
                                 const int *__restrict__ const&epRankToGlobalRank, const uint &epWorld,
                                 const int &myPE, const uint &E, const uint &nLx,
                                 cuda::std::byte *const&sHeap, uint64_t *const&signals,
                                 PEL *const&pel, PLI *const&pli, ELI *const&eli, LXI *const&lxi,
                                 gpuStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
    const flashmoeRange range{"FlashMoE::expertParallelBookkeeping"};
#endif
    if (!flashmoe::shmem::is_initialized()) {
      throw std::runtime_error("shmem is not initialized");
    }
    std::vector<uint> lxIndices(epWorld);
    std::vector<PEL> pelHost(E);
    std::vector<PLI> pliHost(epWorld);
    std::vector<ELI> eliHost(E);
    std::vector<LXI> lxiHost(nLx);

    std::ranges::fill(lxIndices.begin(), lxIndices.end(), 0);
    for (uint i = 0; i < E; ++i) {
      const auto epRank = expertToEpRank[i];
      const auto pe = epRankToGlobalRank[epRank];
      auto *rSheap = static_cast<cuda::std::byte *>(flashmoe::shmem::shmem_ptr(sHeap, pe));
      auto *rFlags = static_cast<uint64_t *>(flashmoe::shmem::shmem_ptr(signals, pe));
      const uint lxIdx = lxIndices[epRank]++;
      const auto isRemote = rSheap == nullptr;

      // PEL
      pelHost[i].isRemote = isRemote;
      pelHost[i].expertLocalIdx = lxIdx;
      pelHost[i].pe = pe;
      pelHost[i].remoteSFlags = rFlags;
      pelHost[i].remoteSHeap = rSheap;
      pelHost[i].peer = epRank;

      // ELI
      eliHost[i].epRank = epRank;
      eliHost[i].isRemote = isRemote;
      eliHost[i].localExpertIndex = lxIdx;

      // PLI
      pliHost[epRank].isRemote = isRemote;
      pliHost[epRank].pe = pe;
      pliHost[epRank].remoteSFlags = rFlags;
      pliHost[epRank].remoteSHeap = rSheap;

      //LXI
      if (pe == myPE) {
        lxiHost[lxIdx].expertIndex = i;
      }
    }

    const auto nlxUniform = lxIndices[0];
    for (uint i = 0; i < E; ++i) {
      auto pt = pelHost[i];
      pt.nLocalExperts = lxIndices[pt.peer];
      if (pt.nLocalExperts != nlxUniform) {
        // may relax this later
        throw std::runtime_error("Number of local experts should be equal across the ep group");
      }
      pelHost[i] = pt;
    }

    gpuMemcpyAsync(pel, pelHost.data(), sizeof(PEL) * pelHost.size(), gpuMemcpyHostToDevice, stream);
    gpuMemcpyAsync(pli, pliHost.data(), sizeof(PLI) * pliHost.size(), gpuMemcpyHostToDevice, stream);
    gpuMemcpyAsync(eli, eliHost.data(), sizeof(ELI) * eliHost.size(), gpuMemcpyHostToDevice, stream);
    gpuMemcpyAsync(lxi, lxiHost.data(), sizeof(LXI) * lxiHost.size(), gpuMemcpyHostToDevice, stream);
  }

  __host__ __forceinline__
  Topology detectTopo() {
    if (!flashmoe::shmem::is_initialized()) {
      throw std::runtime_error("shmem is not initialized");
    }
#if defined(FLASHMOE_PLATFORM_HIP)
    // ROCSHMEM does not have NVSHMEM_TEAM_SHARED_INDEX.
    // Use MPI_Comm_split_type(SHARED) to count node-local PEs.
    // If all PEs are on the same node → XGMI_ONLY (equivalent to NVLINK_ONLY).
    const char* env_topo = std::getenv("FLASHMOE_TOPO");
    if (env_topo) {
      if (std::string(env_topo) == "nvlink" || std::string(env_topo) == "xgmi")
        return Topology::NVLINK_ONLY;
      return Topology::MIXED;
    }
#if defined(MPI_VERSION)
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
      MPI_Comm local_comm;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
      int local_size = 0;
      MPI_Comm_size(local_comm, &local_size);
      MPI_Comm_free(&local_comm);
      int world_size = flashmoe::shmem::n_pes();
      return (local_size == world_size) ? Topology::NVLINK_ONLY : Topology::MIXED;
    }
    return Topology::MIXED;
#else
    // No MPI available — conservatively assume MIXED
    return Topology::MIXED;
#endif
#else
    return nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED_INDEX) == nvshmem_n_pes() ? Topology::NVLINK_ONLY : Topology::MIXED;
#endif
  }

  __host__ __forceinline__
  Context initialize(const MoEArgs &args, const int &arch, const int *__restrict__ const& expertToEpRank,
                     const int *__restrict__ const&epRankToGlobalRank, gpuStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
    const flashmoeRange range{"FlashMoE::initialize"};
#endif
    if (args.tokenDim % args.bK0 != 0 || args.tokenDim % args.bN1 != 0) {
      throw std::runtime_error("token dimension should be multiples of tile dimensions");
    }
    if (args.ffnIntermediateSize % args.bN0 != 0 || args.ffnIntermediateSize % args.bK1 != 0) {
      throw std::runtime_error("Intermediate size should be multiples of tile dimensions");
    }
    if (args.blocks < 2) {
      throw std::runtime_error("blocks must be at least 2");
    }
    const auto processors = args.blocks - 1;
    if (processors > scheduler::MAX_PROCESSORS) {
      const auto errmsg = std::string("processor count: ").append(std::to_string(processors))
          .append(" is too high");
      throw std::runtime_error(errmsg);
    }
    // maximum tiles that a peer will send to another peer in aggregate.
    const auto maxPeerTaskTiles = cute::ceil_div(args.EC, args.bM) * args.numLocalExperts;
    if (maxPeerTaskTiles > cuda::std::numeric_limits<uint16_t>::max()) {
      throw std::runtime_error("Max peer task tiles exceeds supported limit. Inform the maintainer.");
    }
    const auto roundEC = cute::ceil_div(args.EC, args.bM) * args.bM;
    const auto ecTilesM = cute::ceil_div(roundEC, args.bM);
    const auto tilesN0 = cute::ceil_div(args.ffnIntermediateSize, args.bN0);
    const auto tilesN1 = cute::ceil_div(args.tokenDim, args.bN1);

    if (!flashmoe::shmem::is_initialized()) {
      throw std::runtime_error("shmem is not initialized");
    }
    const bool elementBytesConditions = cutlass::ispow2(args.elementBytes) &&
                                        (args.elementBytes == 2 || args.elementBytes == 4 || args.elementBytes == 8);
    if (!elementBytesConditions) {
      throw std::runtime_error("elementBytes not supported");
    }
    // signals ~= tiles(S) * tiles(H)
    // below ensures the following calloc initializes our signals to the expected value 0
    static_assert(SignalConstants::ground == 0);
    const size_t signalLength = (args.epWorld * args.numLocalExperts) + (args.numExperts * ecTilesM * tilesN1);
    auto *signals = static_cast<uint64_t *>(flashmoe::shmem::shmem_calloc(signalLength, sizeof(uint64_t)));
    if (signals == nullptr) {
      throw std::runtime_error("failed to allocate signals via SHMEM");
    }
    // symmetric heap ~= 4*S*H
    const auto heapLength = args.elementBytes * HEAP_STAGES * HEAP_CELLS * static_cast<size_t>(
                              args.epWorld * args.numLocalExperts) * static_cast<size_t>(roundEC * args.tokenDim);
    auto *sHeap = static_cast<cuda::std::byte *>(flashmoe::shmem::shmem_malloc(heapLength));
    if (sHeap == nullptr) {
      throw std::runtime_error("failed to allocate heap via SHMEM");
    }
    const auto supports32 = arch >= 1000;
    checkPtrAlignment(sHeap, supports32);

    Task *tQ = nullptr;
    const bool threadConditions = args.threads >= WARP_SIZE * 2 && args.threads % WARP_SIZE == 0;
    if (!threadConditions) {
      throw std::runtime_error("threads not supported");
    }
    const auto subscriberCount = args.threads - WARP_SIZE;
    const size_t tQLength = subscriberTQLength<WARP_SIZE>(args.epWorld, args.numLocalExperts, ecTilesM, args.numExperts,
                                                          tilesN0,
                                                          tilesN1, args.threads - WARP_SIZE);
    const size_t secondaryTQL = secondaryTQLength(args.epWorld, args.numLocalExperts, ecTilesM, tilesN1);
    CHECK_CUDA(gpuMallocAsync(&tQ, sizeof(Task) * (tQLength + secondaryTQL), stream));
    Task *pTq = tQ + tQLength;
    if (tQLength + secondaryTQL >= cuda::std::numeric_limits<uint>::max()) {
      throw std::runtime_error("Task Queue length should be < UINT32_MAX. Not an error: need to migrate to uint64");
    }
    const size_t gRQIdxMax = (args.numLocalExperts * args.epWorld * ecTilesM * (tilesN0 + tilesN1)) +
                             (cute::ceil_div(roundEC, args.bM) * tilesN1) + (cute::ceil_div(
                               processors, scheduler::SCHEDULER_COUNT));
    if (gRQIdxMax >= cuda::std::numeric_limits<uint>::max()) {
      // catches overflow in scheduler. See circularIdx function
      throw std::runtime_error("gRQIdxMax should be < UINT32_MAX. Not an error: need to migrate to uint64");
    }
    checkPtrAlignment(tQ);
    checkPtrAlignment(pTq);

    cuda::std::byte *GEMM0Staging = nullptr;
    const size_t stagingLength = static_cast<size_t>(args.epWorld * args.numLocalExperts * roundEC) * args.
                                 ffnIntermediateSize;
    CHECK_CUDA(gpuMallocAsync(&GEMM0Staging, stagingLength * args.elementBytes, stream));

    BitSet *consumerBitMap = nullptr;
    const auto cbmLength = nSI(args.numExperts * ecTilesM * tilesN1, subscriberCount);
    CHECK_CUDA(gpuMallocAsync(&consumerBitMap, sizeof(BitSet) * cbmLength, stream));
    CHECK_CUDA(gpuMemsetAsync(consumerBitMap, 0, sizeof(BitSet) * cbmLength, stream));

    uint8_t *producerBitMap = nullptr;
    const auto pbmLength = args.epWorld * args.numLocalExperts * ecTilesM * tilesN1;
    CHECK_CUDA(gpuMallocAsync(&producerBitMap, sizeof(uint8_t) * pbmLength, stream));
    CHECK_CUDA(gpuMemsetAsync(producerBitMap, 0, sizeof(uint8_t) * pbmLength, stream));

    PEL *pel = nullptr;
    CHECK_CUDA(gpuMallocAsync(&pel, sizeof(PEL) * args.numExperts, stream));

    PLI *pli = nullptr;
    CHECK_CUDA(gpuMallocAsync(&pli, sizeof(PLI) * args.epWorld, stream));

    ELI *eli = nullptr;
    CHECK_CUDA(gpuMallocAsync(&eli, sizeof(ELI) * args.numExperts, stream));

    LXI *lxi = nullptr;
    CHECK_CUDA(gpuMallocAsync(&lxi, sizeof(LXI) * args.numLocalExperts, stream));

    TPS *tps = nullptr;
    CHECK_CUDA(gpuMallocAsync(&tps, sizeof(TPS) * args.numExperts * roundEC, stream));

    TQSignal *tqs = nullptr;
    CHECK_CUDA(gpuMallocAsync(&tqs, sizeof(TQSignal) * processors, stream));
    CHECK_CUDA(gpuMemsetAsync(tqs, 0, sizeof(TQSignal) * processors, stream));

    uint *dispatchSync = nullptr;
    CHECK_CUDA(gpuMallocAsync(&dispatchSync, sizeof(uint) * args.numExperts, stream));
    CHECK_CUDA(gpuMemsetAsync(dispatchSync, 0, sizeof(uint) * args.numExperts, stream));

    uint *gtqHeads = nullptr;
    // ~= tiles(S)
    const size_t gtqHeadsLength = args.epWorld * args.numLocalExperts * ecTilesM;
    CHECK_CUDA(gpuMallocAsync(&gtqHeads, sizeof(uint) * gtqHeadsLength, stream));
    CHECK_CUDA(gpuMemsetAsync(gtqHeads, 0, sizeof(uint) * gtqHeadsLength, stream));

    uint *tileSync = nullptr;
    CHECK_CUDA(gpuMallocAsync(&tileSync, sizeof(uint) * gtqHeadsLength, stream));
    CHECK_CUDA(gpuMemsetAsync(tileSync, 0, sizeof(uint) * gtqHeadsLength, stream));

    uint *statusQ = nullptr;
    CHECK_CUDA(gpuMallocAsync(&statusQ, sizeof(uint) * processors, stream));
    static_assert(ReadySignal::observed == 0);
    CHECK_CUDA(gpuMemsetAsync(statusQ, 0, sizeof(uint) * processors, stream));

    uint8_t* stateNumbers = nullptr;
    CHECK_CUDA(gpuMallocAsync(&stateNumbers, sizeof(uint8_t) * args.blocks, stream));
    static_assert(SignalConstants::sequenceStart == 0x01);
    CHECK_CUDA(gpuMemsetAsync(stateNumbers, 0x01, sizeof(uint8_t) * args.blocks, stream));

    CHECK_CUDA(gpuPeekAtLastError());
    expertParallelBookkeeping(expertToEpRank, epRankToGlobalRank, args.epWorld, args.myPE,
                              args.numExperts, args.numLocalExperts, sHeap, signals, pel, pli, eli, lxi, stream);

    return Context{
      .symHeap = sHeap,
      .signals = signals,
      .tQ = tQ,
      .pTq = pTq,
      .GEMM0Staging = GEMM0Staging,
      .consumerCombineBitMap = consumerBitMap,
      .producerCombineBitMap = producerBitMap,
      .pel = pel,
      .pli = pli,
      .eli = eli, .lxi = lxi, .tqs = tqs,
      .dispatchSync = dispatchSync,
      .gTqHeads = gtqHeads,
      .tileSync = tileSync,
      .statusQueue = statusQ,
      .tokenIndices = tps,
      .stateNumbers = stateNumbers,
      .processors_v = cuda::fast_mod_div(processors),
      .blocks = args.blocks,
      .smemSize = args.smemSize,
      .S = args.sequenceLength,
      .H = args.tokenDim,
      .I = args.ffnIntermediateSize,
      .EC = static_cast<uint>(args.EC),
      .bM = static_cast<uint16_t>(args.bM),
      .bN0 = static_cast<uint16_t>(args.bN0),
      .bN1 = static_cast<uint16_t>(args.bN1),
      .nLx = args.numLocalExperts,
      .E = static_cast<uint16_t>(args.numExperts),
      .world = args.epWorld,
      .epRank = args.epRank,
      .myPE = args.myPE,
      .topo = args.topo
    };
  }

  // profiling purposes
  __host__ __forceinline__ size_t getWorkspaceBytes(const MoEArgs &args) {
    const auto roundEC = cute::ceil_div(args.EC, args.bM) * args.bM;
    const auto ecTilesM = cute::ceil_div(roundEC, args.bM);
    const auto tilesN0 = cute::ceil_div(args.ffnIntermediateSize, args.bN0);
    const auto tilesN1 = cute::ceil_div(args.tokenDim, args.bN1);
    const auto subscriberCount = args.threads - WARP_SIZE;
    const auto processors = args.blocks - 1;

    size_t bytes = 0;
    bytes += sizeof(uint64_t) * (args.epWorld * args.numLocalExperts + (args.numExperts * ecTilesM * tilesN1));
    bytes += args.elementBytes * HEAP_STAGES * HEAP_CELLS * args.epWorld * args.numLocalExperts * roundEC * args.
        tokenDim;
    const size_t tQLength = subscriberTQLength<WARP_SIZE>(args.epWorld, args.numLocalExperts, ecTilesM, args.numExperts,
                                                          tilesN0,
                                                          tilesN1, args.threads - WARP_SIZE);
    const size_t secondaryTQL = secondaryTQLength(args.epWorld, args.numLocalExperts, ecTilesM, tilesN1);
    bytes += sizeof(Task) * (tQLength + secondaryTQL);
    bytes += args.elementBytes * static_cast<size_t>(args.epWorld * args.numLocalExperts * roundEC) * args.
        ffnIntermediateSize;
    bytes += sizeof(BitSet) * nSI(args.numExperts * ecTilesM * tilesN1, subscriberCount);
    bytes += sizeof(uint8_t) * args.epWorld * args.numLocalExperts * ecTilesM * tilesN1;
    bytes += sizeof(PEL) * args.numExperts;
    bytes += sizeof(PLI) * args.epWorld;
    bytes += sizeof(ELI) * args.numExperts;
    bytes += sizeof(LXI) * args.numLocalExperts;
    bytes += sizeof(TPS) * args.numExperts * roundEC;
    bytes += sizeof(TQSignal) * processors;
    bytes += sizeof(uint) * args.numExperts;
    bytes += sizeof(uint) * args.epWorld * args.numLocalExperts * ecTilesM;
    bytes += sizeof(uint) * args.epWorld * args.numLocalExperts * ecTilesM;
    bytes += sizeof(uint) * processors;
    bytes += sizeof(uint8_t) * args.blocks;
    return bytes;
  }

  __host__ __forceinline__
  GateContext initializeGate(const uint &bNGate, const uint &numExperts, const uint &S, gpuStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
    const flashmoeRange range{"FlashMoE::initializeGate"};
#endif
    int *ecGuards = nullptr;
    CHECK_CUDA(gpuMallocAsync(&ecGuards, sizeof(int) * numExperts, stream));
    CHECK_CUDA(gpuMemsetAsync(ecGuards, flashmoe::STALE_AS_BYTE, sizeof(int) * numExperts, stream));
    SoftmaxStatePacked *ssp = nullptr;
    RingTopKPayload *rtp = nullptr;
    if (numExperts > bNGate) {
      const auto tE = cute::ceil_div(numExperts, bNGate);
      CHECK_CUDA(gpuMallocAsync(&ssp, sizeof(SoftmaxStatePacked) * S * tE, stream));
      CHECK_CUDA(gpuMemsetAsync(ssp, 0, sizeof(SoftmaxStatePacked) * S * tE, stream));

      CHECK_CUDA(gpuMallocAsync(&rtp, 2 * sizeof(RingTopKPayload) * S * tE, stream));
      CHECK_CUDA(gpuMemsetAsync(rtp, 0, 2 * sizeof(RingTopKPayload) * S * tE, stream));
    }
    return GateContext{ecGuards, ssp, rtp};
  }

  __host__ __forceinline__
  void finalizeGate(const GateContext &gCtx, gpuStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
    const flashmoeRange range{"FlashMoE::finalizeGate"};
#endif
    CHECK_CUDA(gpuFreeAsync(gCtx.ecGuards, stream));
    if (gCtx.ssp != nullptr) {
      CHECK_CUDA(gpuFreeAsync(gCtx.ssp, stream));
    }
    if (gCtx.rtp != nullptr) {
      CHECK_CUDA(gpuFreeAsync(gCtx.rtp, stream));
    }
    CHECK_CUDA(gpuPeekAtLastError());
    CHECK_CUDA(gpuStreamSynchronize(stream));
  }

  __host__ __forceinline__
  void finalize(const Context &ctx, gpuStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
    const flashmoeRange range{"FlashMoE::finalize"};
#endif
    // free workspace memory
    CHECK_CUDA(gpuFreeAsync(ctx.tQ, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.stateNumbers, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.GEMM0Staging, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.pel, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.pli, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.eli, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.lxi, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.consumerCombineBitMap, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.producerCombineBitMap, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.tokenIndices, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.tqs, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.dispatchSync, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.gTqHeads, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.tileSync, stream));
    CHECK_CUDA(gpuFreeAsync(ctx.statusQueue, stream));
    CHECK_CUDA(gpuStreamSynchronize(stream));
    flashmoe::shmem::shmem_free(ctx.symHeap);
    flashmoe::shmem::shmem_free(ctx.signals);
    CHECK_CUDA(gpuPeekAtLastError());
  }
}
#endif //FLASHMOE_BOOTSTRAP_CUH
