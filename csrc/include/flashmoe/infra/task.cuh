//
// Created by osayamen on 1/5/26.
//

#ifndef FLASHMOE_TASK_CUH
#define FLASHMOE_TASK_CUH

#include "flashmoe/platform/platform.h"
#include "flashmoe/platform/math_compat.h"

#if defined(FLASHMOE_PLATFORM_HIP)
#include <cstddef>
#include <array>
#include <type_traits>
// Provide cuda::std::byte, cuda::std::array, cuda::std::is_same_v, etc. on HIP
namespace cuda { namespace std {
  using ::std::byte;
  using ::std::array;
  using ::std::is_trivially_copyable_v;
} }
#else
#include <cuda/type_traits>
#include <cuda/std/array>
#endif

namespace flashmoe {
    constexpr int GEMMs = 2; // within
    enum class TaskType : uint8_t {
        GEMM0,
        GEMM1,
        combine,
    };
    static_assert(cuda::std::is_same_v<cuda::std::underlying_type_t<TaskType>, uint8_t>);

    struct __align__(16) Ingredients{
        unsigned int M = 0; // GEMM0->number of tokens or Combine->tokenBatchStart
        uint16_t localExpertIdx = 0;
        uint16_t expertIdx = 0; // global
        uint16_t peerIdx = 0; // owner of the output
        uint16_t tileSize = 0; // <= BLOCK_M
        uint16_t stash = 0; // GEMM0->flagBatchIdx or Combine->tileIdx
        TaskType taskType = TaskType::GEMM0;
        uint8_t isPeerRemote = 0;

        Ingredients() = default;

        // GEMM0 and GEMM1
        __device__ __forceinline__
        Ingredients(const uint16_t& lei, const uint16_t& pei, const TaskType& tt, const uint8_t& ipr):
        localExpertIdx(lei), peerIdx(pei), taskType(tt), isPeerRemote(ipr){}

        __device__ __forceinline__
        Ingredients(const unsigned int& m, const uint16_t& lei, const uint16_t& ei,
            const uint16_t& pei, const uint16_t& ts, const uint16_t& sta,
            const TaskType& tt, const uint8_t& ipr):
        M(m), localExpertIdx(lei), expertIdx(ei), peerIdx(pei), tileSize(ts), stash(sta),
        taskType(tt), isPeerRemote(ipr){}
    };
    static_assert(sizeof(Ingredients) == 16);

    struct __align__(16) Task {
        Ingredients ingredients{};
        const cuda::std::byte* aData = nullptr;
        cuda::std::array<cuda::std::byte*, GEMMs> cData = {};
        cuda::std::byte* rcData = nullptr;
        uint64_t* flags = nullptr;
        unsigned int syncIdx = 0U;
        unsigned int tileIdx = 0U;

        Task() = default;

        // GEMM0->GEMM1
        __device__ __forceinline__
        Task(const Ingredients& ingredients_,
            const cuda::std::byte* const& aData_,
            const cuda::std::array<cuda::std::byte*, GEMMs>& cData_,
            cuda::std::byte* const& rcData_,
            uint64_t* const& flags_,
            const unsigned int& syncIdx_, const unsigned int& tile):
        ingredients(ingredients_), aData(aData_), cData(cData_), rcData(rcData_), flags(flags_),
        syncIdx(syncIdx_), tileIdx(tile){}

        // Combine
        __device__ __forceinline__
        explicit Task(const Ingredients& _ingredients):
        ingredients(_ingredients){}

        __device__ __forceinline__
        auto getTaskType() const {
            return ingredients.taskType;
        }
        __device__ __forceinline__
        auto isPeerRemote() const {
            return ingredients.isPeerRemote;
        }
        __device__ __forceinline__
        auto localExpertIdx() const {
            return ingredients.localExpertIdx;
        }
        __device__ __forceinline__
        auto M() const {
            return ingredients.M;
        }
        __device__ __forceinline__
        auto tileSize() const {
            return ingredients.tileSize;
        }
        __device__ __forceinline__
        auto flagBatchIdx() const {
            return ingredients.stash;
        }
        __device__ __forceinline__
        auto pe() const {
            return ingredients.peerIdx;
        }
        __device__ __forceinline__
        auto localPeerIdx() const {
            return ingredients.stash;
        }
        __device__ __forceinline__
        auto tokenBatchStart() const {
            return ingredients.M;
        }
        __device__ __forceinline__
        auto combineTileIdx() const{
            return ingredients.stash;
        }
        __device__ __forceinline__
        auto expertIdx() const{
            return ingredients.expertIdx;
        }
        __device__ __forceinline__
        auto epRank() const{
            return ingredients.peerIdx;
        }
    };
    static_assert(sizeof(Task) == 64);
}
#endif //FLASHMOE_TASK_CUH
