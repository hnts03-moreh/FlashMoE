/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_PLATFORM_ATOMIC_H
#define FLASHMOE_PLATFORM_ATOMIC_H

#include "platform.h"
#include "runtime.h"

#if defined(FLASHMOE_PLATFORM_HIP)

#include <hip/hip_runtime.h>

// -------------------------------------------------------
// HIP compatibility layer for cuda::thread_scope and cuda::atomic
// -------------------------------------------------------
// HIP does not ship cuda::atomic or cuda::thread_scope.
// We provide a minimal compatibility shim that maps to
// standard C++ atomics + HIP device intrinsics.

namespace cuda {

enum thread_scope {
    thread_scope_thread  = __HIP_MEMORY_SCOPE_SINGLETHREAD,
    thread_scope_block   = __HIP_MEMORY_SCOPE_WORKGROUP,
    thread_scope_device  = __HIP_MEMORY_SCOPE_AGENT,
    thread_scope_system  = __HIP_MEMORY_SCOPE_SYSTEM
};

// Minimal atomic_ref shim for device code.
// This provides the subset of cuda::atomic_ref used by FlashMoE.
template <typename T, thread_scope Scope = thread_scope_device>
struct atomic_ref {
    T* ptr_;

    __device__ explicit atomic_ref(T& ref) : ptr_(&ref) {}

    __device__ T load(int /* memory_order */ = 0) const {
        return __atomic_load_n(ptr_, __ATOMIC_SEQ_CST);
    }

    __device__ void store(T val, int /* memory_order */ = 0) const {
        __atomic_store_n(ptr_, val, __ATOMIC_SEQ_CST);
    }

    __device__ T fetch_add(T val, int /* memory_order */ = 0) const {
        return __atomic_fetch_add(ptr_, val, __ATOMIC_SEQ_CST);
    }

    __device__ bool compare_exchange_strong(T& expected, T desired,
                                            int /* memory_order */ = 0) const {
        return __atomic_compare_exchange_n(ptr_, &expected, desired,
                                           false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
};

// Memory order constants (matching CUDA's interface)
constexpr int memory_order_relaxed  = __ATOMIC_RELAXED;
constexpr int memory_order_acquire  = __ATOMIC_ACQUIRE;
constexpr int memory_order_release  = __ATOMIC_RELEASE;
constexpr int memory_order_acq_rel  = __ATOMIC_ACQ_REL;
constexpr int memory_order_seq_cst  = __ATOMIC_SEQ_CST;

} // namespace cuda

// -------------------------------------------------------
// Scoped atomic intrinsics
// -------------------------------------------------------
// HIP provides atomicCAS, atomicAdd etc. at device scope.
// Block-scope and system-scope variants:
//   atomicCAS_block / atomicCAS_system are supported in HIP.
//   atomicAdd_block / atomicAdd_system are supported in HIP.

#else // CUDA

#include <cuda/atomic>

#endif // FLASHMOE_PLATFORM_HIP

#endif // FLASHMOE_PLATFORM_ATOMIC_H
