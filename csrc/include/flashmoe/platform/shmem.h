/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 *
 * SHMEM Abstraction Layer -- NVSHMEM <-> ROCSHMEM API mapping
 *
 * Include this header instead of <nvshmem.h> or <rocshmem/rocshmem.hpp> directly.
 * Platform selection is automatic via flashmoe/platform/platform.h.
 */

#ifndef FLASHMOE_PLATFORM_SHMEM_H
#define FLASHMOE_PLATFORM_SHMEM_H

#include "flashmoe/platform/platform.h"

// =====================================================================
//  Platform-specific includes
// =====================================================================

#if defined(FLASHMOE_PLATFORM_HIP)
// ------- ROCm / ROCSHMEM path -------
#  include <rocshmem/rocshmem.hpp>
#  define FLASHMOE_HAS_SHMEM 1
#else
// ------- CUDA / NVSHMEM path -------
#  include <nvshmem.h>
#  define FLASHMOE_HAS_SHMEM 1
#endif

// =====================================================================
//  Signal operation constants
// =====================================================================

#if defined(FLASHMOE_PLATFORM_HIP)
#  define SHMEM_SIGNAL_SET   rocshmem::ROCSHMEM_SIGNAL_SET
#  define SHMEM_SIGNAL_ADD   rocshmem::ROCSHMEM_SIGNAL_ADD
#else
#  define SHMEM_SIGNAL_SET   NVSHMEM_SIGNAL_SET
#  define SHMEM_SIGNAL_ADD   NVSHMEM_SIGNAL_ADD
#endif

// =====================================================================
//  Comparison constants (for wait_until)
// =====================================================================

#if defined(FLASHMOE_PLATFORM_HIP)
#  define SHMEM_CMP_EQ  rocshmem::ROCSHMEM_CMP_EQ
#  define SHMEM_CMP_NE  rocshmem::ROCSHMEM_CMP_NE
#  define SHMEM_CMP_GT  rocshmem::ROCSHMEM_CMP_GT
#  define SHMEM_CMP_GE  rocshmem::ROCSHMEM_CMP_GE
#  define SHMEM_CMP_LT  rocshmem::ROCSHMEM_CMP_LT
#  define SHMEM_CMP_LE  rocshmem::ROCSHMEM_CMP_LE
#else
#  define SHMEM_CMP_EQ  NVSHMEM_CMP_EQ
#  define SHMEM_CMP_NE  NVSHMEM_CMP_NE
#  define SHMEM_CMP_GT  NVSHMEM_CMP_GT
#  define SHMEM_CMP_GE  NVSHMEM_CMP_GE
#  define SHMEM_CMP_LT  NVSHMEM_CMP_LT
#  define SHMEM_CMP_LE  NVSHMEM_CMP_LE
#endif

// =====================================================================
//  HOST API wrappers
// =====================================================================

namespace flashmoe::shmem {

// --------------- Initialization / Finalization -----------------------

/**
 * ROCSHMEM note: rocshmem_init() uses MPI under the hood.
 * There is no direct equivalent of nvshmemx_init_status().
 * On HIP we track initialization state manually.
 */
#if defined(FLASHMOE_PLATFORM_HIP)
// ROCSHMEM does not have an init_status query.
// The user must track initialization state externally.
// We provide a simple inline helper; bootstrap.cuh sets this.
inline bool g_rocshmem_initialized = false;

__host__ inline bool is_initialized() {
  return g_rocshmem_initialized;
}

__host__ inline void init() {
  rocshmem::rocshmem_init();
  g_rocshmem_initialized = true;
}

__host__ inline void init_attr(unsigned int flags, rocshmem::rocshmem_init_attr_t *attr) {
  rocshmem::rocshmem_init_attr(flags, attr);
  g_rocshmem_initialized = true;
}

__host__ inline void finalize() {
  rocshmem::rocshmem_finalize();
  g_rocshmem_initialized = false;
}
#else
__host__ inline bool is_initialized() {
  return nvshmemx_init_status() != NVSHMEM_STATUS_NOT_INITIALIZED;
}
// NVSHMEM init/finalize are called directly by the user.
#endif

// --------------- PE queries ------------------------------------------

__host__ inline int n_pes() {
#if defined(FLASHMOE_PLATFORM_HIP)
  return rocshmem::rocshmem_n_pes();
#else
  return nvshmem_n_pes();
#endif
}

__host__ inline int my_pe() {
#if defined(FLASHMOE_PLATFORM_HIP)
  return rocshmem::rocshmem_my_pe();
#else
  return nvshmem_my_pe();
#endif
}

// --------------- Team queries ----------------------------------------

#if defined(FLASHMOE_PLATFORM_HIP)
__host__ inline int team_n_pes(rocshmem::rocshmem_team_t team) {
  return rocshmem::rocshmem_team_n_pes(team);
}
// Note: ROCSHMEM does not have a direct equivalent of NVSHMEM_TEAM_SHARED_INDEX.
// Topology detection must use an alternative method on HIP.
// See bootstrap.cuh detectTopo() for the workaround.
#else
__host__ inline int team_n_pes_shared() {
  return nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED_INDEX);
}
#endif

// --------------- Symmetric heap allocation ---------------------------

__host__ inline void* shmem_malloc(size_t size) {
#if defined(FLASHMOE_PLATFORM_HIP)
  return rocshmem::rocshmem_malloc(size);
#else
  return nvshmem_malloc(size);
#endif
}

__host__ inline void* shmem_calloc(size_t count, size_t size) {
#if defined(FLASHMOE_PLATFORM_HIP)
  // ROCSHMEM does not have rocshmem_calloc.
  // Emulate: malloc + hipMemset.
  void* ptr = rocshmem::rocshmem_malloc(count * size);
  if (ptr) {
    hipMemset(ptr, 0, count * size);
  }
  return ptr;
#else
  return nvshmem_calloc(count, size);
#endif
}

__host__ inline void shmem_free(void* ptr) {
#if defined(FLASHMOE_PLATFORM_HIP)
  rocshmem::rocshmem_free(ptr);
#else
  nvshmem_free(ptr);
#endif
}

// --------------- Remote pointer query --------------------------------

__host__ inline void* shmem_ptr(const void* dest, int pe) {
#if defined(FLASHMOE_PLATFORM_HIP)
  return rocshmem::rocshmem_ptr(dest, pe);
#else
  return nvshmem_ptr(const_cast<void*>(dest), pe);
#endif
}

// --------------- Barrier / Sync --------------------------------------

__host__ inline void barrier_all() {
#if defined(FLASHMOE_PLATFORM_HIP)
  rocshmem::rocshmem_barrier_all();
#else
  nvshmem_barrier_all();
#endif
}

__host__ inline void sync_all() {
#if defined(FLASHMOE_PLATFORM_HIP)
  rocshmem::rocshmem_sync_all();
#else
  nvshmem_sync_all();
#endif
}

} // namespace flashmoe::shmem

// =====================================================================
//  DEVICE API wrappers
// =====================================================================

namespace flashmoe::shmem::device {

// --------------- Kernel init / finalize (device-side) ----------------
// ROCSHMEM historically required rocshmem_wg_init() at kernel entry and
// rocshmem_wg_finalize() before kernel exit. These APIs are now deprecated
// (the LL_MoE reference kernels no longer call them), but we still provide
// thin wrappers so that older ROCSHMEM builds continue to work.
// On CUDA / NVSHMEM the functions are no-ops.

__device__ __forceinline__
void wg_init() {
#if defined(FLASHMOE_PLATFORM_HIP)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wdeprecated-declarations"
  rocshmem_wg_init();
#  pragma clang diagnostic pop
#endif
  // NVSHMEM has no equivalent -- no-op.
}

__device__ __forceinline__
void wg_finalize() {
#if defined(FLASHMOE_PLATFORM_HIP)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wdeprecated-declarations"
  rocshmem_wg_finalize();
#  pragma clang diagnostic pop
#endif
  // NVSHMEM has no equivalent -- no-op.
}

// --------------- putmem_signal_nbi (device-side) ---------------------
// NVSHMEM: nvshmem_putmem_signal_nbi(dest, source, bytes, sig, sigval, sigop, pe)
// ROCSHMEM: rocshmem_putmem_signal_nbi(dest, source, bytes, sig, sigval, sigop, pe)
//   Signature is identical.

__device__ __forceinline__
void putmem_signal_nbi(void* dest, const void* source, size_t nelems,
                       uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
#if defined(FLASHMOE_PLATFORM_HIP)
  rocshmem::rocshmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe);
#else
  nvshmem_putmem_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe);
#endif
}

// --------------- signal_op (device-side) -----------------------------
// NVSHMEM: nvshmemx_signal_op(sig_addr, sigval, sigop, pe)
// ROCSHMEM: No direct signal_op. Emulate via rocshmem_uint64_atomic_set.
//   For SIGNAL_SET:  rocshmem_uint64_atomic_set(sig_addr, sigval, pe)
//   For SIGNAL_ADD:  rocshmem_uint64_atomic_add(sig_addr, sigval, pe)

__device__ __forceinline__
void signal_op(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe) {
#if defined(FLASHMOE_PLATFORM_HIP)
  if (sig_op == SHMEM_SIGNAL_SET) {
    rocshmem::rocshmem_uint64_atomic_set(sig_addr, signal, pe);
  } else {
    rocshmem::rocshmem_uint64_atomic_add(sig_addr, signal, pe);
  }
#else
  nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
#endif
}

// --------------- shmem_ptr (device-side) -----------------------------

__device__ __forceinline__
void* shmem_ptr(const void* dest, int pe) {
#if defined(FLASHMOE_PLATFORM_HIP)
  return rocshmem::rocshmem_ptr(dest, pe);
#else
  return nvshmem_ptr(const_cast<void*>(dest), pe);
#endif
}

// --------------- PE queries (device-side) ----------------------------

__device__ __forceinline__
int n_pes() {
#if defined(FLASHMOE_PLATFORM_HIP)
  return rocshmem::rocshmem_n_pes();
#else
  return nvshmem_n_pes();
#endif
}

__device__ __forceinline__
int my_pe() {
#if defined(FLASHMOE_PLATFORM_HIP)
  return rocshmem::rocshmem_my_pe();
#else
  return nvshmem_my_pe();
#endif
}

// --------------- fence / quiet (device-side) -------------------------

__device__ __forceinline__
void fence() {
#if defined(FLASHMOE_PLATFORM_HIP)
  rocshmem::rocshmem_fence();
#else
  nvshmem_fence();
#endif
}

__device__ __forceinline__
void quiet() {
#if defined(FLASHMOE_PLATFORM_HIP)
  rocshmem::rocshmem_quiet();
#else
  nvshmem_quiet();
#endif
}

// --------------- signal_wait_until (device-side) ---------------------
// ROCSHMEM provides rocshmem_uint64_wait_until as the type-specific variant.

__device__ __forceinline__
void signal_wait_until(uint64_t* sig_addr, int cmp, uint64_t cmp_value) {
#if defined(FLASHMOE_PLATFORM_HIP)
  rocshmem::rocshmem_uint64_wait_until(sig_addr, cmp, cmp_value);
#else
  nvshmem_signal_wait_until(sig_addr, cmp, cmp_value);
#endif
}

// --------------- signal_fetch (device-side) --------------------------

__device__ __forceinline__
uint64_t signal_fetch(const uint64_t* sig_addr) {
#if defined(FLASHMOE_PLATFORM_HIP)
  return rocshmem::rocshmem_signal_fetch(sig_addr);
#else
  return nvshmem_signal_fetch(sig_addr);
#endif
}

// --------------- uint64_test (device-side) ---------------------------
// NVSHMEM: nvshmem_uint64_test(ptr, cmp, val) -> int (1 if condition met)
// ROCSHMEM: rocshmem_uint64_test(ptr, cmp, val) -> int

__device__ __forceinline__
int uint64_test(uint64_t* ivars, int cmp, uint64_t val) {
#if defined(FLASHMOE_PLATFORM_HIP)
  return rocshmem::rocshmem_uint64_test(ivars, cmp, val);
#else
  return nvshmem_uint64_test(ivars, cmp, val);
#endif
}

} // namespace flashmoe::shmem::device

#endif // FLASHMOE_PLATFORM_SHMEM_H
