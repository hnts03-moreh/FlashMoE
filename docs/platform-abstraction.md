# Platform Abstraction Layer

## Overview

The platform abstraction layer enables FlashMoE to compile for both CUDA and ROCm/HIP from a single codebase. All abstractions resolve at compile time -- there is no runtime overhead, no virtual dispatch, no branching on platform at runtime.

The layer lives in `csrc/include/flashmoe/platform/` and consists of 8 header files. On CUDA builds, these headers are thin pass-throughs (or not included at all, in the case of `math_compat.h`). On HIP builds, they provide the mappings and shims that make CUDA-oriented code compile with the HIP toolchain.

## Directory Structure

```
csrc/include/flashmoe/platform/
  platform.h        Platform detection, warp size, lane mask type
  runtime.h         Runtime API type aliases and macros
  device.h          Device qualifiers, half/bfloat16 types
  atomic.h          cuda::atomic_ref and cuda::thread_scope shim
  intrinsics.h      Warp/wavefront shuffle, ballot, popc, ffs
  math_compat.h     CUTLASS/CuTe/CCCL stubs (HIP only)
  profiling.h       NVTX3 / rocTX scoped ranges
  shmem.h           NVSHMEM / ROCSHMEM API abstraction
```

## Header Reference

### platform.h

**Purpose:** Platform detection macros, warp/wavefront sizing, architecture identification.

**Key definitions:**

```cpp
// Auto-detected from compiler macros, or user-defined before include
FLASHMOE_PLATFORM_CUDA  // Defined to 1 on CUDA
FLASHMOE_PLATFORM_HIP   // Defined to 1 on HIP

// Warp/wavefront size
FLASHMOE_WARP_SIZE       // 32 on CUDA, 64 on HIP

// Lane mask type (ballot return type)
flashmoe_lane_mask_t     // uint32_t on CUDA, uint64_t on HIP
FLASHMOE_LANE_MASK_TYPE  // Same as above, as a macro
FLASHMOE_FULL_LANE_MASK  // 0xFFFFFFFF on CUDA, ~uint64_t(0) on HIP

// Architecture detection
FLASHMOE_ARCH_MI300X     // Defined on gfx942
FLASHMOE_ARCH_MI300A     // Defined on gfx940/941
FLASHMOE_ARCH_MI250      // Defined on gfx90a
FLASHMOE_ARCH_IS_CDNA    // Defined on all AMD CDNA architectures
FLASHMOE_ARCH_IS_HOPPER  // Defined on CUDA arch >= 900
FLASHMOE_ARCH_IS_AMPERE  // Defined on CUDA arch >= 800
```

**Include this header** when you need platform detection or warp size constants.

### runtime.h

**Purpose:** Maps CUDA runtime API calls to HIP equivalents via macros and type aliases.

Includes `platform.h` automatically.

**Type aliases:**

```cpp
gpuStream_t      // cudaStream_t or hipStream_t
gpuEvent_t       // cudaEvent_t or hipEvent_t
gpuError_t       // cudaError_t or hipError_t
GPU_SUCCESS      // cudaSuccess or hipSuccess
```

**Macro mappings (excerpt):**

| Abstraction | CUDA | HIP |
|-------------|------|-----|
| `gpuMalloc` | `cudaMalloc` | `hipMalloc` |
| `gpuFree` | `cudaFree` | `hipFree` |
| `gpuMemcpyAsync` | `cudaMemcpyAsync` | `hipMemcpyAsync` |
| `gpuMemsetAsync` | `cudaMemsetAsync` | `hipMemsetAsync` |
| `gpuSetDevice` | `cudaSetDevice` | `hipSetDevice` |
| `gpuStreamSynchronize` | `cudaStreamSynchronize` | `hipStreamSynchronize` |
| `gpuDeviceGetAttribute` | `cudaDeviceGetAttribute` | `hipDeviceGetAttribute` |
| `gpuFuncSetAttribute` | `cudaFuncSetAttribute` | `hipFuncSetAttribute` |
| `gpuOccupancyMaxActiveBlocksPerMultiprocessor` | `cudaOccupancy...` | `hipOccupancy...` |

**Error checking:**

```cpp
CHECK_GPU(gpuMalloc(&ptr, size));  // Prints file:line and error on failure
CHECK_CUDA(expr);                  // Legacy alias, same as CHECK_GPU
```

### device.h

**Purpose:** Device function qualifiers, `__grid_constant__` compatibility, half-precision type headers.

**Key behavior:**

- `__device__`, `__global__`, `__host__`, `__shared__`, `__constant__`, `__forceinline__`, `__launch_bounds__` are identical on both platforms -- no mapping needed.
- `__grid_constant__` is defined as empty on HIP (and on CUDA < 12).
- `__nv_bfloat16` is aliased to `hip_bfloat16` on HIP.
- Includes `hip/hip_fp16.h` and `hip/hip_bfloat16.h` on HIP (vs `cuda_fp16.h` and `cuda_bf16.h` on CUDA).

### atomic.h

**Purpose:** Provides `cuda::thread_scope` and `cuda::atomic_ref` on HIP using GCC built-in atomics.

On CUDA, this header simply includes `<cuda/atomic>`.

**HIP shim API:**

```cpp
namespace cuda {

enum thread_scope {
    thread_scope_thread,
    thread_scope_block,
    thread_scope_device,
    thread_scope_system
};

template <typename T, thread_scope Scope = thread_scope_device>
struct atomic_ref {
    explicit atomic_ref(T& ref);
    T load(int memory_order = 0) const;
    void store(T val, int memory_order = 0);
    T fetch_add(T val, int memory_order = 0);
    bool compare_exchange_strong(T& expected, T desired, int memory_order = 0);
};

// Memory order constants
constexpr int memory_order_relaxed;
constexpr int memory_order_acquire;
constexpr int memory_order_release;
constexpr int memory_order_acq_rel;
constexpr int memory_order_seq_cst;

} // namespace cuda
```

### intrinsics.h

**Purpose:** Maps CUDA warp primitives to HIP wavefront primitives.

**Shuffle operations** -- CUDA takes a mask argument that HIP does not:

```cpp
// These macros strip the mask on HIP, keeping the CUDA call signature:
__shfl_sync(mask, val, srcLane, width)      -> __shfl(val, srcLane, width)
__shfl_up_sync(mask, val, delta, width)     -> __shfl_up(val, delta, width)
__shfl_down_sync(mask, val, delta, width)   -> __shfl_down(val, delta, width)
__shfl_xor_sync(mask, val, laneMask, width) -> __shfl_xor(val, laneMask, width)
```

**Ballot** -- returns `uint64_t` on HIP (wavefront-64):

```cpp
__ballot_sync(mask, predicate)  -> __ballot(predicate)  // 64-bit result
```

**Synchronization:**

```cpp
__syncwarp(mask)  ->  __builtin_amdgcn_wave_barrier()  // HIP wavefronts are lockstep
__activemask()    ->  __ballot(1)
```

**Width-aware bit operations:**

```cpp
int flashmoe_popc(flashmoe_lane_mask_t mask);  // __popc (32-bit) on CUDA, __popcll (64-bit) on HIP
int flashmoe_ffs(flashmoe_lane_mask_t mask);    // __ffs on CUDA, __ffsll on HIP
```

### math_compat.h

**Purpose:** Provides stub implementations of CUTLASS, CuTe, and CCCL (`cuda::std::`) utilities. **Only active on HIP builds** -- guarded by `#if defined(FLASHMOE_PLATFORM_HIP)`.

On CUDA builds, the real CUTLASS/CuTe/CCCL libraries are used. This header is never included on CUDA.

**`cuda::std::` namespace (mapped to `std::`):**

- `is_same_v`, `conditional_t`, `enable_if_t`, `decay_t`, `integral_constant`
- `is_trivially_copyable_v`, `is_invocable_r_v`
- `byte`, `bit_cast`, `numeric_limits`, `terminate`
- `min(a, b)`, `max(a, b)` (host+device)
- `ignore` (struct with template `operator=`)

**`cuda::` namespace utilities:**

- `round_up(x, m)`, `ceil_div(x, m)`, `round_down(x, m)`, `is_aligned(ptr, alignment)`

**`cute::` namespace:**

- `Int<N>`, `C<N>` -- compile-time integer constants
- `Shape<Ts...>` -- variadic shape type with `get<I>` support (2, 3, 4 elements)
- `Coord<Ts...>` -- coordinate type with `make_coord`, `select`, `get`
- `ceil_div`, `min`, `max` (2 and 3 arg)
- `LayoutRight`, `LayoutLeft` -- layout tags
- `is_rmem_v`, `rank_v`, `is_tuple_v` -- type traits
- `_0`, `_1` -- integer constant types
- `make_int_sequence<N>`, `for_each` -- compile-time loops
- `half_t` -- half-precision wrapper

**`cutlass::` namespace:**

- `round_up(x, m)`
- `AlignedArray<T, N, Alignment>` -- aligned fixed-size array
- `Array<T, N>` -- fixed-size array
- `NumericConverter<To, From>` -- type conversion functor

**Additional shims added in kernel files:**

- `cutlass::is_pow2<N>` (in `infra/vt.cuh`)
- `cutlass::ispow2(n)` (in `infra/vt.cuh`)
- `cuda::fast_mod_div<T>` (in `context.cuh`)
- `hip_compat::ReLU`, `hip_compat::GELU`, `hip_compat::SiLu` (in `infra/activation.cuh`)

### profiling.h

**Purpose:** Maps NVTX3 tracing API to rocTX.

**Scoped range (RAII):**

```cpp
// CUDA: wraps nvtx3::scoped_range_in<flashmoeDomain>
// HIP:  wraps roctxRangePush/Pop
flashmoe::flashmoeRange range("my_operation");
// range pops automatically at scope exit
```

**Push/pop macros:**

```cpp
FLASHMOE_RANGE_PUSH("label");  // nvtxRangePushA or roctxRangePush
FLASHMOE_RANGE_POP();          // nvtxRangePop or roctxRangePop
FLASHMOE_MARK("event");        // nvtxMarkA or roctxMark
```

### shmem.h

**Purpose:** Abstracts NVSHMEM and ROCSHMEM for distributed GPU communication.

Provides two namespaces:
- `flashmoe::shmem` -- host-side API wrappers
- `flashmoe::shmem::device` -- device-side API wrappers

**Signal constants:**

```cpp
SHMEM_SIGNAL_SET  // NVSHMEM_SIGNAL_SET or ROCSHMEM_SIGNAL_SET
SHMEM_SIGNAL_ADD  // NVSHMEM_SIGNAL_ADD or ROCSHMEM_SIGNAL_ADD
SHMEM_CMP_EQ / SHMEM_CMP_NE / SHMEM_CMP_GT / SHMEM_CMP_GE / SHMEM_CMP_LT / SHMEM_CMP_LE
```

**Host API (`flashmoe::shmem`):**

```cpp
bool is_initialized();                      // ROCSHMEM: tracked manually; NVSHMEM: init_status query
void init();                                // ROCSHMEM only (NVSHMEM init called directly)
int n_pes();
int my_pe();
void* shmem_malloc(size_t size);
void* shmem_calloc(size_t count, size_t size);  // ROCSHMEM: emulated via malloc+hipMemset
void shmem_free(void* ptr);
void* shmem_ptr(const void* dest, int pe);
void barrier_all();
void sync_all();
```

**Device API (`flashmoe::shmem::device`):**

```cpp
void wg_init();           // ROCSHMEM workgroup init (no-op on CUDA)
void wg_finalize();       // ROCSHMEM workgroup finalize (no-op on CUDA)
void putmem_signal_nbi(dest, source, nelems, sig_addr, signal, sig_op, pe);
void signal_op(sig_addr, signal, sig_op, pe);  // ROCSHMEM: emulated via atomic_set/add
void* shmem_ptr(dest, pe);
int n_pes();
int my_pe();
void fence();
void quiet();
void signal_wait_until(sig_addr, cmp, cmp_value);
uint64_t signal_fetch(sig_addr);
int uint64_test(ivars, cmp, val);
```

## Architecture Differences Summary

| Property | NVIDIA (H100) | AMD (MI300X) |
|----------|---------------|--------------|
| Thread group | Warp (32 threads) | Wavefront (64 threads) |
| Ballot return type | `uint32_t` | `uint64_t` |
| Matrix instruction | Tensor Core | MFMA |
| Shared memory | Up to 228 KB SMEM | 64 KB LDS per CU |
| Async copy | `cp.async` (hardware) | Synchronous (software) |
| ISA | PTX / SASS | GCN / CDNA ISA |
| RDC flag | `--relocatable-device-code=true` | `-fgpu-rdc` |
| Sub-warp divergence | Yes (independent threads) | No (SIMD lockstep) |
| `__syncwarp` | Required for correctness | No-op (wave barrier) |

## Guidelines for New Files

When adding new `.cuh` or `.cu` files to FlashMoE:

1. **Include platform headers at the top.** At minimum, include `platform/platform.h`. Include additional headers as needed:

```cpp
#include "flashmoe/platform/platform.h"
#include "flashmoe/platform/runtime.h"    // if using runtime API
#include "flashmoe/platform/device.h"     // if using half/bfloat16
#include "flashmoe/platform/intrinsics.h" // if using shuffles/ballot
```

2. **Use `FLASHMOE_WARP_SIZE`** instead of hardcoded `32`.

3. **Use `FLASHMOE_FULL_LANE_MASK`** instead of `0xffffffff` for warp masks.

4. **Use `flashmoe_lane_mask_t`** for ballot return values.

5. **Use `gpu*` macros** from `runtime.h` instead of `cuda*` API calls.

6. **Use `flashmoe_blas::`** namespace (defined in `tile.cuh`) instead of `cublasdx::` directly.

7. **Guard CUDA-specific includes** with `#if !defined(FLASHMOE_PLATFORM_HIP)`:

```cpp
#if !defined(FLASHMOE_PLATFORM_HIP)
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#else
#include "flashmoe/platform/math_compat.h"
#endif
```

8. **For CUB usage**, add the hipcub alias:

```cpp
#if defined(FLASHMOE_PLATFORM_HIP)
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif
```

9. **For NVSHMEM calls**, use the `flashmoe::shmem` wrappers from `shmem.h`:

```cpp
#if __has_include("flashmoe/platform/shmem.h")
#include "flashmoe/platform/shmem.h"
#endif
```

10. **In CMake**, use the existing helper functions:
    - `flash_target_test(tgt)` for test executables
    - `flash_target_common(tgt)` for standard builds
    - `flash_target_lto(tgt)` for link-time optimized builds (distributed)
    - On HIP, set `.cu` files to HIP language: `set_source_files_properties(file.cu PROPERTIES LANGUAGE HIP)`
