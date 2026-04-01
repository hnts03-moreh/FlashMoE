# FlashMoE ROCm Porting Overview

## What is FlashMoE?

FlashMoE is a completely fused distributed Mixture-of-Experts (MoE) kernel library. It fuses the entire MoE forward pass -- gating, expert dispatch, GEMM computation (up-projection, activation, down-projection), and token combine -- into a single persistent kernel. This eliminates kernel launch overhead and intermediate memory traffic that plague conventional MoE implementations.

The original implementation targets NVIDIA GPUs (Volta and above) and relies heavily on CUDA-specific libraries: CUTLASS/CuTe for tile-level abstractions, cuBLASDx for device-side GEMM, NVSHMEM for inter-GPU communication, and CCCL for standard library extensions in device code.

## Why ROCm?

AMD MI300X GPUs offer compelling compute density and memory bandwidth for large-scale MoE workloads. Porting FlashMoE to ROCm via HIP enables:

- Running on MI300X (gfx942) with 192 GB HBM3 and 304 Compute Units
- Access to CDNA3 MFMA (Matrix Fused Multiply-Add) instructions
- Deployment on AMD-based HPC clusters and cloud instances

## Target Platforms

| Platform | Architecture | GPU | Status |
|----------|-------------|-----|--------|
| MI300X | CDNA3 / gfx942 | AMD Instinct MI300X | Primary target |
| MI300A | CDNA3 / gfx940/941 | AMD Instinct MI300A | Planned |
| MI250 | CDNA2 / gfx90a | AMD Instinct MI250X | Future |
| CDNA4 | Next-gen | TBD | Future |

The CUDA build path (H100, A100, etc.) is fully preserved. All changes use compile-time `#ifdef` branching -- zero behavioral impact on existing CUDA builds.

## Porting Strategy

The project uses an **abstraction layer approach** rather than a wholesale rewrite. The key principle: every CUDA API call and NVIDIA-specific library usage is wrapped behind a platform-neutral interface that resolves at compile time to either the CUDA original or the HIP/ROCm equivalent.

### Design Principles

1. **Zero runtime overhead** -- All abstractions use `#define`, `using`, `constexpr`, or `__forceinline__` functions. No virtual dispatch, no runtime branching.
2. **Compile-time platform selection** -- A single CMake variable (`FLASHMOE_PLATFORM`) controls the entire build. Auto-detection works out of the box.
3. **CUDA code preservation** -- The original CUDA code is untouched within `#else` branches. The CUDA build path remains identical to upstream.
4. **Minimal API surface** -- Only the subset of each library actually used by FlashMoE is reimplemented.

## Dependency Replacement Summary

| NVIDIA Dependency | Usage | ROCm Replacement | Approach |
|-------------------|-------|------------------|----------|
| CUTLASS / CuTe | Tile math, arrays, layouts (28 files) | `math_compat.h` | Header-only stubs in `cuda::std::`, `cute::`, `cutlass::` namespaces |
| cuBLASDx | Device-side GEMM (`tile.cuh`, `processor.cuh`) | **rocBLASDx** (new sub-project) | Drop-in API replacement using MFMA intrinsics |
| CCCL (`cuda::std::`) | Standard library in device code (~all files) | `math_compat.h` | `cuda::std::` redirected to `std::` with device-compatible wrappers |
| NVSHMEM | Distributed communication (4 critical files) | **ROCSHMEM** | `shmem.h` abstraction with `flashmoe::shmem` namespace |
| NVTX3 | Profiling / tracing | **rocTX** | `profiling.h` with `flashmoe::flashmoeRange` scoped ranges |
| CUB | Block/warp scan (`gate.cuh`, `scheduler.cuh`) | **hipcub** | `namespace cub = hipcub;` alias |
| `cuda-core` (Python) | Device management in Python layer | **hip-python** | Platform-conditional imports in `cb.py`, `jit.py` |
| `nvshmem4py` | Python distributed communication | ROCSHMEM Python binding (TBD) | Abstracted via `_CommBackend` class in `cb.py` |
| MatX | Test reference implementations | Removed on HIP | Tests run kernel benchmarks without MatX reference comparison |
| cuRANDDx | Test random number generation | **hipRAND** | Philox4x32 generator via `hiprandStatePhilox4_32_10_t` |

## Architecture Changes Summary

### Warp Size: 32 to 64

NVIDIA warps contain 32 threads; AMD wavefronts contain 64. This affects:
- All ballot operations (return `uint64_t` instead of `uint32_t`)
- Shuffle operations (mask parameter stripped on HIP)
- Scheduler thread partitioning (`SCHEDULER_COUNT = WARP_SIZE = 64`)
- Shared memory bank width calculations

The `FLASHMOE_WARP_SIZE` macro and `flashmoe_lane_mask_t` type handle this transparently.

### Matrix Instructions: Tensor Cores to MFMA

NVIDIA Tensor Cores are replaced by AMD MFMA instructions:
- `mfma_f32_32x32x8f16` for fp16 large tiles
- `mfma_f32_16x16x16f16` for fp16 small tiles
- Corresponding bf16, fp32, and fp64 variants

The rocBLASDx sub-project encapsulates all MFMA instruction selection and thread-data mapping.

### Shared Memory: SMEM to LDS

MI300X has 64 KB of Local Data Share (LDS) per compute unit, compared to 228 KB of configurable shared memory on H100. The build sets `MAX_ACCESS_ALIGNMENT` to 16 on HIP and applies LDS bank-conflict padding in rocBLASDx shared memory layouts.

### Async Copy: cp.async to Synchronous

CUDA's `cp.async` pipeline (used for multi-stage shared memory prefetching) has no direct HIP equivalent. The HIP mainloop in `tile.cuh` uses synchronous loads with `__syncthreads()` barriers. This is functionally correct but leaves performance on the table -- LDS double-buffering and `buffer_load` prefetching are future optimization targets.

### PTX Intrinsics to Standard Loads/Stores

CUDA PTX intrinsics (`cuda::ptx::st`, `cuda::ptx::ld_nc_L1_no_allocate_L2_256B`) are replaced with `__builtin_nontemporal_load` / `__builtin_nontemporal_store` on HIP, or plain loads/stores where non-temporal hints are not critical.

## Project Structure

```
FlashMoE/
  CMakeLists.txt                          # Root build (platform-aware)
  csrc/
    CMakeLists.txt                        # Test build (platform-aware)
    include/flashmoe/
      platform/                           # NEW: abstraction layer
        platform.h                        #   Platform detection, warp size
        runtime.h                         #   CUDA/HIP runtime API mapping
        device.h                          #   Device qualifiers, half types
        atomic.h                          #   Atomic operations shim
        intrinsics.h                      #   Warp/wavefront primitives
        math_compat.h                     #   CUTLASS/CuTe/CCCL stubs (HIP only)
        profiling.h                       #   NVTX/rocTX mapping
        shmem.h                           #   NVSHMEM/ROCSHMEM mapping
      *.cuh                               # Kernel headers (ported with #ifdef)
    tests/                                # Test files (ported)
  rocblasdx/                              # NEW: device-side GEMM library
    include/rocblasdx/
      rocblasdx.hpp                       #   Main header
      types.hpp                           #   Descriptor building blocks
      layout.hpp                          #   Shared memory layouts
      fragment.hpp                        #   Accumulator fragments
      mfma.hpp                            #   MFMA intrinsic wrappers
      gemm.hpp                            #   GEMM execution engine
    tests/
  flashmoe/                               # Python layer (ported)
    jit.py                                #   JIT compilation (platform-aware)
    bindings.py                           #   C++ code generation (platform-aware)
    cb.py                                 #   Communication backend (abstracted)
    __init__.py                           #   Public API (platform exports)
    CMakeLists.txt                        #   JIT build (platform-aware)
  docs/                                   # Documentation
```

## Execution Phases

The porting was executed in three phases:

1. **Foundation** -- Platform abstraction headers and CMake build system; rocBLASDx sub-project
2. **Core porting** (parallel) -- 35 kernel `.cuh` files; NVSHMEM-to-ROCSHMEM communication; Python layer
3. **Integration** -- Test porting, cross-boundary verification, CMake test targets
