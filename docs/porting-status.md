# Porting Status and Known Issues

## File-by-File Porting Status

### Platform Abstraction Headers (NEW -- 8 files)

| File | Status | Notes |
|------|--------|-------|
| `platform/platform.h` | Complete | Platform detection, warp size, lane mask, arch detection |
| `platform/runtime.h` | Complete | Full CUDA/HIP runtime API mapping |
| `platform/device.h` | Complete | `__grid_constant__` no-op, `__nv_bfloat16` alias |
| `platform/atomic.h` | Complete | `cuda::atomic_ref` and `cuda::thread_scope` shim |
| `platform/intrinsics.h` | Complete | Shuffle, ballot, popc, ffs, syncwarp |
| `platform/math_compat.h` | Complete | CUTLASS/CuTe/CCCL stubs (HIP only) |
| `platform/profiling.h` | Complete | NVTX3 to rocTX mapping |
| `platform/shmem.h` | Complete | NVSHMEM to ROCSHMEM mapping |

### rocBLASDx Sub-project (NEW -- 7 files)

| File | Status | Notes |
|------|--------|-------|
| `rocblasdx/CMakeLists.txt` | Complete | Standalone INTERFACE library |
| `rocblasdx/include/rocblasdx/rocblasdx.hpp` | Complete | Main include header |
| `rocblasdx/include/rocblasdx/types.hpp` | Complete | All descriptor building blocks |
| `rocblasdx/include/rocblasdx/layout.hpp` | Complete | SmemLayout with LDS padding |
| `rocblasdx/include/rocblasdx/fragment.hpp` | Complete | Fragment/Accumulator type |
| `rocblasdx/include/rocblasdx/mfma.hpp` | Complete | 7 MFMA instruction wrappers |
| `rocblasdx/include/rocblasdx/gemm.hpp` | Complete | GEMM execution, copy, tensor ops |
| `rocblasdx/tests/test_gemm.cpp` | Complete | Static asserts and small GEMM test |

### Kernel Headers -- Infrastructure (16 files)

| File | Status | Difficulty | Notes |
|------|--------|-----------|-------|
| `infra/constants.cuh` | Complete | Low | `WARP_SIZE = FLASHMOE_WARP_SIZE` |
| `infra/math.cuh` | Complete | Low | `cuda::round_up` via compat shim |
| `infra/checks.cuh` | Complete | Low | CuTe `Tensor` specialization CUDA-only |
| `infra/packed.cuh` | Complete | Low | `device.h` includes |
| `infra/structures.cuh` | Complete | Low | `cuda::std::byte` alias |
| `infra/bitset.cuh` | Complete | Medium | `cuda::std::min` via compat shim |
| `infra/dq.cuh` | Complete | Medium | No changes needed |
| `infra/tq.cuh` | Complete | Medium | No changes needed |
| `infra/vt.cuh` | Complete | Medium | `cutlass::is_pow2` shim added |
| `infra/atomics.cuh` | Complete | Medium | `cuda::atomic_ref` via `atomic.h` shim |
| `infra/activation.cuh` | Complete | Medium | `hip_compat::` activation functors |
| `infra/telemetry.cuh` | Complete | Medium | NVTX3 -> `profiling.h` |
| `infra/task.cuh` | Complete | Medium | `std::array` / `std::is_trivially_copyable_v` |
| `infra/rvt.cuh` | Complete | Medium | `atomicAdd` replaces PTX `red.global.add.*` |
| `infra/heap.cuh` | Complete | Low | `cuda::std::byte` namespace alias |
| `infra/signal.cuh` | Complete | Low | No changes needed |

### Kernel Headers -- Tile/Compute Core (2 files)

| File | Status | Difficulty | Notes |
|------|--------|-----------|-------|
| `tile.cuh` | Complete | High | `flashmoe_blas` namespace; synchronous mainloop on HIP; `cp_async` -> `__syncthreads()` |
| `context.cuh` | Complete | Medium | `cuda::fast_mod_div` shim |

### Kernel Headers -- Orchestration (5 files)

| File | Status | Difficulty | Notes |
|------|--------|-----------|-------|
| `gate.cuh` | Complete | High | CUB -> hipcub; `flashmoe_blas::` |
| `combine.cuh` | Complete | High | `flashmoe_blas::` |
| `processor.cuh` | Complete | High | `flashmoe_blas::`; PTX stores replaced; NVSHMEM guarded |
| `scheduler.cuh` | Complete | Medium | CUB -> hipcub; lane mask fixes |
| `subscriber.cuh` | Complete | Medium | Lane mask fixes; NVSHMEM guarded |

### Kernel Headers -- Host-Side / Top-Level (7 files)

| File | Status | Difficulty | Notes |
|------|--------|-----------|-------|
| `dispatch.cuh` | Complete | Critical | PTX intrinsics replaced; NVSHMEM guarded via `shmem.h` |
| `bootstrap.cuh` | Complete | Critical | Runtime API mapped; NVSHMEM via `shmem.h` |
| `os.cuh` | Complete | Low | `cuda::std::cstddef` guarded |
| `moe.cuh` | Complete | Low | `cudaMemsetAsync` -> `gpuMemsetAsync` |
| `flashmoe.cuh` | Complete | Low | No changes needed (aggregator includes) |
| `experimental/topo.cuh` | Partial | Critical | `#error` marker -- NVSHMEM extensions have no ROCSHMEM equivalents |
| `experimental/decider/*.cuh` | Complete | Medium | CuTe guarded; `math_compat.h` |

### Test Files (10 files)

| File | Status | Notes |
|------|--------|-------|
| `tests/common.cuh` | Complete | cuRANDDx -> hipRAND Philox; MatX guarded CUDA-only |
| `tests/debug.cuh` | Complete | Already ported (no changes needed) |
| `tests/gemm.cu` | Complete | `flashmoe_blas::` throughout; event-based benchmark on HIP |
| `tests/gemmMNK.cu` | Complete | `pipeStages=1` on HIP |
| `tests/gatedGEMM.cu` | Complete | MatX gated reference CUDA-only |
| `tests/gate.cu` | Complete | NVTX -> `profiling.h`; MatX CUDA-only |
| `tests/combine.cu` | Complete | MatX comparison CUDA-only |
| `tests/scheduler.cu` | Complete | hipcub; `FLASHMOE_WARP_SIZE` in helpers |
| `tests/flashmoe.cu` | Partial | Stub on HIP (prints "not yet available") |
| `tests/playground.cu` | Complete | No changes needed (pure CPU code) |

### Python Layer (7 files)

| File | Status | Notes |
|------|--------|-------|
| `flashmoe/jit.py` | Complete | `Platform` enum; platform detection; HIP JIT with `.cpp` ext |
| `flashmoe/bindings.py` | Complete | `gpu*` macros; conditional `rocblasdx::`/`cublasdx::` |
| `flashmoe/cb.py` | Complete | `_GPURuntime` and `_CommBackend` abstractions |
| `flashmoe/__init__.py` | Complete | Platform exports; conditional comm init |
| `flashmoe/reference.py` | Complete | No changes needed (uses platform-aware JIT) |
| `flashmoe/router.py` | Complete | No changes needed (uses platform-aware JIT) |
| `quickstart.py` | Complete | Event-based timing on HIP (no hipGraph capture) |

### Build System (4 files)

| File | Status | Notes |
|------|--------|-------|
| `CMakeLists.txt` (root) | Complete | `FLASHMOE_PLATFORM` option; HIP deps; ROCSHMEM optional |
| `csrc/CMakeLists.txt` | Complete | HIP helper functions; all HIP test targets |
| `flashmoe/CMakeLists.txt` | Complete | HIP JIT build path; `-fgpu-rdc` |
| `pyproject.toml` | Complete | `[rocm]` optional dependency |

## Completed Work Summary

- 8 platform abstraction headers created
- 7 rocBLASDx library files created (header-only device GEMM with MFMA)
- 35 kernel `.cuh` files ported with `#ifdef` dual-platform support
- 10 test files ported with HIP test targets in CMake
- 7 Python files ported with platform detection and conditional imports
- 4 build system files updated for dual-platform support
- NVSHMEM-to-ROCSHMEM mapping with host and device APIs
- NVTX3-to-rocTX profiling mapping
- CUB-to-hipcub aliasing
- cuRANDDx-to-hipRAND test random number generation

## Known Limitations

### 1. No MatX Replacement for HIP Numerical Verification

**Impact: Medium**

MatX (NVIDIA's tensor library) provides reference implementations in CUDA tests for correctness checking. There is no ROCm equivalent. HIP tests run kernel benchmarks but cannot verify numerical correctness against a reference library on-device.

**Workaround:** Use PyTorch ROCm or rocBLAS host-side GEMM for offline correctness validation.

### 2. flashmoe.cu E2E Test is a Stub on HIP

**Impact: Low** (other tests cover kernel functionality)

The end-to-end distributed MoE test requires NVSHMEM/ROCSHMEM + MPI integration. On HIP, it compiles to a stub that prints "not yet available". All individual kernel tests (GEMM, gate, combine, scheduler) work independently.

### 3. hipGraph Benchmarking Not Ported

**Impact: Low** (performance measurement only)

The CUDA graph-based benchmarking in `quickstart.py` and `flashmoe.cu` is not ported to HIP graph capture. HIP uses simple event-based timing instead. HIP graph capture is available on ROCm >= 5.x but may not capture all kernel launch patterns correctly.

### 4. ROCSHMEM Python Bindings Do Not Exist

**Impact: Medium** (only for distributed multi-GPU)

There is no official `rocshmem` Python package on PyPI. The `cb.py` `_CommBackend` class raises `NotImplementedError` when ROCSHMEM bindings are not installed. Single-GPU operation works without it.

### 5. Synchronous Mainloop (No Async Copy Pipelining)

**Impact: Medium** (performance)

The HIP mainloop in `tile.cuh` uses `__syncthreads()` instead of CUDA's `cp.async` multi-stage pipelining. This is functionally correct but does not overlap memory loads with compute. Pipeline stages are forced to 1 on HIP.

### 6. Scalar Global Reductions (No Vectorized atomicAdd)

**Impact: Low-Medium** (performance)

CUDA SM90 supports vectorized `red.global.v2/v4/v8` instructions. On HIP, `infra/rvt.cuh` falls back to scalar `atomicAdd` with `VectorWidth=1`. MI300X has high atomic throughput but this is still a gap.

### 7. experimental/topo.cuh Not Ported

**Impact: Low** (experimental code)

This file uses NVSHMEM extension APIs (`nvshmemx_putmem_block`, `nvshmemx_putmem_warp`, `nvshmemx_putmem_signal_nbi_block`) that have different ROCSHMEM equivalents (`_wg` and `_wave` variants). It has an `#error` marker and compiles only on CUDA.

### 8. ROCSHMEM Workgroup Init Not Yet Called in Kernels

**Impact: Low** (only for distributed multi-GPU)

ROCSHMEM requires `rocshmem_wg_init()` at kernel entry points. The `shmem.h` provides `flashmoe::shmem::device::wg_init()` wrappers, but these calls have not yet been inserted into the actual kernel code. Required before distributed tests can work.

### 9. NVSHMEM_TEAM_SHARED_INDEX Has No ROCSHMEM Equivalent

**Impact: Low** (topology detection only)

ROCSHMEM does not have a shared-memory team concept. The `detectTopo()` function in `bootstrap.cuh` defaults to `MIXED` topology on HIP. Proper XGMI topology detection could use `rocm-smi` or ROCm runtime APIs.

## Roadmap

### Near-Term

1. **MFMA correctness validation.** Run the ported GEMM kernels on MI300X hardware and compare results against rocBLAS host-side reference. Verify MFMA thread-data mapping end-to-end.

2. **ROCSHMEM integration testing.** Build with ROCSHMEM, add `wg_init()` calls, and enable the E2E distributed test.

3. **Numerical verification pipeline.** Replace MatX-based reference with PyTorch ROCm or rocBLAS for automated correctness checks.

### Medium-Term

4. **Performance optimization -- MFMA tuning.** Implement software pipelining in the GEMM execution loop. Overlap shared memory loads with MFMA instructions.

5. **Performance optimization -- LDS double-buffering.** Replace synchronous mainloop with LDS double-buffering or `buffer_load` prefetching to hide global memory latency.

6. **Performance optimization -- Global reduction vectorization.** Investigate `buffer_atomic_add` and wider atomic operations for `rvt.cuh`.

7. **ROCSHMEM Python bindings.** Create a minimal `_rocshmem_py` C extension wrapping `rocshmem_init`, `rocshmem_my_pe`, `rocshmem_n_pes`, `rocshmem_barrier_all_on_stream`, `rocshmem_finalize`.

### Long-Term

8. **MI250 (gfx90a) support.** Add MFMA instruction selection for CDNA2. Update `FLASHMOE_WARP_SIZE` and LDS padding for MI250 characteristics.

9. **CDNA4 support.** When available, add new MFMA instruction wrappers and architecture detection.

10. **hipGraph benchmarking.** Port graph capture benchmarking for MI300X.

11. **experimental/topo.cuh porting.** Map NVSHMEM extension APIs to ROCSHMEM `_wg`/`_wave` variants.

12. **Composable_kernel integration.** Evaluate using AMD's composable_kernel library for optimized GEMM kernels as an alternative to the hand-written rocBLASDx MFMA path.
