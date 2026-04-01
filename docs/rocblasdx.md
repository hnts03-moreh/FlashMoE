# rocBLASDx -- Device-Side GEMM Library

## Purpose

rocBLASDx is a header-only device-side GEMM library that replaces NVIDIA's cuBLASDx for AMD MI300X (gfx942) GPUs. It implements only the API subset that FlashMoE actually uses, providing a drop-in `cublasdx::` to `rocblasdx::` namespace replacement.

The library uses AMD MFMA (Matrix Fused Multiply-Add) intrinsics from the CDNA3 architecture to perform tile-level matrix multiplications directly in device code, without host-side library calls.

## Project Structure

```
rocblasdx/
  CMakeLists.txt                    Standalone build, INTERFACE library
  include/rocblasdx/
    rocblasdx.hpp                   Main include header (includes everything)
    types.hpp                       Descriptor building blocks and enums
    layout.hpp                      Shared memory layout with LDS padding
    fragment.hpp                    Accumulator fragment type
    mfma.hpp                        MFMA intrinsic wrappers and selection
    gemm.hpp                        GEMM execution engine, copy, tensor ops
  tests/
    test_gemm.cpp                   Compile-time and runtime tests
```

## Usage

```cpp
#include <rocblasdx/rocblasdx.hpp>

// 1. Build a BLAS descriptor (identical pattern to cuBLASDx):
using BLAS = decltype(
    rocblasdx::Size<64, 64, 32>() +
    rocblasdx::Precision<__half, __half, float>() +
    rocblasdx::Type<rocblasdx::type::real>() +
    rocblasdx::Function<rocblasdx::function::MM>() +
    rocblasdx::Arrangement<rocblasdx::row_major, rocblasdx::col_major, rocblasdx::row_major>() +
    rocblasdx::Block() +
    rocblasdx::Alignment<16, 16, 16>() +
    rocblasdx::BlockDim<256>() +
    rocblasdx::StaticBlockDim() +
    rocblasdx::EnableInputStreaming() +
    rocblasdx::SM<942, rocblasdx::sm_modifier::arch_specific>());

// 2. Query descriptor properties:
constexpr int M = rocblasdx::size_of<BLAS>::m;
constexpr int N = rocblasdx::size_of<BLAS>::n;
constexpr int K = rocblasdx::size_of<BLAS>::k;
constexpr int max_threads = BLAS::max_threads_per_block;

// 3. Set up shared memory and accumulators (in a __global__ kernel):
auto layout_a = BLAS::suggest_layout_smem_a();
auto layout_b = BLAS::suggest_layout_smem_b();
auto layout_c = BLAS::suggest_layout_smem_c();

extern __shared__ char smem[];
auto sA = rocblasdx::make_tensor(reinterpret_cast<__half*>(smem), layout_a);
auto sB = rocblasdx::make_tensor(reinterpret_cast<__half*>(smem + offset_b), layout_b);

auto accumulator = BLAS::suggest_accumulator();
accumulator.clear();

// 4. Copy data to shared memory:
rocblasdx::copy<BLAS, 16>(src_tensor, sA);
rocblasdx::copy_wait();

// 5. Execute GEMM:
BLAS().execute(sA, sB, accumulator, transform_a, transform_b);

// 6. Read results:
auto& results = accumulator.get_results();
```

## cuBLASDx API Mapping

### Descriptor Building Blocks

| cuBLASDx | rocBLASDx | File |
|----------|-----------|------|
| `cublasdx::Size<M,N,K>()` | `rocblasdx::Size<M,N,K>()` | types.hpp |
| `cublasdx::Precision<A,B,C>()` | `rocblasdx::Precision<A,B,C>()` | types.hpp |
| `cublasdx::Type<cublasdx::type::real>()` | `rocblasdx::Type<rocblasdx::type::real>()` | types.hpp |
| `cublasdx::Function<cublasdx::function::MM>()` | `rocblasdx::Function<rocblasdx::function::MM>()` | types.hpp |
| `cublasdx::Block()` | `rocblasdx::Block()` | types.hpp |
| `cublasdx::BlockDim<N>()` | `rocblasdx::BlockDim<N>()` | types.hpp |
| `cublasdx::StaticBlockDim()` | `rocblasdx::StaticBlockDim()` | types.hpp |
| `cublasdx::EnableInputStreaming()` | `rocblasdx::EnableInputStreaming()` | types.hpp |
| `cublasdx::SM<Arch, Mod>()` | `rocblasdx::SM<Arch, Mod>()` | types.hpp |
| `cublasdx::Alignment<a,b,c>()` | `rocblasdx::Alignment<a,b,c>()` | types.hpp |
| `cublasdx::Arrangement<ar,br,cr>()` | `rocblasdx::Arrangement<ar,br,cr>()` | types.hpp |

### Layout and Arrangement

| cuBLASDx | rocBLASDx | File |
|----------|-----------|------|
| `cublasdx::arrangement` (enum) | `rocblasdx::arrangement` | layout.hpp |
| `cublasdx::row_major` | `rocblasdx::row_major` | layout.hpp |
| `cublasdx::col_major` | `rocblasdx::col_major` | layout.hpp |
| `cublasdx::sm_modifier::arch_specific` | `rocblasdx::sm_modifier::arch_specific` | types.hpp |

### Type Utilities

| cuBLASDx | rocBLASDx | File |
|----------|-----------|------|
| `cublasdx::tfloat32_t` | `rocblasdx::tfloat32_t` | types.hpp |
| `cublasdx::identity` | `rocblasdx::identity` | types.hpp |

### BLAS Execution and Query

| cuBLASDx | rocBLASDx | File |
|----------|-----------|------|
| `BLAS::suggest_accumulator()` | Same API | gemm.hpp |
| `BLAS().execute(sA, sB, accum, tA, tB)` | Same API | gemm.hpp |
| `BLAS::suggest_layout_smem_a/b/c()` | Returns `SmemLayout<R,C,S>` | gemm.hpp |
| `BLAS::get_layout_smem_c()` | Same as `suggest_layout_smem_c()` | gemm.hpp |
| `BLAS::max_threads_per_block` | Same `static constexpr` | gemm.hpp |
| `cublasdx::size_of<BLAS>::m/n/k` | `rocblasdx::size_of<BLAS>::m/n/k` | gemm.hpp |
| `cublasdx::alignment_of<BLAS>::a/b/c` | `rocblasdx::alignment_of<BLAS>::a/b/c` | gemm.hpp |
| `cublasdx::arrangement_of_v_c<BLAS>` | `rocblasdx::arrangement_of_v_c<BLAS>` | gemm.hpp |
| `cublasdx::is_blas_execution_v<BLAS>` | `rocblasdx::is_blas_execution_v<BLAS>` | gemm.hpp |

### Tensor and Copy Operations

| cuBLASDx | rocBLASDx | File |
|----------|-----------|------|
| `cublasdx::make_tensor(ptr, layout)` | `rocblasdx::make_tensor(ptr, layout)` | gemm.hpp |
| `cublasdx::copy<BLAS, align>(src, dst)` | `rocblasdx::copy<BLAS, align>(src, dst)` | gemm.hpp |
| `cublasdx::copy<threads, align>(tid, src, dst)` | `rocblasdx::copy<threads, align>(tid, src, dst)` | gemm.hpp |
| `cublasdx::copy_fragment<align>(a, b, accum)` | `rocblasdx::copy_fragment<align>(a, b, accum)` | gemm.hpp |
| `cublasdx::copy_wait()` | `rocblasdx::copy_wait()` | gemm.hpp |
| `cublasdx::cosize(layout)` | `rocblasdx::cosize(SmemLayout<>)` | gemm.hpp |

### Fragment Operations

| cuBLASDx | rocBLASDx | File |
|----------|-----------|------|
| `cublasdx::size(fragment)` | `rocblasdx::size(fragment)` | fragment.hpp |
| `cublasdx::make_fragment_like<T>(frag)` | `rocblasdx::make_fragment_like<T>(frag)` | fragment.hpp |
| `accumulator.get_results()` | `Fragment::get_results()` | fragment.hpp |
| `accumulator.clear()` | `Fragment::clear()` | fragment.hpp |
| `decltype(accumulator)::value_type` | `Fragment::value_type` | fragment.hpp |

## Descriptor Composition

rocBLASDx replicates cuBLASDx's `operator+` descriptor composition pattern:

```cpp
using BLAS = decltype(
    rocblasdx::Size<bM, bN, bK>() +
    rocblasdx::Precision<Element, Element, MMA_C>() +
    rocblasdx::Type<rocblasdx::type::real>() +
    rocblasdx::Function<rocblasdx::function::MM>() +
    rocblasdx::SM<942, rocblasdx::sm_modifier::arch_specific>());
```

Each `operator+` call returns a `detail::BlasBag<...>` with the corresponding field updated. The final type exposes the full BLAS execution API (`execute()`, `suggest_accumulator()`, etc.). Left-to-right associativity matches C++ `operator+` semantics.

## MFMA Instruction Selection

The `SelectMfma` type trait in `mfma.hpp` chooses the appropriate MFMA instruction based on element type and tile dimensions:

| Element Type | Tile >= 32x32 | Tile < 32x32 |
|-------------|---------------|--------------|
| `__half` (fp16) | `mfma_f32_32x32x8f16` | `mfma_f32_16x16x16f16` |
| `hip_bfloat16` | `mfma_f32_32x32x8bf16_1k` | `mfma_f32_16x16x16bf16_1k` |
| `float` | `mfma_f32_32x32x2f32` | `mfma_f32_16x16x4f32` |
| `tfloat32_t` | Falls back to fp32 MFMA | Falls back to fp32 MFMA |
| `double` | `mfma_f64_16x16x4f64` | `mfma_f64_16x16x4f64` |

MI300X has no native TF32 instruction. `tfloat32_t` is a `float` wrapper that maps to the fp32 MFMA path.

All MFMA wrappers assume wavefront_size=64 (CDNA3).

## Shared Memory Layout

`SmemLayout<Rows, Cols, Stride>` includes LDS bank-conflict padding. The padding is computed based on MI300X's 128-byte LDS bank cycle:

```cpp
auto layout = BLAS::suggest_layout_smem_a();
// layout.stride includes padding to avoid bank conflicts
size_t alloc_bytes = rocblasdx::cosize(layout) * sizeof(Element);
```

`cosize()` returns `rows * stride` (including padding), matching cuBLASDx behavior for shared memory allocation.

## Key Design Decisions

1. **Header-only.** All code is in `.hpp` files. No compilation step needed -- just add the include path.

2. **Wavefront 64.** All thread-data mappings assume wavefront_size=64 (CDNA3). Supporting wavefront_size=32 (RDNA) would require additional specializations.

3. **Two execution paths.** The GEMM `execute()` method has an MFMA path (`#if defined(__gfx9__)`) and a scalar fallback for non-GPU targets (host-side testing).

4. **Fragment = flat array.** Unlike cuBLASDx's complex tensor-based accumulator, `Fragment<T,N>` is a simple array. Thread-to-element mapping is `flat = tid + i * blockDim.x`, matching a strided distribution for `copy_fragment`.

5. **No async copy.** cuBLASDx's `copy<BLAS, align>` uses `cp.async` on NVIDIA. The rocBLASDx implementation uses synchronous loads. MI300X's `buffer_load` instructions are not yet leveraged.

6. **Synchronous mainloop.** The `CollectiveMainloop` in `tile.cuh` replaces `cp_async_fence()`/`cp_async_wait<N>()` with `__syncthreads()` on HIP. Multi-stage pipelining is not implemented.

## Building and Testing

### Standalone Build

```bash
cmake -B build -S rocblasdx \
  -DROCBLASDX_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942

cmake --build build
./build/test_gemm
```

### As a Sub-project

The `csrc/CMakeLists.txt` automatically includes rocBLASDx headers via:

```cmake
set(ROCBLASDX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rocblasdx/include")
target_include_directories(${tgt} SYSTEM PRIVATE "${ROCBLASDX_INCLUDE_DIR}")
```

### In Kernel Code

```cpp
// In tile.cuh (already done):
#if defined(FLASHMOE_PLATFORM_HIP)
#include <rocblasdx/rocblasdx.hpp>
namespace flashmoe_blas = rocblasdx;
#else
#include <cublasdx.hpp>
namespace flashmoe_blas = cublasdx;
#endif

// All downstream code uses flashmoe_blas:: namespace
```

## Extending rocBLASDx

### Adding a New Architecture

To support a new AMD GPU architecture (e.g., CDNA4):

1. **Add MFMA wrappers in `mfma.hpp`.**
   Add new `__builtin_amdgcn_mfma_*` intrinsic wrappers for the architecture's matrix instructions.

2. **Update `SelectMfma` in `mfma.hpp`.**
   Add template specializations or `constexpr` branches for the new architecture's instruction set.

3. **Update `SmemLayout` padding in `layout.hpp`.**
   Adjust the LDS bank-conflict padding calculation if the new architecture has a different bank width or cycle.

4. **Update architecture detection in `platform/platform.h`.**
   Add `FLASHMOE_ARCH_<name>` macros for the new GPU.

5. **Update `CMakeLists.txt`.**
   Add the new `gfx` target to the architecture defaults.

### Making rocBLASDx a Standalone Project

rocBLASDx is designed to be usable independently of FlashMoE:

1. The `rocblasdx/CMakeLists.txt` provides a standalone INTERFACE library target.
2. All headers are self-contained (depend only on HIP runtime headers).
3. No dependency on FlashMoE's `platform/` headers.

To use in another project:

```cmake
add_subdirectory(path/to/rocblasdx)
target_link_libraries(my_target PRIVATE rocblasdx::rocblasdx)
```

Or copy the `rocblasdx/include/rocblasdx/` directory and add it to your include path.

## Known Limitations

1. **MFMA thread-data mapping.** The current implementation follows the canonical CDNA3 lane-to-data mapping from composable_kernel. End-to-end correctness with FlashMoE kernels requires integration testing.

2. **No async global-to-LDS copy.** MI300X `buffer_load` instructions are not used. All shared memory loads are synchronous.

3. **No warp-level scheduling optimization.** The MFMA execution loop does not implement software pipelining of shared memory loads and MFMA instructions.

4. **`cosize` for CuTe layouts.** The `cosize()` function only works with `SmemLayout`. CuTe layout overloads may be needed for `combine.cuh` patterns.

5. **No RDNA support.** All thread-data mappings assume wavefront-64. RDNA GPUs (wavefront-32) are not supported.
