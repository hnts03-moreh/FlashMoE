/*
 * Copyright (c) 2025, FlashMoE ROCm Porting Project
 * All rights reserved.
 * BSD 3-Clause License — see LICENSE in the project root.
 *
 * test_gemm.cpp — Basic unit tests for rocBLASDx
 *
 * Tests:
 *   1. BLAS descriptor composition (compile-time)
 *   2. size_of / alignment_of / arrangement_of_v_c traits
 *   3. Fragment creation, clear, element access
 *   4. make_fragment_like
 *   5. SmemLayout cosize
 *   6. Kernel launch: scalar GEMM correctness (small tile)
 *   7. Identity functor
 */

#include <hip/hip_runtime.h>
#include <rocblasdx/rocblasdx.hpp>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

#define HIP_CHECK(e) do {                                       \
    hipError_t err = (e);                                       \
    if (err != hipSuccess) {                                    \
        fprintf(stderr, "HIP error %s at %s:%d\n",             \
                hipGetErrorString(err), __FILE__, __LINE__);    \
        exit(1);                                                \
    }                                                           \
} while (0)

// -------------------------------------------------------
// Test 1: Compile-time descriptor composition
// -------------------------------------------------------
using TestBLAS = decltype(
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

static_assert(rocblasdx::size_of<TestBLAS>::m == 64);
static_assert(rocblasdx::size_of<TestBLAS>::n == 64);
static_assert(rocblasdx::size_of<TestBLAS>::k == 32);
static_assert(rocblasdx::alignment_of<TestBLAS>::a == 16);
static_assert(rocblasdx::alignment_of<TestBLAS>::b == 16);
static_assert(rocblasdx::alignment_of<TestBLAS>::c == 16);
static_assert(rocblasdx::arrangement_of_v_c<TestBLAS> == rocblasdx::row_major);
static_assert(rocblasdx::is_blas_execution_v<TestBLAS>);
static_assert(TestBLAS::max_threads_per_block == 256);

// -------------------------------------------------------
// Test 2: Fragment operations
// -------------------------------------------------------
void test_fragment() {
    rocblasdx::Fragment<float, 16> frag;
    frag.clear();
    for (int i = 0; i < 16; ++i) {
        assert(frag(i) == 0.0f);
    }
    frag(3) = 42.0f;
    assert(frag(3) == 42.0f);

    auto& results = frag.get_results();
    assert(results(3) == 42.0f);

    auto frag2 = rocblasdx::make_fragment_like<__half>(frag);
    static_assert(std::is_same_v<decltype(frag2)::value_type, __half>);
    static_assert(decltype(frag2)::num_elements == 16);

    constexpr int sz = rocblasdx::size(frag);
    static_assert(sz == 16);

    printf("  [PASS] Fragment operations\n");
}

// -------------------------------------------------------
// Test 3: SmemLayout cosize
// -------------------------------------------------------
void test_layout() {
    using L = rocblasdx::SmemLayout<32, 32, 34>;  // 32 rows, 32 cols, stride 34 (padded)
    static_assert(rocblasdx::cosize(L{}) == 32 * 34);

    // Test suggested layouts
    using SLA = rocblasdx::SuggestSmemLayout<rocblasdx::row_major, 64, 32, __half>;
    static_assert(SLA::type::rows == 64);
    static_assert(SLA::type::cols == 32);
    // stride should be >= 32 (with possible padding)
    static_assert(SLA::type::stride >= 32);

    printf("  [PASS] SmemLayout cosize and suggest\n");
}

// -------------------------------------------------------
// Test 4: Identity functor
// -------------------------------------------------------
void test_identity() {
    rocblasdx::identity id;
    assert(id(42.0f) == 42.0f);
    assert(id(3) == 3);
    printf("  [PASS] Identity functor\n");
}

// -------------------------------------------------------
// Test 5: tfloat32_t
// -------------------------------------------------------
void test_tfloat32() {
    rocblasdx::tfloat32_t a(3.0f);
    rocblasdx::tfloat32_t b(4.0f);
    auto c = a + b;
    assert(static_cast<float>(c) == 7.0f);
    auto d = a * b;
    assert(static_cast<float>(d) == 12.0f);
    printf("  [PASS] tfloat32_t operations\n");
}

// -------------------------------------------------------
// Test 6: Small GEMM kernel (scalar path, correctness check)
// -------------------------------------------------------
// Use a small 16x16x16 tile with float for easy verification.
using SmallBLAS = decltype(
    rocblasdx::Size<16, 16, 16>() +
    rocblasdx::Precision<float, float, float>() +
    rocblasdx::Type<rocblasdx::type::real>() +
    rocblasdx::Function<rocblasdx::function::MM>() +
    rocblasdx::Arrangement<rocblasdx::row_major, rocblasdx::col_major, rocblasdx::row_major>() +
    rocblasdx::Block() +
    rocblasdx::Alignment<16, 16, 16>() +
    rocblasdx::BlockDim<64>() +
    rocblasdx::StaticBlockDim() +
    rocblasdx::EnableInputStreaming() +
    rocblasdx::SM<942, rocblasdx::sm_modifier::generic>());

__global__
void test_gemm_kernel(const float* __restrict__ A,
                      const float* __restrict__ B,
                      float* __restrict__ C) {
    constexpr int M = 16, N = 16, K = 16;

    // Shared memory for A and B tiles
    using SLA = SmallBLAS::SmemLayoutA;
    using SLB = SmallBLAS::SmemLayoutB;

    __shared__ float smem_a[SLA::rows * SLA::stride];
    __shared__ float smem_b[SLB::rows * SLB::stride];

    // Load A (row-major) from global to shared
    for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
        int row = i / K;
        int col = i % K;
        smem_a[row * SLA::stride + col] = A[row * K + col];
    }

    // Load B (col-major) from global to shared
    for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
        int row = i / N;
        int col = i % N;
        smem_b[col * SLB::stride + row] = B[row * N + col];  // store col-major
    }
    __syncthreads();

    // Create accumulator and smem tensors
    auto accumulator = SmallBLAS::suggest_accumulator();

    auto sA = rocblasdx::make_tensor(smem_a, SLA{});
    auto sB = rocblasdx::make_tensor(smem_b, SLB{});

    // Execute GEMM
    SmallBLAS().execute(sA, sB, accumulator);

    // Write results: each thread writes its portion
    const auto& results = accumulator.get_results();
    constexpr int total = M * N;
    constexpr int per_thread = (total + 63) / 64;
    for (int idx = 0; idx < per_thread; ++idx) {
        int flat = threadIdx.x + idx * 64;
        if (flat < total && idx < decltype(results)::num_elements) {
            int row = flat / N;
            int col = flat % N;
            C[row * N + col] = results(idx);
        }
    }
}

void test_gemm_correctness() {
    constexpr int M = 16, N = 16, K = 16;

    // Host data
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f), h_ref(M * N, 0.0f);

    // Initialize: A = identity-like, B = identity-like
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            h_A[i * K + j] = (i == j) ? 1.0f : 0.0f;

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            h_B[i * N + j] = (i == j) ? 1.0f : 0.0f;

    // Reference: C = A * B (identity * identity = identity)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int kk = 0; kk < K; ++kk)
                sum += h_A[i * K + kk] * h_B[kk * N + j];
            h_ref[i * N + j] = sum;
        }

    // Device
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));

    test_gemm_kernel<<<1, 64>>>(d_A, d_B, d_C);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float err = std::fabs(h_C[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }

    if (max_err < 1e-3f) {
        printf("  [PASS] Small GEMM correctness (max error: %e)\n", max_err);
    } else {
        printf("  [FAIL] Small GEMM correctness (max error: %e)\n", max_err);
        // Print first few mismatches
        for (int i = 0; i < M * N && i < 5; ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) {
                printf("    C[%d] = %f, ref = %f\n", i, h_C[i], h_ref[i]);
            }
        }
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

// -------------------------------------------------------
// Main
// -------------------------------------------------------
int main() {
    printf("rocBLASDx test suite\n");
    printf("====================\n");

    test_fragment();
    test_layout();
    test_identity();
    test_tfloat32();

    // GPU test (only if device is available)
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count > 0) {
        test_gemm_correctness();
    } else {
        printf("  [SKIP] No GPU device found, skipping GEMM kernel test\n");
    }

    printf("====================\n");
    printf("All tests complete.\n");
    return 0;
}
