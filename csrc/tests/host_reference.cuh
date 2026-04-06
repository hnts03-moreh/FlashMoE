//
// Host-side CPU reference implementations for HIP test accuracy verification.
// Replaces MatX (CUDA-only) for numerical comparison on ROCm.
//

#ifndef FLASHMOE_HOST_REFERENCE_CUH
#define FLASHMOE_HOST_REFERENCE_CUH

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../include/flashmoe/platform/runtime.h"

namespace host_ref {

// Convert half/bfloat16 to float on host
inline float to_float(__half x) {
    return __half2float(x);
}
inline float to_float(__nv_bfloat16 x) {
#if defined(FLASHMOE_PLATFORM_HIP)
    return static_cast<float>(x);
#else
    return __bfloat162float(x);
#endif
}
inline float to_float(float x) { return x; }
inline float to_float(double x) { return static_cast<float>(x); }

// Convert float back to element type
template<typename T>
inline T from_float(float x);

template<> inline __half from_float<__half>(float x) { return __float2half(x); }
template<> inline float from_float<float>(float x) { return x; }
template<> inline double from_float<double>(float x) { return static_cast<double>(x); }
#if defined(FLASHMOE_PLATFORM_HIP)
template<> inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return hip_bfloat16(x); }
#else
template<> inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
#endif

// Host activations (matching device-side functors)
struct HostReLU {
    float operator()(float x) const { return x > 0.f ? x : 0.f; }
};
struct HostSiLu {
    float operator()(float x) const { return x / (1.0f + std::exp(-x)); }
};
struct HostGELU {
    float operator()(float x) const {
        constexpr float kAlpha = 0.7978845608f;
        constexpr float kBeta = 0.044715f;
        float inner = kAlpha * (x + kBeta * x * x * x);
        return x * 0.5f * (1.0f + std::tanh(inner));
    }
};
struct HostIdentity {
    float operator()(float x) const { return x; }
};

// CPU GEMM: C[M,N] = Act(A[M,K] @ B[N,K].T + bias[N])
// B is stored row-major [N,K], so B.T[K,N] means C[i,j] = sum_k A[i,k]*B[j,k]
template<typename Activation, typename Element, typename ElementC>
void gemm_bias_act(
    const Element* a, const Element* b, const ElementC* bias,
    ElementC* c_ref,
    int M, int N, int K)
{
    Activation act{};
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += to_float(a[i * K + k]) * to_float(b[j * K + k]);
            }
            acc += to_float(bias[j]);
            acc = act(acc);
            c_ref[i * N + j] = from_float<ElementC>(acc);
        }
    }
}

// CPU Gated GEMM: out[M,N] = swishAlpha * Act(swishBeta * (A@B.T + bias)) * (A@BV.T + biasV)
template<typename Activation, typename Element, typename ElementC>
void gated_gemm_ref(
    const Element* a, const Element* b, const Element* bv,
    const ElementC* bias, const ElementC* biasV,
    ElementC* c_ref,
    int M, int N, int K,
    float swishAlpha, float swishBeta)
{
    Activation act{};
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // Gate path: A @ B.T + bias
            float gate_acc = 0.f;
            for (int k = 0; k < K; ++k) {
                gate_acc += to_float(a[i * K + k]) * to_float(b[j * K + k]);
            }
            gate_acc += to_float(bias[j]);
            gate_acc = swishAlpha * act(swishBeta * gate_acc);

            // Value path: A @ BV.T + biasV
            float val_acc = 0.f;
            for (int k = 0; k < K; ++k) {
                val_acc += to_float(a[i * K + k]) * to_float(bv[j * K + k]);
            }
            val_acc += to_float(biasV[j]);

            c_ref[i * N + j] = from_float<ElementC>(gate_acc * val_acc);
        }
    }
}

// Compare two arrays with isclose(rtol, atol), return error percentage
template<typename ElementC>
double compare_isclose(
    const ElementC* actual, const ElementC* expected,
    int count, float rtol, float atol)
{
    long matches = 0;
    for (int i = 0; i < count; ++i) {
        float a = to_float(actual[i]);
        float e = to_float(expected[i]);
        float diff = std::fabs(a - e);
        if (diff <= atol + rtol * std::fabs(e)) {
            ++matches;
        }
    }
    return (1.0 - static_cast<double>(matches) / static_cast<double>(count)) * 100.0;
}

// CPU softmax per row: out[i,j] = exp(x[i,j]) / sum_j exp(x[i,j])
// x is [rows, cols] row-major
template<typename Element>
void softmax_rows(const Element* x, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float max_val = -1e30f;
        for (int j = 0; j < cols; ++j) {
            float v = to_float(x[i * cols + j]);
            if (v > max_val) max_val = v;
        }
        float sum = 0.f;
        for (int j = 0; j < cols; ++j) {
            float v = std::exp(to_float(x[i * cols + j]) - max_val);
            out[i * cols + j] = v;
            sum += v;
        }
        float inv_sum = 1.f / sum;
        for (int j = 0; j < cols; ++j) {
            out[i * cols + j] *= inv_sum;
        }
    }
}

// Verbose compare: print first max_print mismatches with (row, col), actual, expected, diff
// Also prints per-tile error breakdown for diagnosing multi-tile MFMA issues
template<typename ElementC>
double compare_isclose_verbose(
    const ElementC* actual, const ElementC* expected,
    int M, int N, float rtol, float atol,
    int tile_m, int tile_n, int max_print = 20)
{
    const int count = M * N;
    long matches = 0;
    int printed = 0;

    // Per-tile error tracking
    const int tiles_m = (M + tile_m - 1) / tile_m;
    const int tiles_n = (N + tile_n - 1) / tile_n;
    const int num_tiles = tiles_m * tiles_n;
    std::vector<int> tile_total(num_tiles, 0);
    std::vector<int> tile_errors(num_tiles, 0);
    // Track max absolute error per tile
    std::vector<float> tile_max_err(num_tiles, 0.f);

    for (int i = 0; i < count; ++i) {
        float a = to_float(actual[i]);
        float e = to_float(expected[i]);
        float diff = std::fabs(a - e);

        int row = i / N;
        int col = i % N;
        int ti = row / tile_m;
        int tj = col / tile_n;
        int tile_idx = ti * tiles_n + tj;
        tile_total[tile_idx]++;

        if (diff <= atol + rtol * std::fabs(e)) {
            ++matches;
        } else {
            tile_errors[tile_idx]++;
            if (diff > tile_max_err[tile_idx]) tile_max_err[tile_idx] = diff;
            if (printed < max_print) {
                printf("  MISMATCH [%d,%d] (tile[%d,%d]): actual=%.6f expected=%.6f diff=%.6f\n",
                       row, col, ti, tj, a, e, diff);
                ++printed;
            }
        }
    }

    double error_pct = (1.0 - static_cast<double>(matches) / static_cast<double>(count)) * 100.0;

    // Print per-tile summary
    printf("\n  === Per-tile error breakdown (bM=%d, bN=%d) ===\n", tile_m, tile_n);
    for (int t = 0; t < num_tiles; ++t) {
        if (tile_errors[t] > 0 || num_tiles <= 16) {
            int ti = t / tiles_n;
            int tj = t % tiles_n;
            double te = (tile_total[t] > 0)
                ? (static_cast<double>(tile_errors[t]) / tile_total[t] * 100.0) : 0.0;
            printf("  tile[%d,%d]: %d/%d errors (%.1f%%) max_abs_err=%.6f\n",
                   ti, tj, tile_errors[t], tile_total[t], te, tile_max_err[t]);
        }
    }
    printf("  === Total: %.2f%% error (%ld/%d match) ===\n\n", error_pct, matches, count);

    return error_pct;
}

} // namespace host_ref

#endif // FLASHMOE_HOST_REFERENCE_CUH
