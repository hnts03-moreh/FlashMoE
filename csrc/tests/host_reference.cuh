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

} // namespace host_ref

#endif // FLASHMOE_HOST_REFERENCE_CUH
