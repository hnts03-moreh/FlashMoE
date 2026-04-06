//
// Created by osayamen on 12/22/25.
//

// place to experiment

#include <cstdio>
#include <cuda_fp16.h>

#include "../include/flashmoe/tile.cuh"

int main() {
  constexpr flashmoe::Converter<float2, __half2> loadOp{};
  __half2_raw a{__half2ushort_rn(__half{1.f}),
    __half2ushort_rn(__half{1.f})};
  const auto c = loadOp(a);

}
