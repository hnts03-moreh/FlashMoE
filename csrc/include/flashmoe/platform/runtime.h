/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_PLATFORM_RUNTIME_H
#define FLASHMOE_PLATFORM_RUNTIME_H

#include "platform.h"

// -------------------------------------------------------
// Runtime API headers
// -------------------------------------------------------
#if defined(FLASHMOE_PLATFORM_HIP)
#  include <hip/hip_runtime.h>
#  include <hip/hip_runtime_api.h>
#else
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#endif

// -------------------------------------------------------
// Type aliases — compile-time zero-overhead mappings
// -------------------------------------------------------
#if defined(FLASHMOE_PLATFORM_HIP)

// Stream / Event
using gpuStream_t     = hipStream_t;
using gpuEvent_t      = hipEvent_t;

// Error type
using gpuError_t      = hipError_t;
#define GPU_SUCCESS    hipSuccess

// Memory
#define gpuMalloc            hipMalloc
#define gpuMallocAsync       hipMallocAsync
#define gpuFree              hipFree
#define gpuFreeAsync         hipFreeAsync
#define gpuMemcpy            hipMemcpy
#define gpuMemcpyAsync       hipMemcpyAsync
#define gpuMemset            hipMemset
#define gpuMemsetAsync       hipMemsetAsync
#define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMallocManaged        hipMallocManaged
#define gpuGetDeviceProperties  hipGetDeviceProperties

// Stream / Event management
#define gpuStreamCreate         hipStreamCreate
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamDestroy        hipStreamDestroy
#define gpuStreamSynchronize    hipStreamSynchronize
#define gpuEventCreate          hipEventCreate
#define gpuEventRecord          hipEventRecord
#define gpuEventSynchronize     hipEventSynchronize
#define gpuEventElapsedTime     hipEventElapsedTime
#define gpuEventDestroy         hipEventDestroy
#define gpuStreamNonBlocking    hipStreamNonBlocking

// Device management
#define gpuSetDevice             hipSetDevice
#define gpuGetDevice             hipGetDevice
#define gpuGetDeviceCount        hipGetDeviceCount
#define gpuDeviceSynchronize     hipDeviceSynchronize
#define gpuDeviceGetAttribute    hipDeviceGetAttribute
#define gpuDeviceReset           hipDeviceReset

// Device attributes
#define gpuDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define gpuDevAttrMultiProcessorCount         hipDeviceAttributeMultiprocessorCount
#define gpuDevAttrComputeCapabilityMajor      hipDeviceAttributeComputeCapabilityMajor
#define gpuDevAttrComputeCapabilityMinor      hipDeviceAttributeComputeCapabilityMinor

// Kernel launch attributes
#define gpuFuncSetAttribute      hipFuncSetAttribute
#define gpuFuncAttributeMaxDynamicSharedMemorySize \
        hipFuncAttributeMaxDynamicSharedMemorySize
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor \
        hipOccupancyMaxActiveBlocksPerMultiprocessor

// Error handling
#define gpuGetErrorName    hipGetErrorName
#define gpuGetErrorString  hipGetErrorString
#define gpuGetLastError    hipGetLastError
#define gpuPeekAtLastError hipPeekAtLastError

#else // CUDA

using gpuStream_t     = cudaStream_t;
using gpuEvent_t      = cudaEvent_t;

using gpuError_t      = cudaError_t;
#define GPU_SUCCESS    cudaSuccess

#define gpuMalloc            cudaMalloc
#define gpuMallocAsync       cudaMallocAsync
#define gpuFree              cudaFree
#define gpuFreeAsync         cudaFreeAsync
#define gpuMemcpy            cudaMemcpy
#define gpuMemcpyAsync       cudaMemcpyAsync
#define gpuMemset            cudaMemset
#define gpuMemsetAsync       cudaMemsetAsync
#define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMallocManaged        cudaMallocManaged
#define gpuGetDeviceProperties  cudaGetDeviceProperties

#define gpuStreamCreate         cudaStreamCreate
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamDestroy        cudaStreamDestroy
#define gpuStreamSynchronize    cudaStreamSynchronize
#define gpuEventCreate          cudaEventCreate
#define gpuEventRecord          cudaEventRecord
#define gpuEventSynchronize     cudaEventSynchronize
#define gpuEventElapsedTime     cudaEventElapsedTime
#define gpuEventDestroy         cudaEventDestroy
#define gpuStreamNonBlocking    cudaStreamNonBlocking

#define gpuSetDevice             cudaSetDevice
#define gpuGetDevice             cudaGetDevice
#define gpuGetDeviceCount        cudaGetDeviceCount
#define gpuDeviceSynchronize     cudaDeviceSynchronize
#define gpuDeviceGetAttribute    cudaDeviceGetAttribute
#define gpuDeviceReset           cudaDeviceReset

#define gpuDevAttrMaxSharedMemoryPerBlockOptin cudaDevAttrMaxSharedMemoryPerBlockOptin
#define gpuDevAttrMultiProcessorCount         cudaDevAttrMultiProcessorCount
#define gpuDevAttrComputeCapabilityMajor      cudaDevAttrComputeCapabilityMajor
#define gpuDevAttrComputeCapabilityMinor      cudaDevAttrComputeCapabilityMinor

#define gpuFuncSetAttribute      cudaFuncSetAttribute
#define gpuFuncAttributeMaxDynamicSharedMemorySize \
        cudaFuncAttributeMaxDynamicSharedMemorySize
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor

#define gpuGetErrorName    cudaGetErrorName
#define gpuGetErrorString  cudaGetErrorString
#define gpuGetLastError    cudaGetLastError
#define gpuPeekAtLastError cudaPeekAtLastError

#endif // FLASHMOE_PLATFORM_HIP

// -------------------------------------------------------
// Unified error-check macro
// -------------------------------------------------------
#include <cstdio>
#include <cstdlib>

#if !defined(CHECK_GPU)
#  define CHECK_GPU(e)                                           \
   do {                                                          \
       gpuError_t code = (e);                                    \
       if (code != GPU_SUCCESS) {                                \
           fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",          \
               __FILE__, __LINE__, #e,                           \
               gpuGetErrorName(code),                            \
               gpuGetErrorString(code));                         \
           fflush(stderr);                                       \
           exit(1);                                              \
       }                                                         \
   } while (0)
#endif

// Legacy compatibility — existing code uses CHECK_CUDA
#if !defined(CHECK_CUDA)
#  define CHECK_CUDA(e) CHECK_GPU(e)
#endif

#endif // FLASHMOE_PLATFORM_RUNTIME_H
