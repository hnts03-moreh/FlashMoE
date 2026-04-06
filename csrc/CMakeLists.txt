# Copyright (c) 2025, Osayamen Jonathan Aimuyo
# All rights reserved.
#
# This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
# See the LICENSE file in the root directory for full terms.

cmake_minimum_required(VERSION 3.27)

# Add COMMAND_ECHO STDOUT to see the commands in standard out
# WAR to use updated gcc on NERSC machines
if(DEFINED ENV{NERSC_HOST})
    execute_process(COMMAND "which" "g++"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_VARIABLE CPP_COMP_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "which" "gcc"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_VARIABLE C_COMP_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_CXX_COMPILER ${CPP_COMP_PATH})
    set(CMAKE_C_COMPILER ${C_COMP_PATH})
endif ()

# -------------------------------------------------------
# Platform detection
# -------------------------------------------------------
set(FLASHMOE_PLATFORM "AUTO" CACHE STRING "GPU platform: CUDA, HIP, or AUTO")

if(FLASHMOE_PLATFORM STREQUAL "AUTO")
  find_program(_HIPCC_FOUND hipcc)
  if(_HIPCC_FOUND AND (DEFINED ENV{ROCM_PATH} OR EXISTS "/opt/rocm"))
    set(FLASHMOE_PLATFORM_DETECTED "HIP")
  else()
    set(FLASHMOE_PLATFORM_DETECTED "CUDA")
  endif()
elseif(FLASHMOE_PLATFORM STREQUAL "HIP" OR FLASHMOE_PLATFORM STREQUAL "CUDA")
  set(FLASHMOE_PLATFORM_DETECTED "${FLASHMOE_PLATFORM}")
else()
  message(FATAL_ERROR "FLASHMOE_PLATFORM must be AUTO, CUDA, or HIP")
endif()

message(STATUS "FlashMoE csrc platform: ${FLASHMOE_PLATFORM_DETECTED}")

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "HIP")
  project(flashmoe LANGUAGES CXX HIP)
else()
  project(flashmoe CUDA CXX)
endif()

# flags
add_library(flashmoe_warnings INTERFACE)

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  target_compile_options(flashmoe_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra
        -fno-strict-aliasing
        -Wno-unknown-pragmas
        -Wnull-dereference
        -Wnarrowing
        -Wno-switch
        -Wduplicated-branches
        -Wformat=2
        -Wno-unused-but-set-parameter
        -Wno-sign-compare
        >

        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall,-Wextra
        -Xcompiler=-fno-strict-aliasing
        -Xcompiler=-Wno-unknown-pragmas,-Wnull-dereference,-Wnarrowing
        -Xcompiler=-Wno-switch,-Wduplicated-branches,-Wformat=2
        -Xcompiler=-Wno-unused-but-set-parameter
        -Xcudafe --display_error_number
        -Xcompiler=-Wno-sign-compare
        >
  )
else()
  target_compile_options(flashmoe_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra
        -fno-strict-aliasing
        -Wno-unknown-pragmas
        -Wnull-dereference
        -Wnarrowing
        -Wno-switch
        -Wformat=2
        -Wno-unused-but-set-parameter
        -Wno-sign-compare
        >

        $<$<COMPILE_LANGUAGE:HIP>:
        -Wall -Wextra
        -fno-strict-aliasing
        -Wno-unknown-pragmas
        -Wnull-dereference
        -Wnarrowing
        -Wno-switch
        -Wformat=2
        -Wno-unused-but-set-parameter
        -Wno-sign-compare
        >
  )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(CMAKE_CUDA_ARCHITECTURES "native")
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
else()
  # HIP standard settings
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "HIP GPU architectures")
  endif()
endif()

# Query number of SMs (CUDA only)
if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
  include(SM)

  include(CheckCompilerFlag)
  check_compiler_flag(CUDA -t4 NVCC_THREADS)

  find_package(CUDAToolkit REQUIRED)
  string(REGEX MATCH "^[0-9]+" ARCH "${CMAKE_CUDA_ARCHITECTURES_NATIVE}") # xx-real -> xx
  math(EXPR GPU_ARCH "${ARCH} * 10" OUTPUT_FORMAT DECIMAL)
  message(STATUS "GPU 0 Compute Capability: ${ARCH}")

  set(ARCH_TAG "${ARCH}")
  if(ARCH GREATER_EQUAL 90)
      string(APPEND ARCH_TAG "a") # accelerated arch-specific instruction set
  endif()
  set(ENV{CUTLASS_NVCC_ARCHS} "${ARCH_TAG}")
else()
  # HIP: set architecture info for MI300X
  find_package(hip REQUIRED)
  # GPU_ARCH is used for FLASHMOE_ARCH define — use a representative value
  # gfx942 maps to CDNA3; we use 942 as the arch identifier
  set(GPU_ARCH 942)
  set(ARCH_TAG "gfx942")
endif()

set(ENV{CPM_SOURCE_CACHE} "./cmake/cache")

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  set(CCCL_ENABLE_UNSTABLE ON)
endif()

include(cmake/CPM.cmake)

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  # CCCL
  CPMAddPackage(
          NAME CCCL
          GITHUB_REPOSITORY nvidia/cccl
          VERSION 3.1.3
  )
  #CUTLASS
  CPMAddPackage(
          NAME CUTLASS
          GITHUB_REPOSITORY nvidia/cutlass
          GIT_TAG v4.4.1
          DOWNLOAD_ONLY TRUE
          OPTIONS
          "CUTLASS_NVCC_ARCHS=${ARCH_TAG}"
  )

  CPMAddPackage(
          NAME NVTX3
          GITHUB_REPOSITORY NVIDIA/NVTX
          GIT_TAG v3.1.1-c-cpp
          GIT_SHALLOW TRUE
  )

  # MatX
  set(MATX_NVTX_FLAGS ON)
  CPMAddPackage(
          NAME MATX
          GITHUB_REPOSITORY NVIDIA/MatX
          GIT_TAG v0.9.4
  )
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Set -DCMAKE_BUILD_TYPE=Release (or Debug).")
endif()

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  ## NVSHMEM
  set(ENV{NVSHMEM_BUILD_LTOIR_LIBRARY} ON)
  set(NVSHMEM_BUILD_LTOIR_LIBRARY ON)
  find_package(NVSHMEM REQUIRED)
  # cuBLASDx
  find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG)
  find_package(mathdx REQUIRED COMPONENTS curanddx CONFIG)
  include(CheckCompilerFlag)
else()
  # ROCm equivalents (optional — may not be installed yet)
  # ROCSHMEM installs cmake configs under /opt/rocm/lib/cmake/rocshmem/
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
  find_package(rocshmem QUIET)
  # ROCSHMEM exports target as roc::rocshmem (not rocshmem::rocshmem)
  if(TARGET roc::rocshmem)
    message(STATUS "ROCSHMEM found")
    set(FLASHMOE_HAS_ROCSHMEM TRUE)
  else()
    message(STATUS "ROCSHMEM not found — distributed tests will be skipped")
    set(FLASHMOE_HAS_ROCSHMEM FALSE)
  endif()

  # rocTX for profiling
  find_library(ROCTX_LIB roctx64 HINTS ENV ROCM_PATH PATH_SUFFIXES lib)
endif()

# -------------------------------------------------------
# Helper functions — platform-specific target configuration
# -------------------------------------------------------
if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")

function(flash_target_test tgt)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvtx3)
    target_link_libraries(${tgt} PRIVATE atomic)
    if(TARGET CCCL::CCCL)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    if(DEFINED CUTLASS_SOURCE_DIR AND EXISTS "${CUTLASS_SOURCE_DIR}/include")
        target_include_directories(${tgt} SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    endif()
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-t0> #use sm_xxa for xx >= 90
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
    )
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0;-g;-G;--expt-relaxed-constexpr;-Xptxas=-v>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--expt-relaxed-constexpr;-Xptxas=-v>
    )
endfunction()

function(flash_target_common tgt)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cuda_driver atomic)
    if(TARGET nvtx3-cpp)
        target_link_libraries(${tgt} PRIVATE nvtx3-cpp)
    endif ()
    if(TARGET CCCL::CCCL)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    if(DEFINED CUTLASS_SOURCE_DIR AND EXISTS "${CUTLASS_SOURCE_DIR}/include")
        target_include_directories(${tgt} SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    endif()
    if(TARGET matx::matx)
        target_link_libraries(${tgt} PRIVATE matx::matx)
    endif()
    target_link_libraries(${tgt} PRIVATE mathdx::cublasdx mathdx::curanddx)
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-t0>
    )
    # Debug/Release opts
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
    )
    target_compile_definitions(${tgt} PRIVATE
            FLASHMOE_ARCH=${GPU_ARCH}
            NUM_SMS=${CUDA_NUM_SMS}
            FLASHMOE_NVTX=1
    )
endfunction()

function(flash_target_lto tgt)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvml atomic)
    target_link_libraries(${tgt} PRIVATE flashmoe_warnings)
    if(TARGET nvtx3-cpp)
        target_link_libraries(${tgt} PRIVATE nvtx3-cpp)
    endif ()
    if(TARGET CCCL::CCCL)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    if(DEFINED CUTLASS_SOURCE_DIR AND EXISTS "${CUTLASS_SOURCE_DIR}/include")
        target_include_directories(${tgt} SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    endif()
    if(TARGET matx::matx)
        target_link_libraries(${tgt} PRIVATE matx::matx)
    endif()
    target_link_libraries(${tgt} PRIVATE mathdx::cublasdx mathdx::curanddx)
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> #--Werror=all-warnings
            $<$<COMPILE_LANGUAGE:CUDA>:-t0; -lineinfo>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
    )
    # Debug/Release opts
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v; -Xcudafe --display_error_number>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=lto_${ARCH_TAG}>
    )
    target_compile_definitions(${tgt} PRIVATE
            FLASHMOE_ARCH=${GPU_ARCH}
            NUM_SMS=${CUDA_NUM_SMS}
            FLASHMOE_NVTX=1
    )
endfunction()

else()
# -------------------------------------------------------
# HIP helper functions
# -------------------------------------------------------

function(flash_target_test tgt)
    target_include_directories(${tgt} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(${tgt} PRIVATE hip::device atomic)
    target_link_libraries(${tgt} PRIVATE flashmoe_warnings)
    # Note: --offload-arch is set automatically by CMake via CMAKE_HIP_ARCHITECTURES
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:HIP>>:-O0;-g;-ggdb>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:HIP>>:-O3>
    )
    target_compile_definitions(${tgt} PRIVATE
            FLASHMOE_PLATFORM_HIP=1
            FLASHMOE_USE_MATH_COMPAT=1
            FLASHMOE_ARCH=${GPU_ARCH}
    )
endfunction()

function(flash_target_common tgt)
    target_include_directories(${tgt} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(${tgt} PRIVATE hip::device atomic)
    target_link_libraries(${tgt} PRIVATE flashmoe_warnings)
    if(ROCTX_LIB)
        target_link_libraries(${tgt} PRIVATE ${ROCTX_LIB})
    endif()
    # Note: --offload-arch is set automatically by CMake via CMAKE_HIP_ARCHITECTURES
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:HIP>>:-O0;-g;-ggdb>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:HIP>>:-O3>
    )
    target_compile_definitions(${tgt} PRIVATE
            FLASHMOE_PLATFORM_HIP=1
            FLASHMOE_USE_MATH_COMPAT=1
            FLASHMOE_ARCH=${GPU_ARCH}
            FLASHMOE_NVTX=1
    )
endfunction()

function(flash_target_lto tgt)
    target_include_directories(${tgt} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(${tgt} PRIVATE hip::device atomic)
    target_link_libraries(${tgt} PRIVATE flashmoe_warnings)
    if(ROCTX_LIB)
        target_link_libraries(${tgt} PRIVATE ${ROCTX_LIB})
    endif()
    # Note: --offload-arch is set automatically by CMake via CMAKE_HIP_ARCHITECTURES
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>
    )
    target_link_options(${tgt} PRIVATE -fgpu-rdc)
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:HIP>>:-O0;-g;-ggdb>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:HIP>>:-O3>
    )
    target_compile_definitions(${tgt} PRIVATE
            FLASHMOE_PLATFORM_HIP=1
            FLASHMOE_USE_MATH_COMPAT=1
            FLASHMOE_ARCH=${GPU_ARCH}
            FLASHMOE_NVTX=1
    )
endfunction()

endif() # platform helper functions

# -------------------------------------------------------
# Test executables
# -------------------------------------------------------
if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  # CUDA test targets (original)
  add_executable(testFlashMoE tests/flashmoe.cu)
  set_target_properties(testFlashMoE PROPERTIES
          CUDA_SEPARABLE_COMPILATION ON
          CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  flash_target_lto(testFlashMoE)

  # WAR for build failure
  if (NOT DEFINED ENV{NVSHMEM_LIB_HOME})
      message(FATAL_ERROR "NVSHMEM_LIB_HOME not defined")
  endif ()
  set_target_properties(nvshmem::nvshmem_host PROPERTIES
          IMPORTED_LOCATION_RELEASE "$ENV{NVSHMEM_LIB_HOME}/libnvshmem_host.so"
  )
  target_link_libraries(testFlashMoE PRIVATE nvshmem::nvshmem_host nvshmem::nvshmem_device)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(testFlashMoE PRIVATE MPI::MPI_CXX)

  add_executable(testScheduler tests/scheduler.cu)
  flash_target_test(testScheduler)

  add_executable(testGEMM tests/gemm.cu)
  flash_target_common(testGEMM)

  set(PYBIND11_FINDPYTHON ON)
  CPMAddPackage(
          NAME pybind11
          GITHUB_REPOSITORY pybind/pybind11
          VERSION 3.0.2
          DOWNLOAD_ONLY TRUE
  )
  add_executable(playground tests/playground.cu)
  flash_target_common(playground)
  find_package(Python3 COMPONENTS Development)
  target_include_directories(playground PRIVATE "${pybind11_SOURCE_DIR}/include" ${Python3_INCLUDE_DIRS})
  target_link_libraries(playground PRIVATE nvshmem::nvshmem_host nvshmem::nvshmem_device)

  add_executable(testGate tests/gate.cu)
  flash_target_common(testGate)

  add_executable(testCombine tests/combine.cu)
  flash_target_common(testCombine)

  add_executable(gemmMNK tests/gemmMNK.cu)
  flash_target_common(gemmMNK)

  add_executable(testGatedGEMM tests/gatedGEMM.cu)
  flash_target_common(testGatedGEMM)

else()
  # -------------------------------------------------------
  # HIP test targets
  # -------------------------------------------------------
  # Detect number of Compute Units (CUs) for MI300X — equivalent to NUM_SMS on CUDA.
  # MI300X has 304 CUs total (8 XCDs x 38 CUs/XCD). Use a sensible default.
  if(NOT DEFINED NUM_CUS)
    # Try to query via rocm-smi; fall back to 304 for MI300X
    execute_process(
      COMMAND bash -c "rocm-smi --showcomputeunit 2>/dev/null | grep -oP '\\d+' | head -1"
      OUTPUT_VARIABLE _DETECTED_CUS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _ROCMSMI_RC
    )
    if(_ROCMSMI_RC EQUAL 0 AND _DETECTED_CUS)
      set(NUM_CUS "${_DETECTED_CUS}")
    else()
      set(NUM_CUS 304) # MI300X default
    endif()
  endif()
  message(STATUS "FlashMoE HIP build: NUM_CUS (NUM_SMS) = ${NUM_CUS}")

  # rocBLASDx include path
  set(ROCBLASDX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rocblasdx/include")

  # Helper: configure a HIP test target with standard defines
  function(flash_hip_test_target tgt)
    flash_target_test(${tgt})
    target_include_directories(${tgt} SYSTEM PRIVATE "${ROCBLASDX_INCLUDE_DIR}")
    target_compile_definitions(${tgt} PRIVATE NUM_SMS=${NUM_CUS})
  endfunction()

  function(flash_hip_common_target tgt)
    flash_target_common(${tgt})
    target_include_directories(${tgt} SYSTEM PRIVATE "${ROCBLASDX_INCLUDE_DIR}")
    target_compile_definitions(${tgt} PRIVATE NUM_SMS=${NUM_CUS})
  endfunction()

  function(flash_hip_lto_target tgt)
    flash_target_lto(${tgt})
    target_include_directories(${tgt} SYSTEM PRIVATE "${ROCBLASDX_INCLUDE_DIR}")
    target_compile_definitions(${tgt} PRIVATE NUM_SMS=${NUM_CUS})
  endfunction()

  # hipRAND for test RNG
  find_package(hiprand QUIET)
  if(NOT TARGET hip::hiprand AND NOT hiprand_FOUND)
    find_library(HIPRAND_LIB hiprand HINTS ENV ROCM_PATH PATH_SUFFIXES lib)
  endif()

  # hipcub for scheduler test
  find_package(hipcub QUIET)

  # --- Scheduler test (no BLAS dependency) ---
  add_executable(testScheduler tests/scheduler.cu)
  set_source_files_properties(tests/scheduler.cu PROPERTIES LANGUAGE HIP)
  flash_hip_test_target(testScheduler)
  if(TARGET hipcub::hipcub)
    target_link_libraries(testScheduler PRIVATE hipcub::hipcub)
  endif()

  # --- GEMM tests ---
  add_executable(testGEMM tests/gemm.cu)
  set_source_files_properties(tests/gemm.cu PROPERTIES LANGUAGE HIP)
  flash_hip_common_target(testGEMM)
  if(TARGET hip::hiprand)
    target_link_libraries(testGEMM PRIVATE hip::hiprand)
  elseif(HIPRAND_LIB)
    target_link_libraries(testGEMM PRIVATE ${HIPRAND_LIB})
  endif()

  add_executable(gemmMNK tests/gemmMNK.cu)
  set_source_files_properties(tests/gemmMNK.cu PROPERTIES LANGUAGE HIP)
  flash_hip_common_target(gemmMNK)
  if(TARGET hip::hiprand)
    target_link_libraries(gemmMNK PRIVATE hip::hiprand)
  elseif(HIPRAND_LIB)
    target_link_libraries(gemmMNK PRIVATE ${HIPRAND_LIB})
  endif()

  add_executable(testGatedGEMM tests/gatedGEMM.cu)
  set_source_files_properties(tests/gatedGEMM.cu PROPERTIES LANGUAGE HIP)
  flash_hip_common_target(testGatedGEMM)
  if(TARGET hip::hiprand)
    target_link_libraries(testGatedGEMM PRIVATE hip::hiprand)
  elseif(HIPRAND_LIB)
    target_link_libraries(testGatedGEMM PRIVATE ${HIPRAND_LIB})
  endif()

  # --- Gate test ---
  add_executable(testGate tests/gate.cu)
  set_source_files_properties(tests/gate.cu PROPERTIES LANGUAGE HIP)
  flash_hip_common_target(testGate)
  if(TARGET hip::hiprand)
    target_link_libraries(testGate PRIVATE hip::hiprand)
  elseif(HIPRAND_LIB)
    target_link_libraries(testGate PRIVATE ${HIPRAND_LIB})
  endif()

  # --- Combine test ---
  add_executable(testCombine tests/combine.cu)
  set_source_files_properties(tests/combine.cu PROPERTIES LANGUAGE HIP)
  flash_hip_common_target(testCombine)
  if(TARGET hip::hiprand)
    target_link_libraries(testCombine PRIVATE hip::hiprand)
  elseif(HIPRAND_LIB)
    target_link_libraries(testCombine PRIVATE ${HIPRAND_LIB})
  endif()

  # --- FlashMoE E2E test (requires ROCSHMEM + MPI) ---
  add_executable(testFlashMoE tests/flashmoe.cu)
  set_source_files_properties(tests/flashmoe.cu PROPERTIES LANGUAGE HIP)
  # Use ROCSHMEM device stubs to avoid RDC device bitcode linking issues.
  # ROCSHMEM host API (init/finalize/malloc) works; device-side SHMEM calls
  # are stubbed out. For real distributed communication, ROCSHMEM must be
  # rebuilt with device bitcode support for -fgpu-rdc linking.
  option(FLASHMOE_SHMEM_DEVICE_STUBS "Stub ROCSHMEM device calls (avoids RDC link issues)" ON)
  if(FLASHMOE_SHMEM_DEVICE_STUBS)
    flash_hip_common_target(testFlashMoE)
  else()
    flash_hip_lto_target(testFlashMoE)
  endif()
  if(FLASHMOE_HAS_ROCSHMEM)
    target_link_libraries(testFlashMoE PRIVATE roc::rocshmem)
    target_compile_definitions(testFlashMoE PRIVATE
      FLASHMOE_HAS_SHMEM=1
      FLASHMOE_SKIP_WG_INIT=1
    )
    if(FLASHMOE_SHMEM_DEVICE_STUBS)
      target_compile_definitions(testFlashMoE PRIVATE FLASHMOE_SHMEM_DEVICE_STUBS=1)
    endif()
    find_package(MPI QUIET COMPONENTS CXX)
    if(TARGET MPI::MPI_CXX)
      target_link_libraries(testFlashMoE PRIVATE MPI::MPI_CXX)
    endif()
  endif()

  # --- GEMM diagnostic (128+ tile MFMA debug) ---
  add_executable(diagGEMM tests/diagGEMM.cu)
  set_source_files_properties(tests/diagGEMM.cu PROPERTIES LANGUAGE HIP)
  flash_hip_common_target(diagGEMM)
  if(TARGET hip::hiprand)
    target_link_libraries(diagGEMM PRIVATE hip::hiprand)
  elseif(HIPRAND_LIB)
    target_link_libraries(diagGEMM PRIVATE ${HIPRAND_LIB})
  endif()

  # --- Precision validation (FP16/BF16/FP32 MFMA accuracy) ---
  add_executable(testPrecision tests/testPrecision.cu)
  set_source_files_properties(tests/testPrecision.cu PROPERTIES LANGUAGE HIP)
  flash_hip_common_target(testPrecision)
  if(TARGET hip::hiprand)
    target_link_libraries(testPrecision PRIVATE hip::hiprand)
  elseif(HIPRAND_LIB)
    target_link_libraries(testPrecision PRIVATE ${HIPRAND_LIB})
  endif()

  # --- Playground (minimal, no GPU deps) ---
  add_executable(playground tests/playground.cu)
  set_source_files_properties(tests/playground.cu PROPERTIES LANGUAGE HIP)
  flash_hip_common_target(playground)

endif()
