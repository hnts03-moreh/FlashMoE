cmake_minimum_required(VERSION 3.27)

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

message(STATUS "FlashMoE JIT platform: ${FLASHMOE_PLATFORM_DETECTED}")

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "HIP")
  project(flashmoe_v010 LANGUAGES CXX HIP)
else()
  project(flashmoe_v010 LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
else()
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "HIP GPU architectures")
  endif()
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

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
else()
  find_package(hip REQUIRED)
endif()

if (NOT DEFINED GENERATED_SRC)
  message(FATAL_ERROR "Need -DGENERATED_SRC")
endif()

if (NOT DEFINED TARGET_MODULE_NAME)
  message(FATAL_ERROR "Need -DTARGET_MODULE_NAME")
endif()

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  if (NOT DEFINED ARCH)
    message(FATAL_ERROR "Need -DARCH")
  endif()

  if(${ARCH} LESS 70)
    message(FATAL_ERROR "Unsupported ARCH")
  endif()

  math(EXPR GPU_ARCH "${ARCH} * 10" OUTPUT_FORMAT DECIMAL)
  set(ARCH_TAG "${ARCH}")
  if(ARCH GREATER_EQUAL 90)
      string(APPEND ARCH_TAG "a") # accelerated arch-specific instruction set
  endif()
else()
  # HIP: ARCH parameter is the gfx target (e.g. gfx942)
  if(NOT DEFINED ARCH)
    set(ARCH "gfx942")
  endif()
  set(GPU_ARCH 942)
  set(ARCH_TAG "${ARCH}")
endif()

# ---- CPM ----
include(${CMAKE_CURRENT_LIST_DIR}/cmake/CPM.cmake)

#pybind11
set(PYBIND11_FINDPYTHON ON)
CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  VERSION 3.0.2
)
if(NOT COMMAND pybind11_add_module)
    message(FATAL_ERROR "pybind11 not found")
endif()
pybind11_add_module(${TARGET_MODULE_NAME} MODULE ${GENERATED_SRC})

if(FLASHMOE_PLATFORM_DETECTED STREQUAL "CUDA")
  set(ENV{NVSHMEM_BUILD_LTOIR_LIBRARY} ON)
  set(NVSHMEM_BUILD_LTOIR_LIBRARY ON)
  CPMAddPackage(
          NAME flashmoe
          GITHUB_REPOSITORY osayamenja/flashmoe
          GIT_TAG v0.1.2
  )

  target_link_libraries(${TARGET_MODULE_NAME} PRIVATE flashmoe::flashmoe)

  FlashMoESetRDC(${TARGET_MODULE_NAME})
  FlashMoEAddOptions(${TARGET_MODULE_NAME})

  target_compile_options(${TARGET_MODULE_NAME} PRIVATE
              $<$<COMPILE_LANGUAGE:CXX>:-O3>
              $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
              $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr; -Xptxas -v; -t0>
              $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
              $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=lto_${ARCH_TAG}>
  )

  target_link_libraries(${TARGET_MODULE_NAME} PRIVATE CUDA::cudart flashmoe_warnings)
  target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE
              FLASHMOE_ARCH=${GPU_ARCH}
  )

else()
  # HIP JIT build path
  # For now, link against the local flashmoe headers (not CPM package)
  # since the ROCm-ported version is being developed locally.
  target_include_directories(${TARGET_MODULE_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../csrc/include
  )

  target_link_libraries(${TARGET_MODULE_NAME} PRIVATE hip::device flashmoe_warnings atomic)

  # ROCSHMEM (optional)
  find_package(rocshmem QUIET)
  if(TARGET rocshmem::rocshmem)
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE rocshmem::rocshmem)
    target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE FLASHMOE_HAS_ROCSHMEM=1)
  endif()

  # rocTX for profiling
  find_library(ROCTX_LIB roctx64 HINTS ENV ROCM_PATH PATH_SUFFIXES lib)
  if(ROCTX_LIB)
    target_link_libraries(${TARGET_MODULE_NAME} PRIVATE ${ROCTX_LIB})
  endif()

  target_compile_options(${TARGET_MODULE_NAME} PRIVATE
              $<$<COMPILE_LANGUAGE:CXX>:-O3>
              $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${CMAKE_HIP_ARCHITECTURES}>
              $<$<COMPILE_LANGUAGE:HIP>:-O3>
              $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>
  )
  target_link_options(${TARGET_MODULE_NAME} PRIVATE -fgpu-rdc)

  target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE
              FLASHMOE_PLATFORM_HIP=1
              FLASHMOE_USE_MATH_COMPAT=1
              FLASHMOE_ARCH=${GPU_ARCH}
  )
endif()
