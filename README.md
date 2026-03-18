# FlashMoE: Fast Distributed MoE in a Single Kernel [NeurIPS'25]
FlashMoE is the first fully fused Distributed MoE system that achieves high tensor core utilization
by eliminating kernel boundaries and enabling fine-grained overlap of communication and computation.
We provide high-performance single- and multi-node EP inference 
and work seamlessly with CUDA graphs. See paper [here](https://arxiv.org/abs/2506.04667).

## Table of Contents
1. [Motivation](#problem-moe-bottlenecks-in-inference)
2. [Our Solution](#our-solution-complete-kernel-fusion)
3. [Installation](#installation)
4. [QuickStart](#-python-quickstart)
5. [Performance Results](#-performance-results)
6. [Running Benchmarks](#run-benchmark-c)

## Problem: MoE Bottlenecks in Inference

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_motivation.png" alt="Opportunity" width="800"/><br>
      <em>Figure 1: Opportunity. MoE takes 67%-95% of inference runtime.</em>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_tensor_core_idle_time.png" alt="Tensor core utilization" width="600"/><br>
      <em>Figure 2: Tensor core Utilization. y-axis is percentage of MoE runtime that tensor cores are inactive.</em>
    </td>
  </tr>
</table>

Distributed Mixture-of-Experts (DMoE) is an extremely demanding workload, both compute- and communication-intensive,
accounting for up to **95% of total inference runtime** (Figure 1). 

This makes DMoE the primary bottleneck in distributed inference and a critical target for optimization.

However, existing implementations leave significant performance untapped, achieving only **26% tensor core utilization** (Figure 2).

We identify three key sources of inefficiency:

1. **Exposed communication on the critical path**  
2. **Straggler-induced delays** from load imbalance  
3. **System overheads** from dynamic token routing (e.g., metadata management, inputs preprocessing for compute operators like GroupedGEMM)

As a result, GPUs spend the majority of time stalled, with only **26% of runtime utilizing tensor cores**.

## Our Solution: Complete Kernel Fusion

<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_Arch_title.png" width="700" alt="">
<p><em>Figure 3: FlashMoE Architecture</em></p>
</div>

We address these inefficiencies through **complete kernel fusion**, enabling:

1. **Fine-grained overlap of communication and computation** at tile granularity  
2. **Latency hiding of preprocessing and system overheads** via SM specialization  
3. **Exploitation of task locality at scale**, allowing SMs to execute ready tasks out-of-order, minimizing idle and boosting resource utilization.

In contrast, existing implementations rely on tens to hundreds of serialized kernels, enforcing strict execution order
and limiting _task locality_.

This results in unnecessary stalls—for example, during collective synchronization (AllGather, ReduceScatter, AllToAll),
where GPUs idle waiting for stragglers instead of executing independent compute tasks.

## Our Work
We present **FlashMoE** (Figure 3), the first **fully fused Distributed MoE system**.

FlashMoE is a high-throughput, portable system that fuses:
- MoE Dispatch  
- Expert Computation (Gated MLP or standard MLP)  
- MoE Combine  

into a **single tile-pipelined persistent kernel**.

At its core, FlashMoE embeds an **Operating System within the kernel**, enabling concurrent scheduling and execution,
thereby hiding system and communication latency. 

FlashMoE is built from the ground up in **CUDA C++**, with selective inline PTX.
It leverages:

- [cuBLASDx](https://docs.nvidia.com/cuda/cublasdx/) for device-side high-performance compute  
- [NVSHMEM](https://developer.nvidia.com/nvshmem) for asynchronous, device-initiated communication  
- [CCCL](https://github.com/nvidia/cccl) and [CUTLASS](https://github.com/NVIDIA/cutlass) for critical infrastructure

### 🏎️ Portability

We support 
- SM70 and above GPUs. Boosting compute performance for Hopper and Blackwell is on the roadmap.
- NVLink and multi-node RDMA (EFA, IBGDA, libfabric as NVSHMEM [supports](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#hardware-requirements)).
- FP16, BF16, FP32 (TF32) and FP64. FP8 and even lower precision types are on the roadmap (we welcome contributions!)

## Requirements
- CUDA toolkit
- C++20
- ninja (`sudo apt install ninja-build`)
- CMake (>= 3.28)

### Hardware Requirements
- GPU architecture of at least SM 70. 
- A P2P GPU interconnect (NVLink, some PCIe and GPUDirect RDMA). NVSHMEM will fail if this criterion is not met.

## Installation
### cuBLASDx
- Download from [here](https://developer.nvidia.com/cublasdx-downloads) and save in `<your_directory>`, e.g `~/.local`.
- export `MATHDX_ROOT=<your_directory>/nvidia-<...>/mathdx/yy.mm/`

### NVSHMEM
- Install as directed [here](https://developer.nvidia.com/nvshmem-downloads).
- export `NVSHMEM_LIB_HOME=/usr/lib/x86_64-linux-gnu/nvshmem/<12 or 13>`. Do confirm this directory exists!

> 👉 Tip: add `MATHDX_ROOT=...` and `NVSHMEM_LIB_HOME=...` to `.bashrc`

## 🚀 Python QuickStart
```bash
pip install flashmoe-py[cu12] # or cu13
```
## Using Python API
```python
# quick.py
import argparse
import cuda.core.experimental as cuda
import flashmoe
import torch

device_id = flashmoe.get_local_rank()
dev = cuda.Device(device_id)
dev.set_current()
stream = dev.create_stream()
stream_ptr = int(stream.handle)
arch = int(dev.arch) * 10
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-init", action="store_true")
    args = parser.parse_args()
    
    if args.torch_init:
        import torch.distributed as dist, os
        assert os.environ.get("LOCAL_RANK") is not None, "need to launch with torchrun if set with torch_init=True"
        world_size = os.environ.get("WORLD_SIZE")
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=int(os.environ.get("RANK")),
            world_size=world_size,
            device_id=device
        )
        
    # Llama4-Scout-17B-16E shapes
    tokens_per_rank = 1024
    token_dim = 5120
    ffn_size = 8192
    num_experts = 16
    k = 1
    
    # define model config
    mlp_type = flashmoe.MLPType.GATED # Gated MLP
    data_type = flashmoe.DataType.BF16
    act_type = flashmoe.ActivationType.SILU
    
    init_args = flashmoe.InitArgs(data_type=data_type,
        mlp_type=mlp_type, act_type=act_type,
        tokens_per_rank=tokens_per_rank, token_dim=token_dim,
        ffn_size=ffn_size, num_experts=num_experts,
        top_k=k, gpu_arch=arch, stream_ptr=stream_ptr, device_id=device_id)
    
    # initialize flashmoe and fused router
    flash_handle = flashmoe.initialize(init_args)
    router_handle = flashmoe.router.initialize(init_args)

    # call forward of fused router
    router_forward_args = ... # see quickstart.py for an example
    flashmoe.router.forward(router_handle, flash_handle, router_forward_args)
    # call forward of FlashMoE
    flashmoe_forward_args = ... # see quickstart.py for an example
    flashmoe.forward(flash_handle, flashmoe_forward_args) # single kernel for Dispatch + Experts + Combine
    
    # call finalize
    flashmoe.finalize(flash_handle, stream_ptr)
    flashmoe.router.finalize(router_handle, stream_ptr)
    stream.close()
    if args.torch_init:
        import torch.distributed as dist
        dist.destroy_process_group()
```
### Running the Python Program
With torchrun:
```shell
torchrun --nproc_per_node=<number of GPUs> quick.py --torch-init
```
With MPI:
```shell
pip install mpi4py
mpiexec -n <number of GPUs> python3 quick.py
```

## Use C++ API (header-only)
Add the following to your `CMakeLists.txt`
```CMake
set(CPM_SOURCE_CACHE
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cache"
        CACHE PATH "Shared CPM source cache"
)
set(CMAKE_CUDA_ARCHITECTURES "native") # or your own architecture

#...
CPMAddPackage(
  NAME flashmoe
  GITHUB_REPOSITORY osayamenja/flashmoe
  GIT_TAG v0.1.0
)

target_link_libraries(app PRIVATE flashmoe::flashmoe)

FlashMoESetRDC(app)
FlashMoEAddOptions(app)
```
and include the header file like below. See `csrc/tests/flashmoe.cu` for more usage details.
```cpp
#include <flashmoe/flashmoe.cuh>
```
---

### ✅ Roadmap
- [ ] Improve MMA for Hopper (WGMMA) and Blackwell (UTCMMA).
- [ ] FP8 support
- [ ] Shared experts
- [ ] AMD support

---

## 📊 Performance Results
- We measure with the EP+DP parallelism scheme.
- We compare against [COMET](https://github.com/bytedance/flux) (MLSys '25), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), and 
[Triton-Distributed](https://github.com/ByteDance-Seed/Triton-distributed). 
- We measure a single layer's execution only. 
- For every model we evaluated, 
we use model shapes and data types as defined in its corresponding `config.json` on HuggingFace. 
- We **do not** execute any shared experts.
> 👉 On frontier MoE models, FlashMoE gives up to 5x speedup and 69% increase in tensor core utilization compared to SOTA baselines.

## Gated MLP

<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_A100_single_node-2.png" width="4101" alt="">
<p><em>Figure 4: Up to 5.1x faster MoE layer runtime on Qwen-30B with single-node EP</em></p>
</div>

---

## Conventional MLP
<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_A100_vs_COMET.png" width="2946" alt="">
<p><em>Figure 5: Up to 2.6x faster runtime DeepSeek-V2-Lite</em></p>
</div>

---

## Multi-node (libfabric on Slingshot 11)
<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_A100_multi_node.png" width="5592" alt="">
<p><em>Figure 6: Up to 3x speedup on Llama4-Scout for multi-node EP!</em></p>
</div>

--- 

## H100s
<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_H100_single_node.png" width="2940" alt="">
<p><em>Figure 7: Up to 2.5x speedup on H100s.</em></p>
</div>

---

## Run Benchmark (C++)
```shell
cd csrc
mkdir cmake-build-release && cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release -Wno-dev -G Ninja -S.. -B.
cmake --build . --target testFlashMoE --parallel
export NVSHMEM_BOOTSTRAP=MPI
mpirun -n <world> ./testFlashMoE <num tokens per rank> <token dim> <ffn dim> <num experts total> <top k>
```


## IDEs
The codebase integrates well with CLion: open the project at `csrc`.

## Contributions
We welcome them! Submit a PR!

## Acknowledgements
Super grateful to the amazing folks behind
- cuBLASDx 
- CUTLASS
- NVSHMEM
- CCCL

This work would not have been possible without the critical building blocks they provide.

# 📖 Citation
If you can, please cite as below:
```bibtex
@misc{aimuyo2025flashmoe,
      title={FlashMoE: Fast Distributed MoE in a Single Kernel}, 
      author={Osayamen Jonathan Aimuyo and Byungsoo Oh and Rachee Singh},
      year={2025},
      eprint={2506.04667},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2506.04667}, 
}
```
