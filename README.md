# AETHER-Q: Hardware-Accelerated GNN Inference Engine
<p align="center">
  <img src="media/dashboard_demo.gif" alt="ARGUS-Q Dashboard Live Demo" width="100%" max-width="900px">
</p>

AETHER-Q is an ultra-low latency, zero-heap C++ edge inference engine designed to execute Graph Neural Networks (GNNs) natively on localized architectures. By bypassing heavy runtime deep learning frameworks, it leverages the Eigen linear algebra library to achieve deterministic, microsecond-level execution.

## Project Architecture
* **`src/`**: Core optimized C++ source code including compile-time weight mapping layouts.
* **`client_visualizer.py`**: Python client utilizing Inter-Process Communication (IPC) to stream 3D telemetry into the live background daemon.
* **`data/` & `results/`**: Real-time evaluation matrices and performance logs.
* **`AMD_SUBMISSION.md`**: Architectural hardware deep-dive evaluation report.

## Setup & Local Deployment

### Compilation Requirements
* GCC/G++ compiler toolchain with C++17 support.
* Eigen library headers extracted into your compiler's search path.

### Run the SIMD Vectorized Hardware Profiler
To run the 5,000-trial benchmark tracking Advanced Vector Extensions (AVX2) and Fused Multiply-Accumulate (FMA) execution speeds:
```bash
g++ -O3 -mavx2 -mfma -march=native -I ./src src/benchmarker.cpp -o benchmarker_simd
./benchmarker_simd
```

---

## Project Evolution: AETHERQ-GPU

The AVX2/FMA implementation in this repository served as the baseline for a follow-up CUDA study focused on throughput-oriented graph inference.

The resulting project, **AETHERQ-GPU**, reimplements the projection workload using a massively parallel CUDA execution model, mapping independent inference iterations across the GPU execution grid.

### Comparison

| Implementation | Execution Model                     |
| -------------- | ----------------------------------- |
| AETHER-Q       | AVX2/FMA vectorized CPU inference   |
| AETHERQ-GPU    | CUDA-parallel batched GPU inference |

The GPU implementation achieved a measured **97.1× throughput improvement** on a benchmark consisting of 5,000 independent projection iterations executed concurrently on an NVIDIA RTX 4060 Laptop GPU.

**Repository:** https://github.com/MHR-Bathaei/AETHERQ-GPU

```text
AETHER-Q (CPU SIMD)
        ↓
AETHERQ-GPU (CUDA)
```

Together, the two repositories document the progression from CPU SIMD optimization to GPU-parallel execution strategies for graph neural network workloads.
