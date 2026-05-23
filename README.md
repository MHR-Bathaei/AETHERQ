# ARGUS-Q: Hardware-Accelerated GNN Inference Engine

ARGUS-Q is an ultra-low latency, zero-heap C++ edge inference engine designed to execute Graph Neural Networks (GNNs) natively on localized architectures. By bypassing heavy runtime deep learning frameworks, it leverages the Eigen linear algebra library to achieve deterministic, microsecond-level execution.

## Project Architecture
* **`src/`**: Core optimized C++ source code including compile-time weight mapping layouts.
* **`client_visualizer.py`**: Python client utilizing Inter-Process Communication (IPC) to stream 3D telemetry into the live background daemon.
* **`data/` & `results/`**: Real-time evaluation matrices and performance logs.
* **`AMD_SUBMISSION.md`**: Architectural hardware deep-dive evaluation report.

## Setup & Local Deployment

### Compilation Requirements
* GCC/G++ compiler toolchain with C++17 support.
* Eigen library headers extracted into your compiler's search path.

### 1. Run the SIMD Vectorized Hardware Profiler
To run the 5,000-trial benchmark tracking Advanced Vector Extensions (AVX2) and Fused Multiply-Accumulate (FMA) execution speeds:
```bash
g++ -O3 -mavx2 -mfma -march=native -I ./src src/benchmarker.cpp -o benchmarker_simd
./benchmarker_simd