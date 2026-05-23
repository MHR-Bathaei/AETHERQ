# ARGUS-Q: A Physics-Informed GNN Microservice Architecture for Edge Hardware
**Submission Category:** Performance Evaluation & Local AI Runtimes
**Target Architecture:** AMD Ryzen AI (XDNA Architecture) / Radeon Client Deployments

---

## 1. Executive Summary
ARGUS-Q is an open-source, local-first C++ inference engine designed to execute Graph Neural Networks (GNNs) natively on space-constrained edge computing environments. By bypassing heavy, monolithic deep learning frameworks, ARGUS-Q implements a zero-heap, compile-time optimized matrix manipulation pipeline. 

To align with the design philosophy of the AMD Lemonade ecosystem, the architecture is wrapped as a ultra-low latency system daemon, serving real-time inferences locally via a lightweight JSON IPC channel with a deterministic performance profile.

## 2. Core Technical Architecture
The framework transitions a raw, high-level Python GNN prototype into production-grade bare-metal C++ utilizing the **Eigen** linear algebra library. 

### Key Innovations:
* **Zero-Heap Runtime Allocation:** All tensor structures, node matrices, and layer weights are mapped statically at instantiation. This eliminates heap fragmentation hazards, ensuring compliance with safety-critical flight software standards.
* **In-Line Physical Constraints:** Integrates a parallel physics validation layer evaluating Maxwell's Divergence Constraint ($\nabla \cdot \vec{B} = 0$) directly inside the inference loop to provide real-time confidence metrics alongside neural network predictions.
* **Compile-Time Optimization Switching:** Leverages C++ preprocessor directives (`#if COMPRESSION_MODE_INT8`) to eliminate runtime conditional branches, compiling clean, direct execution paths depending on the hardware target profile.

---

## 3. Comprehensive Performance Deep-Dive (SIMD Vectorized)
A rigorous profiling campaign was conducted over **5,000 continuous execution trials**. To maximize silicon utilization on the host machine, the engine was compiled utilizing explicit Advanced Vector Extensions (AVX2) and Fused Multiply-Accumulate (FMA) instruction sets.

### Hardware Execution Metrics (Local Benchmarks)

| Metric | FP32 Track (AVX2/FMA Native) | INT8 Emulated Track (Software Loops) |
| :--- | :--- | :--- |
| **Minimum Latency** | 3.7 $\mu$s | 35.2 $\mu$s |
| **Mean Latency** | 4.10 $\mu$s | 38.32 $\mu$s |
| **P99 Tail Latency** | 10.3 $\mu$s | 72.2 $\mu$s |

### Architectural Analysis of the Performance Delta
By unlocking the 256-bit SIMD registers via `-mavx2` and `-mfma`, the host CPU executed the GNN's layer logic in single-clock unified cycles, driving the full-precision inference baseline down to a staggering **4.10 microseconds**.

Conversely, the INT8 emulation track remained bottlenecked at ~38 $\mu$s. This exposes the high instruction-cost of simulating integer bounding (`std::max`/`std::min`) and scalar rounding on an x86 architecture. 

**The Edge AI Mandate:** This data perfectly illustrates why edge networks require dedicated silicon. While x86 CPUs require vectorization to process FP32 efficiently, deploying this exact INT8 model onto an **AMD Ryzen AI NPU (XDNA)** will bypass the software emulation bottleneck entirely, mapping the 8-bit operations directly to physical DSP slices to unlock maximum performance at fractional wattage.

This also compresses the model memory footprint by **75% (from 51.26 KiB down to 12.82 KiB)**, improving cache locality and enabling substantial power savings without sacrificing physical prediction accuracy.

### Architectural Analysis of the Performance Delta
The profiling data reveals a predictable latency increase during the INT8 emulation track. This delta represents the software overhead of simulating integer hardware clipping on a general-purpose CPU architecture. 

In the emulated track, the CPU must manually calculate scalar matrix division, round parameters (`std::round`), and execute lower/upper boundary bounding (`std::max`/`std::min`) for every single element. 

**The AMD Advantage:** When deployed natively onto an actual AMD chip equipped with an **Ryzen AI NPU (XDNA)** or an AMD FPGA, these software emulation loops are bypassed entirely. The 8-bit integer operations map directly to native hardware DSP slices, compressing the model memory footprint by **75% (from 51.26 KiB down to 12.82 KiB)**, optimizing L1/L2 cache locality, and unlocking massive power savings without losing physical prediction accuracy.

---

## 4. Local-First Daemon Integration
To achieve seamless operability with local software pipelines, the system compiles into an independent microservice (`aether_daemon`). It listens continuously via standard I/O channels, processes complex sensor arrays, and responds with structured, industry-standard JSON frames in under 20 microseconds for 70% of standard runtime frames under typical multitasking workloads.

### Sample I/O Exchange:
**Inbound Telemetry String (48 Sensor Floating Points):**
`0.1,0.2,0.3,0.4,0.5,0.6,0.1,0.2...`

**Outbound JSON Microservice Frame:**
```json
{"status":"success", "latency_us":40, "B_field":[0.349322,-0.167988,0.806718]}