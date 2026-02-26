<div align="center">

# AI Mastery: From Ground Up to Cutting Edge
### A Complete CS Student's Handbook: Math, Code, Systems, Optimization, Edge, and Brain-Inspired AI

**Version 1.1 - February 2026**  
**Author:** Grok (with Codex as co-author)

</div>

---

## 0. What This Document Is

This document is **learning material**, not a setup guide.

It is written to help you learn AI from first principles to modern production systems:

- core math and optimization
- model architectures and training dynamics
- hardware and acceleration (GPU/TPU/NPU, CUDA, Triton, systolic arrays, CIM)
- efficient inference (quantization, sparsity, KV-cache, PagedAttention)
- distributed systems and networking (K8s internals, gRPC, RDMA, NVLink)
- decentralized and integrity-focused paradigms (DTN, decentralized training, ZKP)

---

## 1. Table of Contents

1. [AI System Map: The Full Stack](#2-ai-system-map-the-full-stack)
2. [Mathematical Foundations](#3-mathematical-foundations)
3. [Training vs Inference](#4-training-vs-inference)
4. [Hardware Landscape and Industry Reality (2026)](#5-hardware-landscape-and-industry-reality-2026)
5. [Hardware Acceleration Deep Dive](#6-hardware-acceleration-deep-dive)
6. [Efficient Inference Deep Dive](#7-efficient-inference-deep-dive)
7. [Extreme Networking and Distributed AI](#8-extreme-networking-and-distributed-ai)
8. [Platform and Orchestration: Kubernetes Internals](#9-platform-and-orchestration-kubernetes-internals)
9. [Distributed Training/Inference Frameworks](#10-distributed-traininginference-frameworks)
10. [Security, Trust, and Data Integrity](#11-security-trust-and-data-integrity)
11. [Roadmap: From Foundations to Cutting Edge](#12-roadmap-from-foundations-to-cutting-edge)
12. [Glossary + Reference Links](#13-glossary--reference-links)

---

## 2. AI System Map: The Full Stack

AI in production is a layered system:

1. **Math layer:** linear algebra, probability, optimization.
2. **Model layer:** architectures (CNN, Transformer, diffusion, etc.).
3. **Kernel layer:** matrix multiplication, attention kernels, communication kernels.
4. **Runtime layer:** PyTorch/JAX/XLA/TensorRT/vLLM.
5. **Hardware layer:** GPU/TPU/NPU/CPU + interconnects (PCIe, NVLink, InfiniBand).
6. **Cluster layer:** schedulers, distributed runtime, orchestration (Kubernetes).
7. **Application layer:** search, agents, recommendation, robotics, multimodal systems.

A strong engineer connects all layers, not just one.

---

## 3. Mathematical Foundations

### 3.1 Core Topics

- Linear algebra: vectors, matrices, eigenvalues, SVD.
- Calculus: partial derivatives, chain rule, Jacobian, Hessian.
- Probability: distributions, expectation, variance, Bayes rule.
- Optimization: convexity intuition, SGD/Adam, regularization.
- Information theory: entropy, cross-entropy, KL divergence.

### 3.2 Canonical Objectives

- Supervised learning objective:

`min_theta E_(x,y) [ L(f_theta(x), y) ]`

- Gradient descent update:

`theta_(t+1) = theta_t - eta * grad_theta L(theta_t)`

- Cross-entropy for classification:

`L = - sum_i y_i log p_i`

### 3.3 Why This Matters

Hardware speedups are useless if you cannot reason about loss surfaces, gradient noise, and generalization.

---

## 4. Training vs Inference

### Simple

- **Training:** fit parameters using data + gradients.
- **Inference:** run forward pass to generate predictions/tokens.

### Intermediate

- Training cost is dominated by forward + backward + optimizer states.
- Inference cost is dominated by attention and memory bandwidth at decode time.

### Deep Dive

- Training complexity scales with model size, sequence length, and batch size.
- Autoregressive inference scales token-by-token and is bottlenecked by:
  - memory movement (KV cache reads/writes)
  - kernel launch overhead
  - inter-GPU communication for tensor parallelism

---

## 5. Hardware Landscape and Industry Reality (2026)

### 5.1 General Compute Environments

- **Cloud GPU clusters:** flexible, expensive, fastest to start.
- **On-prem GPU clusters:** high upfront cost, strong control, lower long-run cost.
- **TPU pods:** optimized matrix workloads inside Google ecosystem.
- **Edge NPU devices:** low-power inference on phones, PCs, IoT.
- **Hybrid:** train in cloud/on-prem, serve at edge + regional cloud.

### 5.2 Market Reality (High-Level)

As of **February 2026**, large-scale training and inference are still mostly **GPU-centric**, with CUDA ecosystem lock-in being a major factor. TPU usage is strong in Google-native stacks, and NPU usage is rapidly increasing for on-device AI.

### 5.3 Accelerator Comparison

| Accelerator | Strength | Weakness | Typical Use |
|---|---|---|---|
| GPU | Mature software ecosystem, high throughput | Power/cost | Training + serving |
| TPU | Efficient dense tensor ops at scale | Ecosystem narrower outside Google stack | Large training jobs |
| NPU | Excellent perf/Watt for edge inference | Limited general programmability | Mobile/edge inference |
| CPU | Flexible, good control logic | Lower dense tensor throughput | Pre/post processing, orchestration |

---

## 6. Hardware Acceleration Deep Dive

### 6.1 CUDA Kernels

A CUDA kernel is a parallel function launched over many threads/blocks. Performance depends on:

- memory coalescing
- shared-memory tiling
- occupancy
- avoiding divergence

### 6.2 Triton

Triton lets you write custom GPU kernels in Python-like code while still controlling memory tiling and block-level parallelism.

Why it matters:

- faster kernel iteration than raw CUDA
- practical for custom attention/GEMM fusion

### 6.3 Systolic Arrays

Systolic arrays are regular grids of processing elements where data flows rhythmically between neighbors. They are excellent for matrix multiply workloads due to predictable data reuse.

### 6.4 Compute-in-Memory (CIM)

CIM reduces data movement by performing parts of computation near or inside memory arrays. Since data movement often dominates energy, CIM can drastically improve energy efficiency.

### 6.5 Interconnects: PCIe, NVLink, InfiniBand

- **PCIe:** default host/device interconnect.
- **NVLink:** high-bandwidth GPU-to-GPU interconnect; critical for multi-GPU scaling.
- **InfiniBand/RoCE:** low-latency cluster networking for distributed training.

---

## 7. Efficient Inference Deep Dive

### 7.1 Quantization (\"quantization / 양자화\")

Quantization maps high-precision values (FP32/FP16) to lower-bit formats (INT8/INT4/FP8).

#### Key terms

- **PTQ (Post-Training Quantization):** quantize a trained model without full retraining.
- **QAT (Quantization-Aware Training):** simulate quantization during training.
- **GPTQ:** PTQ method using approximate second-order information for LLM weights.
- **AWQ:** Activation-aware weight quantization emphasizing important channels.

#### Core tradeoff

Lower precision -> lower memory and higher speed, but potentially lower accuracy.

### 7.2 Sparsity

Sparsity removes or skips low-importance parameters/activations.

- **Unstructured sparsity:** arbitrary zeros, harder to accelerate efficiently.
- **Structured sparsity:** regular patterns (for example N:M), easier on hardware.

### 7.3 KV-Cache Optimization

For autoregressive transformers, attention uses past Keys/Values.

- Without cache: recompute full history each token.
- With cache: reuse past K/V and only compute new token state.

Bottleneck shifts to cache memory layout and bandwidth.

### 7.4 PagedAttention (vLLM)

PagedAttention manages KV cache in page blocks (virtual-memory-like). Benefits:

- less memory fragmentation
- better continuous batching
- higher throughput under many concurrent requests

---

## 8. Extreme Networking and Distributed AI

### 8.1 DTN (Delay-Tolerant Networking)

DTN is designed for intermittent or high-latency links. For AI, DTN-style ideas matter for:

- disconnected edge environments
- asynchronous model updates
- eventual-consistency data exchange

### 8.2 Decentralized Training

Instead of centralized parameter servers, peers exchange updates directly or hierarchically.

Advantages:

- resilience to central failures
- possible privacy/data locality benefits

Challenges:

- stale gradients
- consensus/aggregation stability
- adversarial update handling

### 8.3 gRPC and RDMA

- **gRPC:** high-level RPC framework using HTTP/2 and Protocol Buffers.
- **RDMA:** direct memory access over network with low CPU overhead and low latency.

In practice: control plane often uses gRPC; data plane for high-performance clusters leverages NCCL + RDMA-capable fabrics.

### 8.4 Communication Topologies

- Ring all-reduce
- Tree all-reduce
- Hierarchical all-reduce

Each topology has bandwidth/latency tradeoffs depending on cluster shape.

---

## 9. Platform and Orchestration: Kubernetes Internals

### 9.1 Core Control Plane

- `kube-apiserver`: entry point for all state changes.
- `etcd`: consistent key-value store for cluster state.
- `kube-scheduler`: assigns pods to nodes.
- `kube-controller-manager`: reconciliation loops.

### 9.2 Worker Plane

- `kubelet`: node agent.
- container runtime (`containerd`, etc.).
- CNI plugin for networking.

### 9.3 Why K8s Matters for AI

- multi-tenant resource scheduling
- autoscaling of serving workloads
- reproducible deployment of training/inference services

For GPU workloads, scheduling reliability depends on device plugins, node labeling, and topology awareness.

---

## 10. Distributed Training/Inference Frameworks

### 10.1 Ray

General distributed compute framework for Python workloads. Useful for data pipelines, hyperparameter sweeps, and serving.

### 10.2 DeepSpeed

Optimization library for large-model training and inference (ZeRO stages, memory partitioning, kernel optimizations).

### 10.3 Additional Core Tools

- **PyTorch Distributed / FSDP**
- **NCCL** (GPU collectives)
- **Megatron-LM** (tensor/pipeline parallel patterns)
- **vLLM / TensorRT-LLM** (high-throughput serving)

Selection should match bottleneck: memory, communication, latency, or throughput.

---

## 11. Security, Trust, and Data Integrity

### 11.1 ZKP (Zero-Knowledge Proofs)

ZKP allows proving a statement is true without revealing underlying private data. In AI pipelines, this can support integrity claims (for example, proof that a step was executed correctly) without exposing sensitive data.

### 11.2 Data Integrity in AI Systems

- cryptographic hashing for dataset/model artifacts
- signed model registries
- immutable lineage and provenance metadata
- reproducible training manifests

### 11.3 Threat Model Basics

- data poisoning
- model extraction
- prompt injection / tool abuse
- supply-chain compromise

---

## 12. Roadmap: From Foundations to Cutting Edge

- **Level 0:** Math + Python + algorithms
- **Level 1:** Classical ML + evaluation
- **Level 2:** Deep learning fundamentals + backprop from scratch
- **Level 3:** Transformers, LLM pretraining/fine-tuning
- **Level 4:** Systems optimization (kernels, memory, parallelism)
- **Level 5:** MLOps + serving architecture
- **Level 6:** Efficient/edge AI (quantization, NPUs, deployment)
- **Level 7:** Trustworthy/safe AI and red-teaming
- **Level 8:** Physical AI (robotics/simulation/control)
- **Level 9:** Brain-inspired AI (SNN, LNN, neuromorphic)

Rule for every level:

1. Learn the theory.
2. Implement a minimal working version.
3. Benchmark and analyze bottlenecks.
4. Write what failed and why.

---

## 13. Glossary + Reference Links

### Hardware and acceleration

- CUDA: [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone)
- Triton: [Triton Language](https://triton-lang.org/main/index.html)
- TPU: [Google Cloud TPU](https://cloud.google.com/tpu)
- NPU: [Neural processing unit (Wikipedia)](https://en.wikipedia.org/wiki/Neural_processing_unit)
- Systolic array: [Systolic array (Wikipedia)](https://en.wikipedia.org/wiki/Systolic_array)
- Compute-in-memory: [Compute-in-memory (Wikipedia)](https://en.wikipedia.org/wiki/Computing-in-memory)
- NVLink: [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)

### Inference optimization

- Quantization: [Model quantization (Wikipedia)](https://en.wikipedia.org/wiki/Quantization_(machine_learning))
- GPTQ: [GPTQ paper](https://arxiv.org/abs/2210.17323)
- AWQ: [AWQ paper](https://arxiv.org/abs/2306.00978)
- KV cache: [KV cache explainer (Hugging Face)](https://huggingface.co/docs/transformers/main/cache_explanation)
- vLLM / PagedAttention: [vLLM docs](https://docs.vllm.ai/)

### Networking and systems

- DTN: [Delay-tolerant networking (Wikipedia)](https://en.wikipedia.org/wiki/Delay-tolerant_networking)
- gRPC: [gRPC](https://grpc.io/)
- RDMA: [RDMA (Wikipedia)](https://en.wikipedia.org/wiki/Remote_direct_memory_access)
- Kubernetes: [Kubernetes docs](https://kubernetes.io/docs/home/)
- Ray: [Ray docs](https://docs.ray.io/)
- DeepSpeed: [DeepSpeed docs](https://www.deepspeed.ai/)

### Security and integrity

- ZKP: [Zero-knowledge proof (Wikipedia)](https://en.wikipedia.org/wiki/Zero-knowledge_proof)

---

## Version History

- **v1.1 (February 2026):** Rewritten as general AI learning material with expanded hardware, optimization, networking, orchestration, and integrity topics.
- **v1.0 (February 2026):** Initial compact handbook structure.
