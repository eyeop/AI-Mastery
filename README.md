<div align="center">

# AI Mastery: From Ground Up to Cutting Edge
### A Complete Computer Science Student's Handbook

**Version 1.0 - February 2026**  
**Author:** Grok (with Codex as co-author)  
**Primary Goal:** Build practical mastery across the full AI stack, from math to systems to deployment.

</div>

---

## Why This Document Exists

This is a **single, living GitHub document** you can read on desktop or phone anytime.  
It is designed to be:

- **Practical:** every section maps to things you can build.
- **Layered:** each topic follows `Simple -> Intermediate -> Deep Dive`.
- **Compounding:** each level unlocks the next one.

---

## Table of Contents

1. [Hardware Setup & First GitHub Repo](#1-hardware-setup--first-github-repo)
2. [Training vs Inference - The Two Phases of AI](#2-training-vs-inference---the-two-phases-of-ai)
3. [Hardware Acceleration: CUDA, Triton, TensorRT](#3-hardware-acceleration-cuda-triton-tensorrt)
4. [Efficient Inference Deep Dive](#4-efficient-inference-deep-dive)
   - [4.1 Quantization](#41-quantization-simple---intermediate---deep-dive)
   - [4.2 Sparsity](#42-sparsity)
   - [4.3 KV-Cache Optimization](#43-kv-cache-optimization)
   - [4.4 PagedAttention & vLLM](#44-pagedattention--vllm)
5. [Full AI Mastery Roadmap (Level 0 -> 9)](#5-full-ai-mastery-roadmap-level-0---9)
6. [Learning Protocol & Next Steps](#6-learning-protocol--next-steps)
7. [Version History](#7-version-history)

---

## 1. Hardware Setup & First GitHub Repo

### Recommended Starter Machines

| Machine | Best For | VRAM / Unified Memory | Recommendation |
|---|---|---|---|
| Desktop + RTX 3060 Ti | Training, custom kernels, TensorRT | 8 GB VRAM | Primary GPU machine |
| M4 Pro MacBook | Daily coding, notebooks, MLX inference | Up to 64 GB unified memory | Portable + efficient secondary machine |

### 10-Minute Setup Checklist

1. Install NVIDIA driver and CUDA 12.4 on desktop.
2. Install CUDA-enabled PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

3. Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

4. Create repo: `https://github.com/<your-username>/ai-mastery-journey`
5. Start with this structure:

```text
ai-mastery-journey/
├── README.md
├── notebooks/          # 01_math.ipynb, 02_micrograd.ipynb
├── src/
├── configs/
├── data/               # add .gitignore
└── models/             # add .gitignore
```

---

## 2. Training vs Inference - The Two Phases of AI

<details open>
<summary><strong>Simple</strong></summary>

- **Training** = teaching the model.
- **Inference** = using the model.

</details>

<details>
<summary><strong>Intermediate</strong></summary>

- Training uses forward pass + backward pass + optimizer update.
- Training stores activations, gradients, and optimizer states.
- Inference only needs forward pass, so it is much lighter.

</details>

<details>
<summary><strong>Deep Dive</strong></summary>

- Typical supervised training loop:
  - `loss = CrossEntropy(logits, labels)`
  - `loss.backward()`
  - `optimizer.step()` and `optimizer.zero_grad()`
- Gradient descent intuition:
  - `theta = theta - eta * dL/dtheta`
- LLM generation is autoregressive:
  - predict token `t+1` from prefix `[1..t]`, append, repeat.

</details>

---

## 3. Hardware Acceleration: CUDA, Triton, TensorRT

### CUDA (NVIDIA)

- Core platform for NVIDIA GPU programming.
- You write kernels for massively parallel execution.

### Triton (OpenAI)

- Python-first language for custom GPU kernels.
- Faster development cycle than raw CUDA for many operations.

```python
@triton.jit
def matmul_kernel(...):
    ...
```

### TensorRT (NVIDIA)

- High-performance inference runtime and optimizer.
- Performs graph optimizations, kernel autotuning, mixed precision/quantization.
- Typical gain profile: lower latency, higher throughput, reduced memory.

---

## 4. Efficient Inference Deep Dive

### 4.1 Quantization (Simple -> Intermediate -> Deep Dive)

<details open>
<summary><strong>Simple</strong></summary>

Represent parameters with fewer bits (`FP32 -> INT8/INT4`) to reduce memory and improve speed.

</details>

<details>
<summary><strong>Intermediate</strong></summary>

- Post-training quantization (PTQ): GPTQ, AWQ.
- Quantization-aware training (QAT): train with quantization behavior included.

</details>

<details>
<summary><strong>Deep Dive</strong></summary>

- Weight-only vs weight+activation quantization.
- Per-tensor vs per-channel scaling.
- Common LLM tooling: `LLM.int8()`, bitsandbytes NF4, SmoothQuant.

</details>

### 4.2 Sparsity

- Remove low-value weights to reduce effective computation.
- Structured sparsity (N:M) better matches modern accelerator kernels.

### 4.3 KV-Cache Optimization

- Without KV cache: each new token recomputes attention over full prior context.
- With KV cache: persist past Keys/Values and only compute new step.
- Result: major decode speedup, with memory tradeoff at long context lengths.

### 4.4 PagedAttention & vLLM

- Treat KV cache as paged memory blocks.
- Reduces fragmentation and improves allocator efficiency.
- Enables continuous batching and prefix cache reuse.

---

## 5. Full AI Mastery Roadmap (Level 0 -> 9)

- **Level 0:** Mathematical & Programming Foundations
- **Level 1:** Classical Machine Learning
- **Level 2:** Deep Learning Basics (micrograd, backprop from scratch)
- **Level 3:** Transformers & Modern LLMs (nanoGPT, Llama fine-tuning)
- **Level 4:** Systems & Optimization (Triton kernels, distributed training)
- **Level 5:** Production & MLOps (RAG, agents, vLLM serving)
- **Level 6:** Efficient & Edge AI (quantization, MLX, ExecuTorch)
- **Level 7:** Trustworthy & Safe AI (alignment, red-teaming)
- **Level 8:** Physical / Embodied AI (robotics, simulation)
- **Level 9:** Brain-Inspired AI (SNN, LNN, neuromorphic hardware)

### Roadmap Rule

For each level, complete:

- 1 theory summary (short notes)
- 1 implementation (notebook or script)
- 1 project artifact (demo, benchmark, or report)

---

## 6. Learning Protocol & Next Steps

### Weekly Cadence (Suggested)

- **Mon-Tue:** theory and derivations.
- **Wed-Thu:** implementation and debugging.
- **Fri:** benchmark + write-up.
- **Weekend:** revision and portfolio cleanup.

### Immediate Next Actions

1. Push this README as your root document.
2. Create `notebooks/01_math.ipynb` and `notebooks/02_micrograd.ipynb`.
3. Start Level 0 with linear algebra + calculus refresh + Python fluency.

---

## 7. Version History

- **v1.0 (February 2026):** Initial handbook structure with hardware, systems, inference optimization, and full 0-9 roadmap.

---

## Publish / Update This Single-Document GitHub Book

```bash
git add README.md
git commit -m "Update AI Mastery handbook"
git push
```

If creating from scratch:

```bash
git init
git add README.md
git commit -m "Add AI Mastery handbook v1.0"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-mastery-journey.git
git push -u origin main
```
