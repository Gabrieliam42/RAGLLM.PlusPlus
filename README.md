# RAGLLM.PlusPlus

Local RAG pipeline with dual embedding models and LLM inference. Scripts come in two backend variants: **vLLM** (safetensors, GPU half/bfloat16 inference) OR **llama-cpp-python + ggml-python** (GGUF format).

Both vLLM and llama-cpp-python are inference engines: vLLM is optimized for multi-user high-throughput production, while llama-cpp-python prioritizes portability for single-user workloads.

---

## Scripts

| Script | Backend | Embedder 1 | Embedder 2 | LLM | Focus |
|---|---|---|---|---|---|
| `RAGLLM_Code_Reasoning-Nemotron-1.1-7B_LLAMA-CPP.py` | llama-cpp-python | `nomic-ai/CodeRankEmbed` | `BAAI/bge-code-v1` | `OpenCodeReasoning-Nemotron-1.1-7B-F16.gguf` | Code / technical |
| `RAGLLM_Code_Reasoning-Nemotron-1.1-7B_VLLM.py` | vLLM | `nomic-ai/CodeRankEmbed` | `BAAI/bge-code-v1` | `nvidia/OpenCodeReasoning-Nemotron-1.1-7B` | Code / technical |
| `RAGLLM_English_GLM-4.7-Flash-Q4_1_LLAMA-CPP.py` | llama-cpp-python | `BAAI/bge-m3` | `nomic-ai/nomic-embed-text-v2-moe` | `zai-org_GLM-4.7-Flash-Q4_1.gguf` | English / general |
| `RAGLLM_English_Llama-3.1-Nemotron-Nano-8B-v1_LLAMA-CPP.py` | llama-cpp-python | `BAAI/bge-m3` | `nomic-ai/nomic-embed-text-v2-moe` | `nvidia_Llama-3.1-Nemotron-Nano-8B-v1-bf16.gguf` | English / general |
| `RAGLLM_English_Llama-3.1-Nemotron-Nano-8B-v1_VLLM.py` | vLLM | `BAAI/bge-m3` | `nomic-ai/nomic-embed-text-v2-moe` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | English / general |

All scripts use dual-embed retrieval: embeddings from both models are loaded through `SentenceTransformer`, concatenated, and L2-normalized (`concat_l2norm`), producing higher-recall retrieval than a single embedder.

FlashAttention-2 is enabled for embedding models where the underlying architecture and Transformers backend support it, with automatic fallback to standard attention where FA2 is unsupported.

---

## Requirements
**Hardware**
- NVIDIA GPU with CUDA 12.x, Ampere or newer
- 16+ GB RAM

**OS**
- Linux only (Ubuntu or WSL2)

**Python**
- 3.12+

---

## Installation

### 1. Install RAPIDS dependencies (required for faiss-gpu-cu12-cuvs)

```bash
pip install libcuvs-cu12==25.10.0 librmm-cu12==25.10.0 libraft-cu12==25.10.0 \
    rapids-logger "nvidia-nvjitlink-cu12>=12.9" --extra-index-url https://pypi.nvidia.com
```

### 2. Install system dependency

```bash
sudo apt-get install -y libopenblas-dev
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install LLM backend

**For vLLM scripts** (`_VLLM.py`):
```bash
pip install vllm==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu128
```
> vLLM is not in `requirements.txt` due to its build complexity. Requires `transformers>=4.46,<5.0` (already pinned in requirements).
