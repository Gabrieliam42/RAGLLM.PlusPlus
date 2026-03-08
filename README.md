# RAGLLM.PlusPlus

Local RAG pipeline with dual embedding models and LLM inference. Scripts come in two backend variants: **vLLM** (safetensors, full precision) OR **llama-cpp-python + ggml-python** (GGUF format).

Both vLLM and llama-cpp-python are both inference engines; vLLM is optimized for multi-user high-throughput production, while llama-cpp-python prioritizes portability for single-user workloads.

---

## Scripts

| Script | Backend | Embedder 1 | Embedder 2 | LLM | Focus |
|---|---|---|---|---|---|
| `RAGLLM_Code_Reasoning-Nemotron-1.1-7B_LLAMA-CPP.py` | llama-cpp-python | `nomic-ai/CodeRankEmbed` | `BAAI/bge-code-v1` | `OpenCodeReasoning-Nemotron-1.1-7B-F16.gguf` | Code / technical |
| `RAGLLM_Code_Reasoning-Nemotron-1.1-7B_VLLM.py` | vLLM | `nomic-ai/CodeRankEmbed` | `BAAI/bge-code-v1` | `nvidia/OpenCodeReasoning-Nemotron-1.1-7B` | Code / technical |
| `RAGLLM_English_GLM-4.7-Flash-Q4_1_LLAMA-CPP.py` | llama-cpp-python | `BAAI/bge-m3` | `nomic-ai/nomic-embed-text-v2-moe` | `zai-org_GLM-4.7-Flash-Q4_1.gguf` | English / general |
| `RAGLLM_English_Llama-3.1-Nemotron-Nano-8B-v1_LLAMA-CPP.py` | llama-cpp-python | `BAAI/bge-m3` | `nomic-ai/nomic-embed-text-v2-moe` | `nvidia_Llama-3.1-Nemotron-Nano-8B-v1-bf16.gguf` | English / general |
| `RAGLLM_English_Llama-3.1-Nemotron-Nano-8B-v1_VLLM.py` | vLLM | `BAAI/bge-m3` | `nomic-ai/nomic-embed-text-v2-moe` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | English / general |

All scripts use dual-embed retrieval: embeddings from both models are concatenated and L2-normalized (`concat_l2norm`), producing higher-recall retrieval than a single embedder.

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

**For llama-cpp-python scripts** (`_LLAMA-CPP.py`):
```bash
pip install https://github.com/abetlen/ggml-python/releases/download/v0.0.37-cu124/ggml_python-0.0.37-cp312-cp312-linux_x86_64.whl
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.16-cu124/llama_cpp_python-0.3.16-cp312-cp312-linux_x86_64.whl
```
> Pre-built CUDA 12.4 wheels for Python 3.12 Linux. GGUF models are downloaded automatically from Hugging Face on first run.

---

## Usage

Create a `data/` directory in the same directory as the script (the working directory) and place your documents inside it, then run either script:

```bash
python RAGLLM_Code_Reasoning-Nemotron-1.1-7B_LLAMA-CPP.py
# or
python RAGLLM_Code_Reasoning-Nemotron-1.1-7B_VLLM.py
# or
python RAGLLM_English_GLM-4.7-Flash-Q4_1_LLAMA-CPP.py
# or
python RAGLLM_English_Llama-3.1-Nemotron-Nano-8B-v1_LLAMA-CPP.py
# or
python RAGLLM_English_Llama-3.1-Nemotron-Nano-8B-v1_VLLM.py
```

A desktop GUI (tkinter) will open. On first run, models are downloaded from Hugging Face into `models/` under the working directory. Subsequent runs reuse local files.

**GUI workflow:**

