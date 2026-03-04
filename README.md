# RAGLLM.PlusPlus

Local RAG pipeline with dual embedding models and vLLM inference. Two script variants — one code-focused, one English-focused.

---

## Scripts

| Script | Embedder 1 | Embedder 2 | LLM | Focus |
|---|---|---|---|---|
| `RAGLLM_Code_Plus_Plus.py` | `nomic-ai/CodeRankEmbed` | `BAAI/bge-code-v1` | `nvidia/OpenCodeReasoning-Nemotron-1.1-7B` | Code / technical |
| `RAGLLM_English_Plus_Plus.py` | `BAAI/bge-m3` | `nomic-ai/nomic-embed-text-v2-moe` | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` | English / general |

Both scripts use dual-embed retrieval: embeddings from both models are concatenated and L2-normalized (`concat_l2norm`), producing higher-recall retrieval than a single embedder.

---

## Requirements

**Hardware**
- NVIDIA GPU with CUDA 12.x, Ampere or newer (developed and tested on RTX 3090 24 GB)
- 16+ GB RAM (128 GB recommended for large document sets)

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

### 4. Install vLLM

```bash
pip install vllm==0.10.1 --extra-index-url https://download.pytorch.org/whl/cu128
```

> vLLM is not in `requirements.txt` due to its build complexity. Requires `transformers>=4.46,<5.0` (already pinned in requirements).

---

## Usage

Place your documents in a `data/` directory, then run either script:

```bash
python RAGLLM_Code_Plus_Plus.py
# or
python RAGLLM_English_Plus_Plus.py
```

A desktop GUI (tkinter) will open. On first run, models are downloaded from Hugging Face into `models/` under the working directory. Subsequent runs reuse local files.

**GUI workflow:**
1. Engine starts in background: loads embedders → builds/loads FAISS index → loads LLM
2. Type a question in the input field
3. Click **Ask** — top-K chunks are retrieved and passed to the LLM as context
4. The answer and retrieved source chunks are displayed

---

## RAGLLM_Code_Plus_Plus.py

Code and technical document retrieval using two code-specialized embedding models and a reasoning LLM.

### Models

| Role | Model |
|---|---|
| Embedder 1 | `nomic-ai/CodeRankEmbed` |
| Embedder 2 | `BAAI/bge-code-v1` |
| LLM | `nvidia/OpenCodeReasoning-Nemotron-1.1-7B` |

### Pipeline

```
data/ files
    └─► chunk (1200 chars, 180 overlap)
        └─► embed: CodeRankEmbed + bge-code-v1 → concat + L2-normalize
            └─► FAISS IndexFlatIP (GPU/cuVS if available)
                └─► top-K retrieval
                    └─► prompt → vLLM (OpenCodeReasoning-Nemotron-1.1-7B)
                        └─► answer
```

### Defaults

| Setting | Value |
|---|---|
| Top K | 5 |
| Max New Tokens | 4096 |
| Temperature | 0.1 |
| Embed batch size | 4 |
| Index cache | `data/.rag_cache_code_plus_plus/` |
| Session log | `RAGLLM_Code_Plus_Plus_Session.txt` |

> `PYTORCH_CUDA_ALLOC_CONF` is set to `""` to prevent CUDA graph handle conflicts between PyTorch embedding inference and vLLM.

---

## RAGLLM_English_Plus_Plus.py

English and general-purpose document retrieval using a multilingual embedding model paired with a MoE text embedder, and a general-purpose Nemotron LLM.

### Models

| Role | Model |
|---|---|
| Embedder 1 | `BAAI/bge-m3` |
| Embedder 2 | `nomic-ai/nomic-embed-text-v2-moe` |
| LLM | `nvidia/Llama-3.1-Nemotron-Nano-8B-v1` |

### Pipeline

```
data/ files
    └─► chunk (1200 chars, 180 overlap)
        └─► embed: bge-m3 + nomic-embed-text-v2-moe → concat + L2-normalize
            └─► FAISS IndexFlatIP (GPU/cuVS if available)
                └─► top-K retrieval
                    └─► prompt → vLLM (Llama-3.1-Nemotron-Nano-8B-v1)
                        └─► answer
```

### Defaults

| Setting | Value |
|---|---|
| Top K | 7 |
| Max New Tokens | 4096 |
| Temperature | 0.7 |
| Embed batch size | 4 |
| Index cache | `data/.rag_cache_english_plus_plus/` |
| Session log | `RAGLLM_English_Plus_Plus_Session.txt` |

> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce VRAM fragmentation during embedding inference.

---

## Shared Pipeline Details

### Embedding
- Both models run on CUDA (CPU embedding is rejected)
- Vectors are L2-normalized on GPU via `torch.nn.functional.normalize` before CPU transfer
- Concatenated dual-embed vectors are re-normalized after concatenation
- `EMBED_MAX_LENGTH=1024` tokens

### FAISS Index
- Index type: `IndexFlatIP` (exact inner-product / cosine similarity on L2-normalized vectors)
- GPU promotion: `StandardGpuResources` + float16 cloner options
- cuVS acceleration auto-detected at runtime (`faiss-gpu-cu12-cuvs`)
- Index cached per script — reused across runs if documents are unchanged

### LLM (vLLM)
- Backend: vLLM 0.10.1 with PagedAttention
- dtype: `bfloat16` on Ampere, `float16` fallback
- `gpu_memory_utilization=0.88` (~20.2 GiB on RTX 3090)
- On OOM: retries with `gpu_memory_utilization=0.73`, `cpu_offload_gb=16`, `enforce_eager=True`

### Document ingestion
Supported file types: `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.go`, `.rs`, `.md`, `.txt`, `.json`, `.yaml`, `.toml`, `.csv`, `.log`, `.rst`, `.html`, `.docx`, `.doc`, and more.

`.doc` files require `libreoffice` or `antiword`/`catdoc` installed on the system.

---

## Configuration

Environment variables (all optional, apply to both scripts):

| Variable | Default | Description |
|---|---|---|
| `RAG_GPU_MEM_UTIL` | `0.88` | vLLM GPU memory utilization fraction |
| `RAG_MAX_MODEL_LEN` | `4096` | Maximum token context length |
| `RAG_CPU_OFFLOAD_GB` | `0` | GB to offload to CPU RAM (0 = disabled) |
| `RAG_EMBED_BATCH` | `4` | Embedding batch size |

---

## FAISS Package

Uses [`faiss-gpu-cu12-cuvs`](https://pypi.org/project/faiss-gpu-cu12-cuvs/) — an unofficial pip wheel for FAISS 1.14.0 with GPU + NVIDIA cuVS support, not available in the official `faiss-gpu-cu12` package.

- PyPI: https://pypi.org/project/faiss-gpu-cu12-cuvs/
- GitHub: https://github.com/Gabrieliam42/faiss-gpu-cu12-cuvs

---

## File Structure

```
.
├── RAGLLM_Code_Plus_Plus.py          # Code-focused script
├── RAGLLM_English_Plus_Plus.py       # English-focused script
├── requirements.txt
├── data/                             # Place documents here
│   ├── .rag_cache_code_plus_plus/    # Auto-generated (Code++ script)
│   └── .rag_cache_english_plus_plus/ # Auto-generated (English++ script)
├── models/                           # Auto-populated on first run
│   ├── embeddings/
│   └── llm/
├── RAGLLM_Code_Plus_Plus_Session.txt    # Auto-generated session log
└── RAGLLM_English_Plus_Plus_Session.txt # Auto-generated session log
```

---

## Notes

- Linux only — hard exits on non-Linux platforms
- WSL2: Ctrl+C in the GUI copies to the Windows clipboard via `clip.exe`
- Model files are downloaded to `models/` in the working directory, not the global HF cache
- Each script has its own index cache and session log — they can run independently without conflict
- Session log appended after each query with timestamp, question, and answer
