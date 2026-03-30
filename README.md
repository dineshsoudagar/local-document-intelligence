# 📚 Local Document Intelligence

A fully local document intelligence system for building a private document workspace with persistent indexing, hybrid retrieval, reranking, and grounded answer generation over local models.

Documents stay on disk, retrieval runs against a local Qdrant index, and answer generation runs through locally loaded models. The system is designed for packaged end-user use as well as source-based local development.

---

## ✨ Features

- 📄 Upload and store PDF documents locally
- 🗂️ Build a persistent local knowledge base
- 🔎 Search across the full document corpus
- 📑 Restrict search to a single selected document
- 🧠 Generate grounded answers from local models
- ⚡ Choose between smaller and larger Qwen model options based on hardware
- 💾 Keep documents, indexes, and models local on disk

<!--
## ⬇️ Latest Release

- [Download the latest Windows EXE](#)
- [View the latest release notes](#)

### ✨ Release Highlights

- Added support for Qwen 0.6B and Qwen 1.7B
- Broadened support across lower-memory, mid-range, and higher-end PCs
- Added packaged Windows EXE distribution for easier installation
-->

---

## 🧭 What the Product Does

Local Document Intelligence lets users build a private local document workspace and query it with grounded answers backed by retrieved evidence.

It is designed to:

- ingest and store documents locally
- persist indexing across sessions
- retrieve evidence from one document or the full corpus
- rerank evidence before answer generation
- generate answers with local models instead of cloud services

---

## 🎬 Demo

<p align="center">
  <img src="data/demo.gif" alt="App UI Demo" width="90%" style="margin: 1%;" />
</p>

---

## 🚀 Installation

For full setup instructions, see [installation.md](installation.md).

For packaged downloads, use the latest release page once published.

<!--
### ⬇️ Packaged Windows Download

[Download the latest Windows EXE](#)
-->

### ▶️ Run from source

```powershell
powershell -ExecutionPolicy Bypass -File .\launch-app.ps1
```

---

## 🧠 Supported Models

### Generator Models

- `Qwen 0.6B`  
  Best for CPU-only fallback and very low-memory systems. CPU-only use is supported, but not recommended for normal use because it will be slow.

- `Qwen 1.7B`  
  Best for broader compatibility on lower-memory and mid-range machines.

- `Qwen/Qwen3-4B`  
  Best for balanced local use on machines with around 8 GB VRAM.

- `Qwen/Qwen3-4B-Instruct-2507`  
  Best for stronger local setups with around 8 GB VRAM or better for a more comfortable experience.

### 🔎 Retrieval and Pipeline Models

- Dense embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Reranking: `Qwen/Qwen3-Reranker-0.6B`
- Sparse retrieval: `Qdrant/bm25`

### 💻 Hardware Support

The application is designed to support a broad range of PCs by allowing smaller and larger model choices.

- Lower-end systems can use `Qwen 0.6B`
- Mid-range systems are better suited for `Qwen 1.7B`
- More capable systems can use the `4B` models for better answer quality

CPU-only use is supported with the smaller model path, but it is mainly a compatibility option and will be noticeably slower.

### ⚙️ Low-Memory Runtime Options

Lower-memory setups can use configurable bitsandbytes loading options, including reduced-memory configurations such as 4-bit and 8-bit loading where enabled in the application setup.

These options are intended to improve compatibility on constrained systems, with the usual tradeoff of reduced speed or output quality compared with stronger GPU setups.

All models are stored locally under `models/`.

---

## 🔀 Query Experience

The system supports three user-facing interaction patterns:

### Auto

Auto mode decides whether a request should be handled as direct assistant chat or as document-grounded retrieval.

### Chat

Chat mode skips retrieval and responds directly as a local assistant.

### Grounded Document Answering

Grounded mode retrieves evidence from the local index and produces a cited answer from the retrieved context.

In the UI, this can be used across the full corpus or restricted to selected documents.

---

## ⚙️ How It Works

At a high level, the system does the following:

1. Parse uploaded PDF documents into chunks
2. Store document files and metadata locally
3. Index chunks into a local Qdrant collection
4. Retrieve candidates with dense and sparse search
5. Fuse retrieval results
6. Rerank fused candidates
7. Build grounded context from the strongest evidence
8. Generate a supported answer or decline unsupported claims instead of guessing
9. Return citations, source references, and timing data

---

## 🏗️ Tech Stack

- Backend: FastAPI
- Frontend: React
- Desktop shell: pywebview
- Vector store: Qdrant
- Parsing: Docling
- Local models: Qwen embedding, reranker, and generator models
- Packaged desktop release: PyInstaller one-file build
