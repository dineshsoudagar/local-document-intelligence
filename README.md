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

<!--
## ⬇️ Latest Release

- [Download the latest Windows EXE](#)
- [View the latest release notes](#)

### ✨ Release Highlights

- Added support for Qwen 0.6B and Qwen 1.7B
- Broadened support across lower-memory, mid-range, and higher-end PCs
- Added packaged Windows EXE distribution for easier installation
-->


## 🎬 Demo

<p align="center">
  <img src="src/assets/demo.gif" alt="App UI Demo" width="90%" style="margin: 1%;" />
</p>

---

## 🚀 Installation

Packaged downloads should be published on the release page.

<!--
### ⬇️ Packaged Windows Download

[Download the latest Windows EXE](#)
-->

### ▶️ Run from source

```powershell
powershell -ExecutionPolicy Bypass -File .\launch-app.ps1
```

The repository keeps `frontend/dist/` checked in because normal desktop startup
and PyInstaller packaging both expect a prebuilt frontend. Rebuild it only when
you change the UI.

---

## 🧠 Supported Models

### Generator Models

- `Qwen/Qwen3-4B`  
  **More than 8 GB VRAM.** Best for users who want the strongest reasoning-focused model for deeper analysis, more complex instructions, and higher-quality thinking.

- `Qwen/Qwen3-4B-Instruct-2507`  
  **More than 8 GB VRAM.** Best for users who want a polished, instruction-following experience for everyday document Q and A, summarization, and general use.

- `Qwen 1.7B`  
  **Less than 8 GB VRAM.** Best for users who want a balanced experience with good responsiveness and solid general document interaction, with slightly lower capability on harder tasks than the 4B models.

- `Qwen 0.6B`  
  **Less than 4 GB VRAM.** Best for users who want the lightest setup for constrained hardware and basic document interaction, with lower performance on more complex tasks.

### VRAM Guidance

Higher-VRAM systems can use the 4B models for the strongest overall experience, while lower-VRAM systems can use the smaller models for a lighter and more accessible local setup.

### 🔎 Retrieval and Pipeline Models

- Dense embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Reranking: `Qwen/Qwen3-Reranker-0.6B`
- Sparse retrieval: `Qdrant/bm25`

### ⚙️ Low-Memory Runtime Options

Lower-memory setups can use configurable bitsandbytes loading options, including reduced-memory configurations such as 4-bit and 8-bit loading where enabled in the application setup.

These options are intended to improve compatibility on constrained systems, with the usual tradeoff of reduced speed or output quality compared with stronger GPU setups.

---

## 🔀 Query Experience

The system supports four user-facing query modes:

### Auto

Auto mode decides whether a request should be handled as direct assistant chat or as document-grounded retrieval.

### Chat

Chat mode skips retrieval and responds directly as a local assistant.

### Corpus

Corpus mode searches across the full indexed document workspace and generates an answer from the retrieved evidence.

### Single Document

Single Document mode restricts retrieval to one selected document and generates an answer only from that document’s evidence.

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
