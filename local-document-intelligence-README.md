# local-document-intelligence

A fully local document intelligence system for building a private document workspace with persistent indexing, hybrid retrieval, reranking, and grounded answer generation over local models.

This repository is aimed at a real local workflow. Documents stay on disk, retrieval runs against a local Qdrant index, and answer generation runs through locally loaded models.

## Query Modes

The application currently exposes four practical query modes.

### Auto

Auto mode decides whether a query should be handled as normal assistant chat or as document-grounded retrieval.

Use this when you do not want to manually choose between free chat and document search. The system first classifies the request. If the query looks like a normal conversational or meta request, it answers directly. If the query needs document evidence, it runs retrieval instead.

### Chat

Chat mode skips retrieval and answers directly as a local assistant.

Use this for general interaction, system questions, or messages that do not require document evidence. This mode is intentionally separate from grounded search so conversational requests do not pay retrieval cost unnecessarily.

### Corpus

Corpus mode searches across the full indexed knowledge base.

Use this when the answer may be spread across multiple uploaded documents or when you want the system to search the entire local corpus. This is the default grounded search behavior when no specific document is selected.

### Single Document

Single document mode restricts retrieval to one selected document.

Use this when you want the answer to come only from one document instead of the full corpus. This is useful for focused review, contract inspection, paper reading, and situations where cross-document mixing would be undesirable.

## Runtime Design

The system is built around a controlled local retrieval pipeline instead of a single vector lookup.

At a high level, the runtime flow is:

1. parse the uploaded document into chunks
2. store chunk text and metadata locally
3. index chunks into a local Qdrant collection
4. retrieve candidates with dense and sparse search
5. fuse retrieval results
6. rerank the fused candidates
7. build grounded context from the strongest evidence
8. judge whether the evidence is sufficient
9. generate either:
   - a grounded supported answer
   - a partial answer with explicit incompleteness
   - or an unsupported response instead of bluffing

The retrieval layer is not dense-only. It uses dense search, sparse BM25-style search, reciprocal-rank fusion, reranking, and a bounded second pass when early evidence is weak.

## Technical Characteristics

- Local Qdrant storage for persistent indexing
- Hybrid retrieval using dense embeddings and sparse BM25-style search
- Reranker-based final evidence selection
- Evidence judgment before final answer generation
- Selected-document filtering when a single document is chosen
- Local FastAPI backend
- React frontend
- Release runtime serves the built frontend directly from `frontend/dist`

## Models

Current default model stack:

- Dense embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Reranking: `Qwen/Qwen3-Reranker-0.6B`
- Answer generation: `Qwen/Qwen3-4B-Instruct-2507`
- Sparse retrieval: `Qdrant/bm25`

Models are stored locally under `models/`.

## Runtime Requirements

### Prerequisites

- Python 3.11 or newer
- PowerShell

### Current hardware target

The current setup is aimed at machines with at least **8 GB VRAM** for a comfortable local run with the present model stack.

This is not yet tuned for low-end PCs. Lower-memory support is planned as part of the next round of optimization work.

## Release Runtime

For release-style usage, npm is not required at runtime. FastAPI serves the built frontend directly from `frontend/dist`.

Start the application with:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch-app.ps1
```

`launch-app.ps1` is expected to:

- create `./local_int_venv` if it does not already exist
- install CUDA-enabled `torch` into that environment
- install `requirements.txt`
- download configured models into `./models` if missing
- start the FastAPI app on `http://localhost:8000`
- open the application in the browser

## Frontend Build

Node.js and npm are only required when building or rebuilding the frontend.

Build the frontend once before using `launch-app.ps1`:

```powershell
Set-Location .\frontend
npm install
npm run build
Set-Location ..
```

This generates the release frontend under `frontend/dist`.

## Development Mode

If you are editing the frontend, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch-dev.ps1
```

This mode requires Node.js and npm because it runs the Vite development server.

## Optional Node.js Install

If you need Node.js on Windows, install the LTS version with:

```powershell
winget install OpenJS.NodeJS.LTS
```

Then verify:

```powershell
node -v
npm -v
```

If `node` or `npm` is still not found, add the usual Node.js install paths to your user `PATH`:

```powershell
$nodePaths = @(
  "$env:ProgramFiles\nodejs",
  "$env:AppData\npm"
)

$currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$missingPaths = $nodePaths | Where-Object { $_ -and ($currentUserPath -notlike "*$_*") }

if ($missingPaths.Count -gt 0) {
  $updatedPath = ($currentUserPath, $missingPaths -join ";").Trim(";")
  [Environment]::SetEnvironmentVariable("Path", $updatedPath, "User")
}
```

Restart the terminal and verify again:

```powershell
node -v
npm -v
```

## Upcoming Features

Planned next improvements include:

- DOCX support
- image-aware document support
- better support for low-end PCs
- lower-memory runtime options
- more packaging and deployment polish
- continued refinement of corpus and single-document workflows
