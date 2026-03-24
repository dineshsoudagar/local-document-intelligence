# local-document-intelligence

A fully local document intelligence system for building a private document workspace with persistent indexing, hybrid retrieval, reranking, and grounded answer generation over local models.

Documents stay on disk, retrieval runs against a local Qdrant index, and answer generation runs through locally loaded models.

## Features

- Upload and store PDF documents locally
- Build a persistent local knowledge base
- Search across the full document corpus
- Restrict search to a single selected document
- Use hybrid retrieval with dense and sparse search
- Rerank retrieved evidence before answer generation
- Generate grounded answers from local models
- Return partial or unsupported responses when evidence is weak instead of guessing
- Run through a local FastAPI backend and React frontend

## Demo

Add your demo GIF here.

<p align="center">
  <img src="demo.gif" App UI Demo " width="90%" style="margin: 1%"/>
</p>

## Getting Started

### Requirements

- Python 3.11 or newer
- PowerShell

### Hardware

The current setup is aimed at machines with at least **8 GB VRAM** for a comfortable local run with the present model stack.

This is not yet tuned for low-end PCs. Lower-memory support is planned in a later iteration.

### Run the application

Start the app with:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch-app.ps1
```

The launch script is expected to:

- create `./local_int_venv` if it does not already exist
- install CUDA-enabled `torch` into that environment
- install `requirements.txt`
- download configured models into `./models` if missing
- start the FastAPI app on `http://localhost:8000`
- open the application in the browser

## Query Modes

### Auto

Auto mode decides whether a query should be handled as normal assistant chat or as document-grounded retrieval.

Use this when you do not want to manually choose between free chat and document search.

### Chat

Chat mode skips retrieval and answers directly as a local assistant.

Use this for general interaction, system questions, or messages that do not require document evidence.

### Corpus

Corpus mode searches across the full indexed knowledge base.

Use this when the answer may be spread across multiple uploaded documents or when you want to search the entire local corpus.

### Single Document

Single document mode restricts retrieval to one selected document.

Use this when you want the answer to come only from one document instead of the full corpus.

## Models

Current default model stack:

- Dense embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Reranking: `Qwen/Qwen3-Reranker-0.6B`
- Answer generation: `Qwen/Qwen3-4B-Instruct-2507`
- Sparse retrieval: `Qdrant/bm25`

All models are stored locally under `models/`.

## How It Works

The system uses a local retrieval pipeline instead of a single vector lookup.

At a high level, the flow is:

1. Parse the uploaded document into chunks
2. Store chunk text and metadata locally
3. Index chunks into a local Qdrant collection
4. Retrieve candidates with dense and sparse search
5. Fuse retrieval results
6. Rerank the fused candidates
7. Build grounded context from the strongest evidence
8. Judge whether the evidence is sufficient
9. Generate either:
   - a grounded supported answer
   - a partial answer with explicit incompleteness
   - or an unsupported response instead of guessing

## Tech Stack

- Backend: FastAPI
- Frontend: React
- Vector store: Qdrant
- Parsing: Docling
- Local models: Qwen embedding, reranker, and generator models

## Frontend Development

The built frontend is already used for normal application startup.

Only rebuild the frontend if you are editing the UI.

### Run in development mode

```powershell
powershell -ExecutionPolicy Bypass -File .\launch-dev.ps1
```

This mode requires Node.js and npm because it runs the Vite development server.

### Rebuild the frontend

```powershell
Set-Location .\frontend
npm install
npm run build
Set-Location ..
```

This generates the production frontend under `frontend/dist`.

## Installing Node.js

Node.js and npm are only required when developing or rebuilding the frontend.

On Windows, install the LTS version with:

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

## Planned Improvements

- DOCX support
- Image-aware document support
- Better support for low-end PCs
- Lower-memory runtime options
- More packaging and deployment polish
- Continued refinement of corpus and single-document workflows
