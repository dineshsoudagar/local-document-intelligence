# local-document-intelligence

A fully local document intelligence system for uploading PDFs, indexing them into a private knowledge base, and querying them through a local retrieval-augmented generation pipeline.

## What It Does

- Stores uploaded documents locally
- Indexes document chunks for retrieval
- Runs retrieval and answer generation against local models
- Serves a FastAPI backend and a Vite/React frontend for interactive use

## Prerequisites

- Python 3.11 or newer
- Node.js and npm
- PowerShell

## Quick Start

### 1. Create the environment and install requirements

The launch script looks for a virtual environment at `local_int_venv`, so keep that name.

```powershell
python -m venv .\local_int_venv
.\local_int_venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

### 2. Download the models

This downloads the configured local models and Docling artifacts into the project `models/` directory.

```powershell
python .\scripts\download_models.py
```

Optional: download only specific assets.

```powershell
python .\scripts\download_models.py --only embedder reranker generator picture_description docling_artifacts
```

### Node.js and npm

`npm` is required. The launch script runs the frontend with `npm.cmd run dev`, so the project will not start fully without Node.js and npm available on `PATH`.

Install Node.js LTS on Windows with `winget`:

```powershell
winget install OpenJS.NodeJS.LTS
```

After installation, open a new terminal and verify:

```powershell
node -v
npm -v
```

If `node` or `npm` is still not found, add the standard Node.js install locations to your user `PATH`:

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

Then restart the terminal and verify again:

```powershell
node -v
npm -v
```

### 3. Run the project from the CLI

Install the frontend dependencies once, then start the backend and frontend together with the launch script.

```powershell
Set-Location .\frontend
npm install
Set-Location ..
powershell -ExecutionPolicy Bypass -File .\launch-dev.ps1
```

By default the script starts:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

## Notes

- The launch script prefers `.\local_int_venv\Scripts\python.exe` and falls back to `python` on `PATH` if that virtual environment does not exist.
- Models are stored under the local `models/` directory.
- If a Hugging Face token is required for any model in your environment, set `HF_TOKEN` before running the downloader.
