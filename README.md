# local-document-intelligence

A fully local document intelligence system for uploading PDFs, indexing them into a private knowledge base, and querying them through a local retrieval-augmented generation pipeline.

## What It Does

- Stores uploaded documents locally
- Indexes document chunks for retrieval
- Runs retrieval and answer generation against local models
- Serves a FastAPI backend and a React frontend

## Prerequisites

- Python 3.11 or newer
- PowerShell

## Release Runtime

For release-style usage, npm is not required at runtime. FastAPI serves the built frontend directly from `frontend/dist`.

Start the app with one command:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch-app.ps1
```

`launch-app.ps1` will:

- create `.\local_int_venv` if it does not exist
- install CUDA-enabled `torch` from the official PyTorch wheel index into that environment
- install `requirements.txt` into that environment
- download the configured models into `.\models` if they are not already present
- start the FastAPI app on `http://localhost:8000`
- open the app in your browser

## Frontend Build

You only need Node.js and npm when you want to build or rebuild the frontend.

Build the frontend once before using `launch-app.ps1`:

```powershell
Set-Location .\frontend
npm install
npm run build
Set-Location ..
```

That generates the release UI under `frontend/dist`.

## Development Mode

If you are editing the frontend, use the dev launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch-dev.ps1
```

That mode still requires Node.js and npm because it runs the Vite dev server.

## Optional Node.js Install

If you need to build the frontend, install Node.js LTS on Windows with `winget`:

```powershell
winget install OpenJS.NodeJS.LTS
```

Then verify:

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

Restart the terminal and verify again:

```powershell
node -v
npm -v
```

## Notes

- `launch-app.ps1` expects a built frontend at `frontend/dist/index.html`.
- `launch-app.ps1` installs `torch==2.10.0` from the official PyTorch CUDA 12.8 wheel index by default.
- Models are stored under the local `models/` directory.
- `requirements.txt` intentionally does not install `torch` directly. PyTorch is installed separately so the launcher can use the official CUDA wheel instead of a generic pip resolution.
- If a Hugging Face token is required for any model in your environment, set `HF_TOKEN` before running the downloader.
