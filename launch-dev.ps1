param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ProjectRoot = $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$VenvPython = Join-Path $ProjectRoot "local_int_venv\Scripts\python.exe"

if (-not (Test-Path $FrontendRoot)) {
    throw "Frontend folder not found: $FrontendRoot"
}

function Resolve-PythonExecutable {
    param(
        [string]$PreferredPython
    )

    if (Test-Path $PreferredPython) {
        return $PreferredPython
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Python executable not found. Expected '$PreferredPython' or 'python' on PATH."
}

$PythonExe = Resolve-PythonExecutable -PreferredPython $VenvPython

Write-Host "Starting backend in background on http://localhost:$BackendPort"
$backendProcess = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @(
        "-m", "uvicorn",
        "src.api.main:app",
        "--host", "localhost",
        "--port", "$BackendPort",
        "--reload"
    ) `
    -WorkingDirectory $ProjectRoot `
    -NoNewWindow `
    -PassThru

try {
    Write-Host "Starting frontend in foreground on http://localhost:$FrontendPort"
    Push-Location $FrontendRoot
    npm.cmd run dev -- --host localhost --port $FrontendPort --strictPort
}
finally {
    Pop-Location

    if ($backendProcess -and -not $backendProcess.HasExited) {
        Write-Host "Stopping backend process $($backendProcess.Id)"
        Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
