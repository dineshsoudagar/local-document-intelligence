param(
    [int]$Port = 8000,
    [switch]$ForceModelDownload = $false,
    [switch]$ForceTorchInstall = $false,
    [switch]$SkipRequirementsInstall = $false,
    [switch]$OpenBrowser = $true,
    [string]$TorchVersion = "2.10.0",
    [string]$TorchCudaChannel = "cu128"
)

$ProjectRoot = $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot "local_int_venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$FrontendIndexPath = Join-Path $ProjectRoot "frontend\dist\index.html"
$ModelManifestPath = Join-Path $ProjectRoot "models\manifest.json"
$HealthUrl = "http://localhost:$Port/healthz"
$AppUrl = "http://localhost:$Port/"
$TorchIndexUrl = "https://download.pytorch.org/whl/$TorchCudaChannel"
$ExpectedTorchVersion = "$TorchVersion+$TorchCudaChannel"

function Get-SystemPythonCommand {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return @{
            FilePath = $pythonCommand.Source
            Prefix = @()
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        return @{
            FilePath = $pyLauncher.Source
            Prefix = @("-3")
        }
    }

    throw "Python was not found. Install Python 3.11+ and make sure 'python' or 'py' is available on PATH."
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = $ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw $FailureMessage
        }
    }
    finally {
        Pop-Location
    }
}

function Test-AppHealth {
    try {
        $response = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 5
        return $response.status -eq "ok"
    }
    catch {
        return $false
    }
}

function Test-PortInUse {
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return [bool]$connections
}

function Wait-AppHealth {
    param(
        [int]$ProcessId,
        [int]$Attempts = 90,
        [int]$DelaySeconds = 2
    )

    for ($attempt = 0; $attempt -lt $Attempts; $attempt += 1) {
        if (Test-AppHealth) {
            return $true
        }

        if ($ProcessId) {
            $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
            if (-not $process) {
                return $false
            }
        }

        Start-Sleep -Seconds $DelaySeconds
    }

    return $false
}

function Get-InstalledTorchVersion {
    try {
        $versionOutput = & $VenvPython -c "import importlib.util, sys; spec = importlib.util.find_spec('torch');`nif spec is None:`n    sys.exit(2)`nimport torch`nprint(torch.__version__)"
        if ($LASTEXITCODE -ne 0) {
            return $null
        }

        return (($versionOutput | Select-Object -Last 1) -as [string]).Trim()
    }
    catch {
        return $null
    }
}

function Ensure-TorchInstalled {
    $installedTorchVersion = Get-InstalledTorchVersion

    if (-not $ForceTorchInstall -and $installedTorchVersion -eq $ExpectedTorchVersion) {
        Write-Host "CUDA PyTorch $installedTorchVersion is already installed."
        return
    }

    if ($installedTorchVersion) {
        Write-Host "Replacing torch $installedTorchVersion with official PyTorch build $ExpectedTorchVersion"
    }
    else {
        Write-Host "Installing official PyTorch build $ExpectedTorchVersion"
    }

    $torchInstallArgs = @(
        "-m", "pip", "install",
        "--upgrade",
        "--index-url", $TorchIndexUrl,
        "torch==$TorchVersion"
    )

    if ($ForceTorchInstall -or $installedTorchVersion) {
        $torchInstallArgs += "--force-reinstall"
    }

    Invoke-CheckedCommand `
        -FilePath $VenvPython `
        -Arguments $torchInstallArgs `
        -FailureMessage "Failed to install CUDA-enabled PyTorch from the official PyTorch wheel index."

    $installedTorchVersion = Get-InstalledTorchVersion
    if ($installedTorchVersion -ne $ExpectedTorchVersion) {
        throw "PyTorch installation completed, but the installed version is '$installedTorchVersion' instead of '$ExpectedTorchVersion'."
    }

    Write-Host "Installed CUDA PyTorch $installedTorchVersion from $TorchIndexUrl"
}

if (-not (Test-Path $FrontendIndexPath)) {
    throw "Built frontend not found at '$FrontendIndexPath'. Run 'npm run build' inside the 'frontend' folder before using launch-app.ps1."
}

$systemPython = Get-SystemPythonCommand

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment at $VenvDir"
    Invoke-CheckedCommand `
        -FilePath $systemPython.FilePath `
        -Arguments ($systemPython.Prefix + @("-m", "venv", $VenvDir)) `
        -FailureMessage "Failed to create the virtual environment."
}
else {
    Write-Host "Using existing virtual environment at $VenvDir"
}

if (-not $SkipRequirementsInstall) {
    Write-Host "Installing Python requirements"
    Invoke-CheckedCommand `
        -FilePath $VenvPython `
        -Arguments @("-m", "pip", "install", "--upgrade", "pip") `
        -FailureMessage "Failed to upgrade pip in the virtual environment."

    Ensure-TorchInstalled

    Invoke-CheckedCommand `
        -FilePath $VenvPython `
        -Arguments @("-m", "pip", "install", "-r", $RequirementsPath) `
        -FailureMessage "Failed to install Python requirements."
}
else {
    Ensure-TorchInstalled
}

if ($ForceModelDownload -or -not (Test-Path $ModelManifestPath)) {
    Write-Host "Downloading local models into the project models directory"
    Invoke-CheckedCommand `
        -FilePath $VenvPython `
        -Arguments @(
            (Join-Path $ProjectRoot "scripts\download_models.py"),
            "--project-root",
            $ProjectRoot
        ) `
        -FailureMessage "Failed to download the configured models."
}
else {
    Write-Host "Model manifest already exists at $ModelManifestPath. Skipping model download."
}

if (Test-AppHealth) {
    Write-Host "The application is already running at $AppUrl"
    if ($OpenBrowser) {
        Start-Process $AppUrl
    }
    return
}

if (Test-PortInUse) {
    throw "Port $Port is already in use, but the health endpoint at $HealthUrl is not responding for this app."
}

Write-Host "Starting the application on $AppUrl"
$serverProcess = Start-Process `
    -FilePath $VenvPython `
    -ArgumentList @(
        "-m", "uvicorn",
        "src.api.main:app",
        "--host", "localhost",
        "--port", "$Port"
    ) `
    -WorkingDirectory $ProjectRoot `
    -PassThru

if (-not (Wait-AppHealth -ProcessId $serverProcess.Id)) {
    throw "The application process started, but the health endpoint did not become ready at $HealthUrl."
}

Write-Host "Application is running at $AppUrl"
Write-Host "Server PID: $($serverProcess.Id)"

if ($OpenBrowser) {
    Start-Process $AppUrl
}
