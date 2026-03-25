param(
    [int]$Port = 8000,
    [switch]$ForceModelDownload = $false,
    [switch]$ForceTorchInstall = $false,
    [switch]$SkipRequirementsInstall = $false,
    [switch]$OpenBrowser = $true,
    [string]$TorchVersion = "2.10.0",
    [string]$TorchVisionVersion = "0.25.0",
    [string]$TorchAudioVersion = "2.10.0",
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
$ExpectedTorchVisionVersion = "$TorchVisionVersion+$TorchCudaChannel"
$ExpectedTorchAudioVersion = "$TorchAudioVersion+$TorchCudaChannel"

function Test-PythonCandidate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Prefix = @()
    )

    $probeCode = @(
        "import ssl, sys"
        "raise SystemExit(0 if sys.version_info >= (3, 11) else 3)"
    ) -join "; "

    try {
        & $FilePath @($Prefix + @("-c", $probeCode)) 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Get-SystemPythonCommand {
    $candidates = [System.Collections.Generic.List[hashtable]]::new()
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($versionSelector in @("-3.13", "-3.12", "-3.11", "-3")) {
            $candidates.Add(@{
                FilePath = $pyLauncher.Source
                Prefix = @($versionSelector)
                Label = "py $versionSelector"
            })
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates.Add(@{
            FilePath = $pythonCommand.Source
            Prefix = @()
            Label = $pythonCommand.Source
        })
    }

    foreach ($candidate in $candidates) {
        if (Test-PythonCandidate -FilePath $candidate.FilePath -Prefix $candidate.Prefix) {
            Write-Host "Using Python interpreter: $($candidate.Label)"
            return @{
                FilePath = $candidate.FilePath
                Prefix = $candidate.Prefix
            }
        }
    }

    throw "Python 3.11+ with a working SSL module was not found. Install or repair a standard Python build and make sure 'py' or 'python' is available on PATH."
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

function Test-VenvPython {
    if (-not (Test-Path $VenvPython)) {
        return $false
    }

    return Test-PythonCandidate -FilePath $VenvPython
}

function Start-BrowserOnReadyJob {
    param(
        [int]$Attempts = 90,
        [int]$DelaySeconds = 2
    )

    Start-Job `
        -Name "local-document-intelligence-open-browser-$Port" `
        -ScriptBlock {
            param(
                [string]$BrowserHealthUrl,
                [string]$BrowserAppUrl,
                [int]$BrowserAttempts,
                [int]$BrowserDelaySeconds
            )

            for ($attempt = 0; $attempt -lt $BrowserAttempts; $attempt += 1) {
                try {
                    $response = Invoke-RestMethod -Uri $BrowserHealthUrl -Method Get -TimeoutSec 5
                    if ($response.status -eq "ok") {
                        Start-Process $BrowserAppUrl
                        return
                    }
                }
                catch {
                }

                Start-Sleep -Seconds $BrowserDelaySeconds
            }
        } `
        -ArgumentList $HealthUrl, $AppUrl, $Attempts, $DelaySeconds | Out-Null
}

function Get-InstalledTorchPackageVersions {
    try {
        $versionOutput = & $VenvPython -c "import importlib.metadata, sys;`npkgs = ('torch', 'torchvision', 'torchaudio')`nfor name in pkgs:`n    try:`n        print(f'{name}={importlib.metadata.version(name)}')`n    except importlib.metadata.PackageNotFoundError:`n        sys.exit(2)"
        if ($LASTEXITCODE -ne 0) {
            return $null
        }

        $versions = @{}
        foreach ($line in $versionOutput) {
            $entry = ($line -as [string]).Trim()
            if (-not $entry) {
                continue
            }

            $name, $version = $entry -split "=", 2
            if ($name -and $version) {
                $versions[$name] = $version
            }
        }

        if ($versions.Count -ne 3) {
            return $null
        }

        return $versions
    }
    catch {
        return $null
    }
}

function Ensure-TorchInstalled {
    $installedTorchPackages = Get-InstalledTorchPackageVersions
    $torchPackagesMatch = $installedTorchPackages `
        -and $installedTorchPackages["torch"] -eq $ExpectedTorchVersion `
        -and $installedTorchPackages["torchvision"] -eq $ExpectedTorchVisionVersion `
        -and $installedTorchPackages["torchaudio"] -eq $ExpectedTorchAudioVersion

    if (-not $ForceTorchInstall -and $torchPackagesMatch) {
        Write-Host "CUDA PyTorch packages are already installed: torch $ExpectedTorchVersion, torchvision $ExpectedTorchVisionVersion, torchaudio $ExpectedTorchAudioVersion."
        return
    }

    if ($installedTorchPackages) {
        Write-Host ("Replacing torch stack with official CUDA builds. Current versions: " +
            "torch {0}, torchvision {1}, torchaudio {2}" -f
            $installedTorchPackages["torch"],
            $installedTorchPackages["torchvision"],
            $installedTorchPackages["torchaudio"])
    }
    else {
        Write-Host "Installing official CUDA PyTorch packages from $TorchIndexUrl"
    }

    $torchInstallArgs = @(
        "-m", "pip", "install",
        "torch==$TorchVersion",
        "torchvision==$TorchVisionVersion",
        "torchaudio==$TorchAudioVersion",
        "--index-url", $TorchIndexUrl
    )

    if ($ForceTorchInstall -or $installedTorchPackages) {
        $torchInstallArgs += "--force-reinstall"
    }

    Invoke-CheckedCommand `
        -FilePath $VenvPython `
        -Arguments $torchInstallArgs `
        -FailureMessage "Failed to install CUDA-enabled PyTorch from the official PyTorch wheel index."

    $installedTorchPackages = Get-InstalledTorchPackageVersions
    $torchPackagesMatch = $installedTorchPackages `
        -and $installedTorchPackages["torch"] -eq $ExpectedTorchVersion `
        -and $installedTorchPackages["torchvision"] -eq $ExpectedTorchVisionVersion `
        -and $installedTorchPackages["torchaudio"] -eq $ExpectedTorchAudioVersion

    if (-not $torchPackagesMatch) {
        $installedSummary = if ($installedTorchPackages) {
            "torch {0}, torchvision {1}, torchaudio {2}" -f
                $installedTorchPackages["torch"],
                $installedTorchPackages["torchvision"],
                $installedTorchPackages["torchaudio"]
        }
        else {
            "not installed"
        }

        throw ("PyTorch installation completed, but the installed package set is '{0}' instead of " +
            "'torch {1}, torchvision {2}, torchaudio {3}'." -f
            $installedSummary,
            $ExpectedTorchVersion,
            $ExpectedTorchVisionVersion,
            $ExpectedTorchAudioVersion)
    }

    Write-Host ("Installed CUDA PyTorch packages from {0}: torch {1}, torchvision {2}, torchaudio {3}" -f
        $TorchIndexUrl,
        $installedTorchPackages["torch"],
        $installedTorchPackages["torchvision"],
        $installedTorchPackages["torchaudio"])
}

if (-not (Test-Path $FrontendIndexPath)) {
    throw "Built frontend not found at '$FrontendIndexPath'. Run 'npm run build' inside the 'frontend' folder before using launch-app.ps1."
}

$systemPython = Get-SystemPythonCommand

if ((Test-Path $VenvDir) -and -not (Test-VenvPython)) {
    Write-Host "Existing virtual environment at $VenvDir is invalid. Recreating it with the selected system Python."
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}

if (-not (Test-VenvPython)) {
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

    Invoke-CheckedCommand `
        -FilePath $VenvPython `
        -Arguments @("-m", "pip", "install", "-r", $RequirementsPath) `
        -FailureMessage "Failed to install Python requirements."

    Ensure-TorchInstalled
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
if ($OpenBrowser) {
    Start-BrowserOnReadyJob
}

Write-Host "Application will run in this terminal."
Write-Host "Press Ctrl+C to stop the server."

Push-Location $ProjectRoot
try {
    & $VenvPython @(
        "-m", "uvicorn",
        "src.api.main:app",
        "--host", "localhost",
        "--port", "$Port"
    )

    $serverExitCode = $LASTEXITCODE
    if ($serverExitCode -ne 0 -and $serverExitCode -ne 130 -and $serverExitCode -ne -1073741510) {
        throw "The application exited with code $serverExitCode."
    }
}
finally {
    Pop-Location
}
