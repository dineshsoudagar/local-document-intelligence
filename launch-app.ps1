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
$VenvDir = Join-Path $ProjectRoot "test_venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$FrontendIndexPath = Join-Path $ProjectRoot "frontend\dist\index.html"
$BindHost = "127.0.0.1"
$HealthUrl = "http://${BindHost}:$Port/healthz"
$AppUrl = "http://${BindHost}:$Port/"
$TorchIndexUrl = "https://download.pytorch.org/whl/$TorchCudaChannel"
$ExpectedTorchVersion = "$TorchVersion+$TorchCudaChannel"
$ExpectedTorchVisionVersion = "$TorchVisionVersion+$TorchCudaChannel"
$ExpectedTorchAudioVersion = "$TorchAudioVersion+$TorchCudaChannel"

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

function Get-PortListeners {
    try {
        return @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
    }
    catch {
        return @()
    }
}

function Get-PortListenerDetails {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Listeners
    )

    $processIds = @(
        $Listeners |
            Select-Object -ExpandProperty OwningProcess -Unique |
            Where-Object { $_ -and $_ -gt 0 }
    )
    if ($processIds.Count -eq 0) {
        return @()
    }

    $processesById = @{}
    foreach ($process in @(Get-Process -Id $processIds -ErrorAction SilentlyContinue)) {
        $processesById[$process.Id] = $process
    }

    $details = foreach ($processId in $processIds) {
        $process = $processesById[$processId]
        if ($process) {
            [PSCustomObject]@{
                Id = $processId
                ProcessName = $process.ProcessName
                Path = $process.Path
            }
        }
        else {
            [PSCustomObject]@{
                Id = $processId
                ProcessName = "unknown"
                Path = $null
            }
        }
    }

    return @($details)
}

function Format-PortConflictMessage {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Listeners
    )

    $details = Get-PortListenerDetails -Listeners $Listeners
    if ($details.Count -eq 0) {
        return "Port $Port is already in use by another listener, but its owning process could not be resolved. Health endpoint checked: $HealthUrl"
    }

    $detailLines = $details | ForEach-Object {
        if ($_.Path) {
            "PID $($_.Id): $($_.ProcessName) ($($_.Path))"
        }
        else {
            "PID $($_.Id): $($_.ProcessName)"
        }
    }

    return @(
        "Port $Port is already in use by another listener, and the health endpoint at $HealthUrl is not responding for this app."
        "Listeners:"
        ($detailLines -join [Environment]::NewLine)
    ) -join [Environment]::NewLine
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

    Invoke-CheckedCommand `
        -FilePath $VenvPython `
        -Arguments @("-m", "pip", "install", "-r", $RequirementsPath) `
        -FailureMessage "Failed to install Python requirements."

    Ensure-TorchInstalled
}
else {
    Ensure-TorchInstalled
}

$modelDownloadArgs = @(
    (Join-Path $ProjectRoot "scripts\download_models.py"),
    "--project-root",
    $ProjectRoot
)
if ($ForceModelDownload) {
    $modelDownloadArgs += "--force"
}

Write-Host "Ensuring required local models are available"
Invoke-CheckedCommand `
    -FilePath $VenvPython `
    -Arguments $modelDownloadArgs `
    -FailureMessage "Failed to download the configured models."

if (Test-AppHealth) {
    Write-Host "The application is already running at $AppUrl"
    if ($OpenBrowser) {
        Start-Process $AppUrl
    }
    return
}

$portListeners = Get-PortListeners
if ($portListeners.Count -gt 0) {
    throw (Format-PortConflictMessage -Listeners $portListeners)
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
        "--host", $BindHost,
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
