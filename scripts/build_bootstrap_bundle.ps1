param(
    [string]$Python = "python",
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BundleRoot = Join-Path $ProjectRoot "bundle"
$BundlePythonDir = Join-Path $BundleRoot "python"
$WheelhouseDir = Join-Path $BundleRoot "wheels"
$OnedirSpecPath = Join-Path $ProjectRoot "LocalDocumentIntelligence.spec"
$OnefileSpecPath = Join-Path $ProjectRoot "LocalDocumentIntelligence.onefile.spec"

function Get-PythonInstallRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonCommand
    )

    $installRootOutput = & $PythonCommand -c "import pathlib, sys; print(pathlib.Path(sys.base_prefix).resolve())"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to resolve the base Python installation root."
    }

    $installRoot = ($installRootOutput | Select-Object -Last 1).Trim()
    if (-not $installRoot) {
        throw "The base Python installation root was empty."
    }

    return $installRoot
}

function Copy-BootstrapPythonRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRoot,
        [Parameter(Mandatory = $true)]
        [string]$DestinationRoot
    )

    if (-not (Test-Path -LiteralPath $SourceRoot)) {
        throw "Python installation root not found at '$SourceRoot'."
    }

    if (Test-Path -LiteralPath $DestinationRoot) {
        Remove-Item -LiteralPath $DestinationRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

    foreach ($file in @("python.exe", "pythonw.exe", "LICENSE.txt")) {
        $sourcePath = Join-Path $SourceRoot $file
        if (Test-Path -LiteralPath $sourcePath) {
            Copy-Item -LiteralPath $sourcePath -Destination $DestinationRoot -Force
        }
    }

    foreach ($pattern in @("python*.dll", "vcruntime*.dll")) {
        Get-ChildItem -Path $SourceRoot -File -Filter $pattern | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $DestinationRoot -Force
        }
    }

    $sourceDllsDir = Join-Path $SourceRoot "DLLs"
    if (-not (Test-Path -LiteralPath $sourceDllsDir)) {
        throw "Required Python runtime directory missing: '$sourceDllsDir'."
    }
    Copy-Item -LiteralPath $sourceDllsDir -Destination $DestinationRoot -Recurse -Force

    $sourceLibDir = Join-Path $SourceRoot "Lib"
    if (-not (Test-Path -LiteralPath $sourceLibDir)) {
        throw "Required Python runtime directory missing: '$sourceLibDir'."
    }

    $destinationLibDir = Join-Path $DestinationRoot "Lib"
    New-Item -ItemType Directory -Force -Path $destinationLibDir | Out-Null
    Get-ChildItem -LiteralPath $sourceLibDir -Force | Where-Object {
        $_.Name -notin @("site-packages", "__pycache__")
    } | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $destinationLibDir -Recurse -Force
    }
}

Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Force -Path $BundleRoot | Out-Null
    $PythonInstallRoot = Get-PythonInstallRoot -PythonCommand $Python
    Copy-BootstrapPythonRuntime -SourceRoot $PythonInstallRoot -DestinationRoot $BundlePythonDir
    New-Item -ItemType Directory -Force -Path $WheelhouseDir | Out-Null

    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip."
    }

    & $Python -m pip install -r requirements.txt -r requirements-launcher.txt pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install launcher build dependencies."
    }

    & $Python -m pip wheel -r requirements.txt --wheel-dir $WheelhouseDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build the offline wheelhouse."
    }

    $SpecPath = $OnedirSpecPath
    if ($OneFile) {
        $SpecPath = $OnefileSpecPath
    }

    $PyInstallerArgs = @(
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        $SpecPath
    )

    & $Python @PyInstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
}
finally {
    Pop-Location
}
