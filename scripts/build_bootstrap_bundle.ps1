param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BundleRoot = Join-Path $ProjectRoot "bundle"
$WheelhouseDir = Join-Path $BundleRoot "wheels"

Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Force -Path $BundleRoot | Out-Null
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

    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --name LocalDocumentIntelligence `
        --paths $ProjectRoot `
        --collect-submodules webview `
        --add-data "frontend/dist;frontend/dist" `
        --add-data "scripts;scripts" `
        --add-data "src;src" `
        --add-data "requirements.txt;." `
        --add-data "requirements-launcher.txt;." `
        launcher/desktop_launcher.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }
}
finally {
    Pop-Location
}
