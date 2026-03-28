param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$BundleScript = Join-Path $PSScriptRoot "build_bootstrap_bundle.ps1"
& $BundleScript -Python $Python -OneFile

if ($LASTEXITCODE -ne 0) {
    throw "One-file PyInstaller build failed."
}
