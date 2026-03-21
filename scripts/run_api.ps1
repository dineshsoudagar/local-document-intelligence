param(
    [int]$Port = 8000,
    [int]$MaxPortSearch = 20,
    [switch]$KillProjectPythonProcesses = $true
)

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$AppModule = "src.api.main:app"

function Get-PortConnections {
    param(
        [int]$LocalPort
    )

    Get-NetTCPConnection -LocalPort $LocalPort -ErrorAction SilentlyContinue
}

function Test-PortFree {
    param(
        [int]$LocalPort
    )

    $connections = Get-PortConnections -LocalPort $LocalPort
    return -not $connections
}

function Stop-ProcessSafe {
    param(
        [int]$ProcessId
    )

    if ($ProcessId -in @(0, 4)) {
        Write-Host "Skipping system PID=$ProcessId"
        return $false
    }

    try {
        $process = Get-Process -Id $ProcessId -ErrorAction Stop
        Write-Host "Stopping PID=$ProcessId Name=$($process.ProcessName)"
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
        return $true
    }
    catch {
        Write-Host "Failed to stop PID=$ProcessId"
        Write-Host $_.Exception.Message
        return $false
    }
}

function Stop-PortOwners {
    param(
        [int]$LocalPort
    )

    $connections = Get-PortConnections -LocalPort $LocalPort

    if (-not $connections) {
        Write-Host "Port $LocalPort is already free."
        return $true
    }

    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($pidValue in $pids) {
        Write-Host "Port $LocalPort is owned by PID=$pidValue"
        [void](Stop-ProcessSafe -ProcessId $pidValue)
    }

    Start-Sleep -Seconds 2
    return (Test-PortFree -LocalPort $LocalPort)
}

function Get-ProjectPythonProcesses {
    param(
        [string]$RootPath,
        [string]$ModuleName
    )

    $escapedRoot = [Regex]::Escape($RootPath)
    $escapedModule = [Regex]::Escape($ModuleName)

    Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
        Where-Object {
            $_.CommandLine -and (
                $_.CommandLine -match $escapedModule -or
                $_.CommandLine -match "uvicorn" -or
                $_.CommandLine -match $escapedRoot
            )
        }
}

function Stop-ProjectPythonProcesses {
    param(
        [string]$RootPath,
        [string]$ModuleName
    )

    $processes = Get-ProjectPythonProcesses -RootPath $RootPath -ModuleName $ModuleName

    if (-not $processes) {
        Write-Host "No matching project Python processes found."
        return
    }

    foreach ($proc in $processes) {
        $pidValue = [int]$proc.ProcessId
        Write-Host "Found project Python process PID=$pidValue"
        Write-Host "CommandLine: $($proc.CommandLine)"
        [void](Stop-ProcessSafe -ProcessId $pidValue)
    }

    Start-Sleep -Seconds 2
}

function Find-NextFreePort {
    param(
        [int]$StartPort,
        [int]$MaxSearch
    )

    for ($candidate = $StartPort + 1; $candidate -le ($StartPort + $MaxSearch); $candidate++) {
        if (Test-PortFree -LocalPort $candidate) {
            return $candidate
        }
    }

    throw "No free port found in range $($StartPort + 1)-$($StartPort + $MaxSearch)."
}

if ($KillProjectPythonProcesses) {
    Write-Host "Stopping stale project Python processes..."
    Stop-ProjectPythonProcesses -RootPath $ProjectRoot -ModuleName $AppModule
}

$selectedPort = $Port

if (-not (Test-PortFree -LocalPort $selectedPort)) {
    Write-Host "Preferred port $selectedPort is busy. Trying to kill owner..."
    $freed = Stop-PortOwners -LocalPort $selectedPort

    if (-not $freed) {
        Write-Host "Port $selectedPort is still busy. Searching for another free port..."
        $selectedPort = Find-NextFreePort -StartPort $Port -MaxSearch $MaxPortSearch
    }
}

Write-Host "Starting API on port $selectedPort"
python -m uvicorn $AppModule --port $selectedPort --reload