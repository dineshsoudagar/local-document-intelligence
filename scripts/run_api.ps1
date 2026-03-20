param(
    [int]$Port = 8000,
    [int]$MaxPortSearch = 20
)

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

        try {
            $process = Get-Process -Id $pidValue -ErrorAction Stop
            Write-Host "Process name: $($process.ProcessName)"

            Stop-Process -Id $pidValue -Force -ErrorAction Stop
            Write-Host "Killed PID=$pidValue"
        }
        catch {
            Write-Host "Failed to kill PID=$pidValue"
            Write-Host $_.Exception.Message
        }
    }

    Start-Sleep -Seconds 2

    return (Test-PortFree -LocalPort $LocalPort)
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
python -m uvicorn src.api.main:app --port $selectedPort --reload