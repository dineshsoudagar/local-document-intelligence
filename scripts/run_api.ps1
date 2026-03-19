param(
    [int]$Port = 8000
)

$connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue

if ($connections) {
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($pidValue in $pids) {
        Write-Host "Port $Port is owned by PID=$pidValue"

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
}

$stillUsed = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue

if ($stillUsed) {
    Write-Host "Port $Port is still in use. Aborting start."
    exit 1
}

python -m uvicorn src.api.main:app --port $Port