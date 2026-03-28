param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [int]$StartupTimeoutSec = 90,
    [switch]$OpenBrowser = $false
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$VenvPython = Join-Path $ProjectRoot "local_int_venv\Scripts\python.exe"
$BackendHost = "localhost"
$FrontendHost = "localhost"
$BackendHealthUrl = "http://${BackendHost}:$BackendPort/healthz"
$BackendUrl = "http://${BackendHost}:$BackendPort/"
$FrontendUrl = "http://${FrontendHost}:$FrontendPort/"
$LogRoot = Join-Path $ProjectRoot ".launch-dev"
$BackendStdOutLog = Join-Path $LogRoot "backend.stdout.log"
$BackendStdErrLog = Join-Path $LogRoot "backend.stderr.log"

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

function Resolve-CommandExecutable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    $command = Get-Command $CommandName -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw $FailureMessage
}

function Ensure-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Reset-LogFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    }
}

function Get-PortListeners {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    try {
        return @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
    }
    catch {
        return @()
    }
}

function Assert-PortAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [Parameter(Mandatory = $true)]
        [string]$ServiceName
    )

    $listeners = Get-PortListeners -Port $Port
    if ($listeners.Count -eq 0) {
        return
    }

    $processIds = @(
        $listeners |
            Select-Object -ExpandProperty OwningProcess -Unique |
            Where-Object { $_ -and $_ -gt 0 }
    )

    $details = foreach ($processId in $processIds) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            if ($process.Path) {
                "PID ${processId}: $($process.ProcessName) ($($process.Path))"
            }
            else {
                "PID ${processId}: $($process.ProcessName)"
            }
        }
        else {
            "PID ${processId}: unknown"
        }
    }

    if ($details.Count -eq 0) {
        throw "$ServiceName port $Port is already in use."
    }

    throw @(
        "$ServiceName port $Port is already in use."
        "Listeners:"
        ($details -join [Environment]::NewLine)
    ) -join [Environment]::NewLine
}

function Start-ManagedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string]$StdOutLog,
        [Parameter(Mandatory = $true)]
        [string]$StdErrLog
    )

    Reset-LogFile -Path $StdOutLog
    Reset-LogFile -Path $StdErrLog

    return Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $StdOutLog `
        -RedirectStandardError $StdErrLog `
        -PassThru `
        -WindowStyle Hidden
}

function Get-LogTail {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [int]$LineCount = 40
    )

    if (-not (Test-Path $Path)) {
        return "[no log output]"
    }

    $content = @(Get-Content -Path $Path -Tail $LineCount -ErrorAction SilentlyContinue)
    if ($content.Count -eq 0) {
        return "[no log output]"
    }

    return $content -join [Environment]::NewLine
}

function Format-ServiceFailure {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$Reason,
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [string]$StdOutLog,
        [Parameter(Mandatory = $true)]
        [string]$StdErrLog
    )

    $message = @("$Label failed: $Reason")

    if ($Process) {
        try {
            $Process.Refresh()
            if ($Process.HasExited) {
                $message += "Exit code: $($Process.ExitCode)"
            }
        }
        catch {
        }
    }

    $message += ""
    $message += "stdout tail:"
    $message += (Get-LogTail -Path $StdOutLog)
    $message += ""
    $message += "stderr tail:"
    $message += (Get-LogTail -Path $StdErrLog)

    return $message -join [Environment]::NewLine
}

function Wait-ForServiceReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Probe,
        [Parameter(Mandatory = $true)]
        [string]$StdOutLog,
        [Parameter(Mandatory = $true)]
        [string]$StdErrLog,
        [int]$TimeoutSec = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $Process.Refresh()
        }
        catch {
        }

        if ($Process.HasExited) {
            throw (Format-ServiceFailure `
                -Label $Label `
                -Reason "process exited before readiness checks completed." `
                -Process $Process `
                -StdOutLog $StdOutLog `
                -StdErrLog $StdErrLog)
        }

        if (& $Probe) {
            return
        }

        Start-Sleep -Milliseconds 500
    }

    throw (Format-ServiceFailure `
        -Label $Label `
        -Reason "timed out after $TimeoutSec seconds waiting for readiness." `
        -Process $Process `
        -StdOutLog $StdOutLog `
        -StdErrLog $StdErrLog)
}

function Test-BackendHealth {
    try {
        $response = Invoke-RestMethod -Uri $BackendHealthUrl -Method Get -TimeoutSec 3
        return $response.status -eq "ok"
    }
    catch {
        return $false
    }
}

function Stop-ProcessTree {
    param(
        [System.Diagnostics.Process]$Process
    )

    if (-not $Process) {
        return
    }

    try {
        $Process.Refresh()
    }
    catch {
        return
    }

    $rootProcessId = $Process.Id
    if ($rootProcessId -le 0) {
        return
    }

    $allProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    if ($allProcesses.Count -eq 0) {
        Stop-Process -Id $rootProcessId -Force -ErrorAction SilentlyContinue
        return
    }

    $queuedIds = [System.Collections.Generic.Queue[int]]::new()
    $discoveredIds = [System.Collections.Generic.HashSet[int]]::new()
    $queuedIds.Enqueue($rootProcessId)
    $discoveredIds.Add($rootProcessId) | Out-Null

    while ($queuedIds.Count -gt 0) {
        $parentId = $queuedIds.Dequeue()
        foreach ($child in $allProcesses | Where-Object { $_.ParentProcessId -eq $parentId }) {
            $childProcessId = [int]$child.ProcessId
            if ($discoveredIds.Add($childProcessId)) {
                $queuedIds.Enqueue($childProcessId)
            }
        }
    }

    foreach ($processId in ($discoveredIds | Sort-Object -Descending)) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
}

function Start-BrowserOnReadyJob {
    Start-Job `
        -Name "local-document-intelligence-open-browser-$FrontendPort" `
        -ScriptBlock {
            param(
                [string]$Url
            )

            for ($attempt = 0; $attempt -lt 60; $attempt += 1) {
                try {
                    Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 3 | Out-Null
                    Start-Process $Url | Out-Null
                    return
                }
                catch {
                }

                Start-Sleep -Seconds 1
            }
        } `
        -ArgumentList $FrontendUrl | Out-Null
}

if (-not (Test-Path $FrontendRoot)) {
    throw "Frontend folder not found: $FrontendRoot"
}

$PythonExe = Resolve-PythonExecutable -PreferredPython $VenvPython
$NpmExe = Resolve-CommandExecutable `
    -CommandName "npm.cmd" `
    -FailureMessage "npm.cmd was not found on PATH. Install Node.js and make sure npm is available."

Ensure-Directory -Path $LogRoot
Assert-PortAvailable -Port $BackendPort -ServiceName "Backend"
Assert-PortAvailable -Port $FrontendPort -ServiceName "Frontend"

$backendProcess = $null

try {
    $backendProcess = Start-ManagedProcess `
        -FilePath $PythonExe `
        -Arguments @(
            "-m", "uvicorn",
            "src.api.main:app",
            "--host", $BackendHost,
            "--port", "$BackendPort",
            "--reload"
        ) `
        -WorkingDirectory $ProjectRoot `
        -StdOutLog $BackendStdOutLog `
        -StdErrLog $BackendStdErrLog

    Wait-ForServiceReady `
        -Label "Backend" `
        -Process $backendProcess `
        -Probe { Test-BackendHealth } `
        -StdOutLog $BackendStdOutLog `
        -StdErrLog $BackendStdErrLog `
        -TimeoutSec $StartupTimeoutSec

    if ($OpenBrowser) {
        Start-BrowserOnReadyJob
    }

    Write-Host "Backend ready at $BackendUrl"
    Write-Host "Backend logs: $LogRoot"
    Write-Host "Starting frontend on $FrontendUrl"

    Push-Location $FrontendRoot
    try {
        & $NpmExe @(
            "run", "dev",
            "--",
            "--host", $FrontendHost,
            "--port", "$FrontendPort",
            "--strictPort"
        )

        $frontendExitCode = $LASTEXITCODE
        if ($frontendExitCode -ne 0 -and $frontendExitCode -ne 130 -and $frontendExitCode -ne -1073741510) {
            throw "Frontend dev server exited with code $frontendExitCode."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    Stop-ProcessTree -Process $backendProcess
}
