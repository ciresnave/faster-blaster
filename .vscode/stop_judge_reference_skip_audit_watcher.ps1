$artifactDir = 'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\build-clang\tests'
$statusPath = Join-Path $artifactDir 'judge_reference_skip_audit_latest_status.txt'
$tailPath = Join-Path $artifactDir 'judge_reference_skip_audit_latest_tail.txt'
$workerPidPath = Join-Path $artifactDir 'judge_reference_skip_audit_latest_worker_pid.txt'
$runnerPidPath = Join-Path $artifactDir 'judge_reference_skip_audit_latest_runner_pid.txt'

function Read-KeyValueFile {
    param([string]$Path)

    $map = [ordered]@{}

    if (-not (Test-Path $Path)) {
        return $map
    }

    $raw = Get-Content $Path -Raw
    $matches = [regex]::Matches($raw, '(?ms)([A-Z_]+)=(.*?)(?=(?:\r?\n|\s+[A-Z_]+=)|$)')
    foreach ($match in $matches) {
        $map[$match.Groups[1].Value] = $match.Groups[2].Value.Trim()
    }

    return $map
}

function Write-KeyValueFile {
    param(
        [string]$Path,
        [hashtable]$Data
    )

    $lines = foreach ($entry in $Data.GetEnumerator()) {
        $entry.Key + '=' + $entry.Value
    }

    [System.IO.File]::WriteAllText($Path, ($lines -join "`r`n"), [System.Text.Encoding]::ASCII)
}

$stoppedIds = @()

foreach ($pidPath in @($runnerPidPath, $workerPidPath)) {
    if (-not (Test-Path $pidPath)) {
        continue
    }

    $pidValue = Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pidValue -and ($pidValue -match '^[0-9]+$')) {
        $process = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
        if ($process) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            $stoppedIds += $process.Id
        }
    }

    Remove-Item $pidPath -ErrorAction SilentlyContinue
}

$status = Read-KeyValueFile $statusPath
if ($status.Count -gt 0) {
    $status['STATE'] = 'STOPPED'
    $status['RUNNER_PID'] = '0'
    $status['WORKER_PID'] = '0'
    $status['UPDATED'] = (Get-Date -Format o)
    Write-KeyValueFile -Path $statusPath -Data $status
}

$tailMessage = if ($stoppedIds.Count -gt 0) {
    'Audit watcher stopped. PIDs=' + ($stoppedIds -join ',')
} else {
    'Audit watcher stop requested, but no active worker or runner PID was found.'
}

Set-Content $tailPath -Value $tailMessage -Encoding utf8
Write-Output $tailMessage