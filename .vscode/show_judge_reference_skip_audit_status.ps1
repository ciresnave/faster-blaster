$artifactDir = 'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\build-clang\tests'
$statusPath = Join-Path $artifactDir 'judge_reference_skip_audit_latest_status.txt'
$completionPath = Join-Path $artifactDir 'judge_reference_skip_audit_latest_completion.txt'
$tailPath = Join-Path $artifactDir 'judge_reference_skip_audit_latest_tail.txt'

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

$status = Read-KeyValueFile $statusPath
$completion = Read-KeyValueFile $completionPath

if ($status.Count -eq 0) {
    Write-Output 'No audit status file found.'
    exit 1
}

Write-Output 'Current audit status:'
foreach ($key in 'STATE', 'STAMP', 'UPDATED', 'REFERENCE_SKIP_COUNT', 'EXIT', 'RUNNER_PID', 'WORKER_PID', 'PENDING', 'LAST_CHANGE', 'TRIGGER_PATH', 'LOG', 'ERR', 'TAIL') {
    if ($status.Contains($key)) {
        Write-Output ('  ' + $key + '=' + $status[$key])
    }
}

if ($completion.Count -gt 0) {
    Write-Output ''
    Write-Output 'Latest completed audit:'
    foreach ($key in 'STATE', 'STAMP', 'UPDATED', 'REFERENCE_SKIP_COUNT', 'EXIT', 'LAST_CHANGE', 'TRIGGER_PATH', 'LOG', 'ERR') {
        if ($completion.Contains($key)) {
            Write-Output ('  ' + $key + '=' + $completion[$key])
        }
    }
}

if (Test-Path $tailPath) {
    Write-Output ''
    Write-Output 'Tail:'
    Get-Content $tailPath -Tail 40
}