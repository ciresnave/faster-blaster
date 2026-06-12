Remove-Item Env:FB_JUDGE_START_OP_ID -ErrorAction SilentlyContinue
Remove-Item Env:FB_JUDGE_END_OP_ID -ErrorAction SilentlyContinue
Remove-Item Env:FB_JUDGE_OP_FILTER -ErrorAction SilentlyContinue
Remove-Item Env:FB_JUDGE_TRACE_OP_START -ErrorAction SilentlyContinue
Remove-Item Env:FB_JUDGE_TRACE_OP_RETURN -ErrorAction SilentlyContinue
Remove-Item Env:FB_JUDGE_VALIDATE_HEAP -ErrorAction SilentlyContinue

$env:CMAKE_BUILD_PARALLEL_LEVEL = '1'
$env:CTEST_PARALLEL_LEVEL = '1'
$env:NINJAFLAGS = '-j1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'

function Write-AuditMetadata {
    param(
        [string]$Path,
        $Data
    )

    $lines = foreach ($key in $Data.Keys) {
        '{0}={1}' -f $key, $Data[$key]
    }

    [System.IO.File]::WriteAllLines($Path, $lines, [System.Text.Encoding]::ASCII)
}

function Sync-RuntimeArtifact {
    param(
        [string]$SourcePath,
        [string]$DestinationPath
    )

    if (-not (Test-Path $SourcePath)) {
        return 'Missing runtime artifact: ' + $SourcePath
    }

    try {
        Copy-Item $SourcePath $DestinationPath -Force
    } catch {
        return $_.Exception.Message -replace "`r?`n", ' '
    }

    return $null
}

$artifactDir = 'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\build-clang\tests'
$buildRoot = Split-Path -Parent $artifactDir
$repoRoot = Split-Path -Parent $buildRoot
$referenceRoot = Join-Path (Split-Path -Parent $repoRoot) 'faster-blaster-reference'
$referenceBuildRoot = Join-Path $referenceRoot 'build-extended'
$testExe = Join-Path $artifactDir 'test_judge_runner.exe'

$env:FB_JUDGE_SUPPRESS_OP_RESULTS = '1'
$env:FB_JUDGE_FAIL_ON_REFERENCE_SKIP = '1'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $artifactDir ('judge_reference_skip_audit_' + $stamp + '.txt')
$err = Join-Path $artifactDir ('judge_reference_skip_audit_' + $stamp + '.err.txt')
$latestLog = Join-Path $artifactDir 'judge_reference_skip_audit_latest.txt'
$latestErr = Join-Path $artifactDir 'judge_reference_skip_audit_latest.err.txt'
$tail = Join-Path $artifactDir 'judge_reference_skip_audit_latest_tail.txt'
$status = Join-Path $artifactDir 'judge_reference_skip_audit_latest_status.txt'
$completion = Join-Path $artifactDir 'judge_reference_skip_audit_latest_completion.txt'

Remove-Item $latestLog -ErrorAction SilentlyContinue
Remove-Item $latestErr -ErrorAction SilentlyContinue
Remove-Item $tail -ErrorAction SilentlyContinue

$code = 1
$nativeError = ''
$referenceSkipCount = 0

Write-AuditMetadata -Path $status -Data ([ordered]@{
    STATE = 'RUNNING'
    STAMP = $stamp
    UPDATED = (Get-Date).ToString('o')
    LOG = $log
    ERR = $err
    LATEST_LOG = $latestLog
    LATEST_ERR = $latestErr
    TAIL = $tail
    EXIT = ''
    SIZE = '0'
    ERR_SIZE = '0'
    REFERENCE_SKIP_COUNT = ''
    RUNNER_PID = $PID
    WORKER_PID = ''
    ERROR = ''
})

try {
    $syncErrors = @()
    foreach ($pair in @(
        @{
            Source = Join-Path $buildRoot 'faster-blaster.dll'
            Destination = Join-Path $artifactDir 'faster-blaster.dll'
        },
        @{
            Source = Join-Path $referenceBuildRoot 'libfaster_blaster_reference.dll'
            Destination = Join-Path $artifactDir 'libfaster_blaster_reference.dll'
        }
    )) {
        $syncError = Sync-RuntimeArtifact -SourcePath $pair.Source -DestinationPath $pair.Destination
        if (-not [string]::IsNullOrWhiteSpace($syncError)) {
            $syncErrors += $syncError
        }
    }

    if ($syncErrors.Count -gt 0) {
        $nativeError = $syncErrors -join ' | '
        $code = 1
    } else {
        Push-Location $artifactDir
        & $testExe 2>&1 | Tee-Object -FilePath $log
        $code = $LASTEXITCODE
    }
} catch {
    if ([string]::IsNullOrWhiteSpace($nativeError)) {
        $nativeError = $_.Exception.Message
    }
    if ($null -ne $LASTEXITCODE) {
        $code = $LASTEXITCODE
    }
} finally {
    Pop-Location -ErrorAction SilentlyContinue

    if (Test-Path $log) {
        Copy-Item $log $latestLog -Force
    }

    $referenceSkipLines = @()
    if (Test-Path $log) {
        $referenceSkipLines = @(Select-String -Path $log -Pattern '^\[REF-SKIP\]' | ForEach-Object { $_.Line })
        $referenceSkipCount = $referenceSkipLines.Count
    }

    if ($referenceSkipCount -gt 0) {
        $referenceSkipLines | Set-Content $err -Encoding utf8
        Copy-Item $err $latestErr -Force
    } else {
        '' | Set-Content $err -Encoding utf8
        '' | Set-Content $latestErr -Encoding utf8
    }

    if (Test-Path $log) {
        try {
            Get-Content $log -Tail 120 | Set-Content $tail -Encoding utf8
        } catch {
            'Unable to read stdout log tail.' | Set-Content $tail -Encoding utf8
        }
    }

    if ($referenceSkipCount -gt 0) {
        Add-Content $tail '--- STDERR ---'
        Get-Content $err -Tail 120 | Add-Content $tail
    }

    if (-not (Test-Path $tail)) {
        'No tail available.' | Set-Content $tail -Encoding utf8
    }

    $errSize = 0
    if (Test-Path $err) {
        $errSize = (Get-Item $err).Length
    }

    $size = 0
    if (Test-Path $log) {
        $size = (Get-Item $log).Length
    }

    $errorText = $nativeError -replace "`r?`n", ' '
    $payload = [ordered]@{
        STATE = 'COMPLETE'
        STAMP = $stamp
        UPDATED = (Get-Date).ToString('o')
        LOG = $log
        ERR = $err
        LATEST_LOG = $latestLog
        LATEST_ERR = $latestErr
        TAIL = $tail
        EXIT = $code
        SIZE = $size
        ERR_SIZE = $errSize
        REFERENCE_SKIP_COUNT = $referenceSkipCount
        RUNNER_PID = $PID
        WORKER_PID = ''
        ERROR = $errorText
    }

    Write-AuditMetadata -Path $status -Data $payload
    Write-AuditMetadata -Path $completion -Data $payload
}

Get-Content $tail
Write-Output ('LOG:' + $log)
Write-Output ('ERR:' + $err)
    Write-Output ('LATEST_LOG:' + $latestLog)
    Write-Output ('LATEST_ERR:' + $latestErr)
Write-Output ('TAIL:' + $tail)
Write-Output ('STATUS:' + $status)
Write-Output ('REFERENCE_SKIP_COUNT:' + $referenceSkipCount)
Write-Output ('DONE:' + $code)

exit $code