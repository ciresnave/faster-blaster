Remove-Item Env:FB_JUDGE_START_OP_ID -ErrorAction SilentlyContinue
Remove-Item Env:FB_JUDGE_END_OP_ID -ErrorAction SilentlyContinue
Remove-Item Env:FB_JUDGE_FAIL_ON_REFERENCE_SKIP -ErrorAction SilentlyContinue
Remove-Item Env:FB_JUDGE_REFERENCE_OP_LIST -ErrorAction SilentlyContinue

$env:CMAKE_BUILD_PARALLEL_LEVEL = '1'
$env:CTEST_PARALLEL_LEVEL = '1'
$env:NINJAFLAGS = '-j1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:BLIS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'

Set-Location 'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\build-clang\tests'

$env:FB_JUDGE_VALIDATE_HEAP = '1'
$env:FB_JUDGE_SUPPRESS_OP_RESULTS = '1'
$env:FB_JUDGE_TRACE_OP_START = '1'
$env:FB_JUDGE_TRACE_OP_RETURN = '1'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = '.\judge_trace_full_' + $stamp + '.txt'
$err = '.\judge_trace_full_' + $stamp + '.err.txt'
$tail = '.\judge_trace_full_latest_tail.txt'
$status = '.\judge_trace_full_latest_status.txt'

Remove-Item $tail -ErrorAction SilentlyContinue
Remove-Item $status -ErrorAction SilentlyContinue

$code = 1
$nativeError = ''

try {
    $process = Start-Process -FilePath '.\test_judge_runner.exe' -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
    $code = $process.ExitCode
} catch {
    $nativeError = $_.Exception.Message
    if ($null -ne $LASTEXITCODE) {
        $code = $LASTEXITCODE
    }
} finally {
    if (Test-Path $log) {
        try {
            Get-Content $log -Tail 80 | Set-Content $tail -Encoding utf8
        } catch {
            'Unable to read stdout log tail.' | Set-Content $tail -Encoding utf8
        }
    }

    if ((Test-Path $err) -and ((Get-Item $err).Length -gt 0)) {
        Add-Content $tail '--- STDERR ---'
        Get-Content $err -Tail 80 | Add-Content $tail
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

    @(
        'LOG=' + $log,
        'ERR=' + $err,
        'TAIL=' + $tail,
        'EXIT=' + $code,
        'SIZE=' + $size,
        'ERR_SIZE=' + $errSize,
        'ERROR=' + $nativeError
    ) | Set-Content $status -Encoding ascii
}

Get-Content $tail
Write-Output ('LOG:' + $log)
Write-Output ('ERR:' + $err)
Write-Output ('TAIL:' + $tail)
Write-Output ('STATUS:' + $status)
Write-Output ('DONE:' + $code)

exit $code