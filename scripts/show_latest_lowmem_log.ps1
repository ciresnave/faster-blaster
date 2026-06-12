$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$root = Join-Path $repoRoot "build-logs"
if (-not (Test-Path $root)) {
    Write-Host "No build-logs directory found." -ForegroundColor Yellow
    exit 1
}
$latest = Get-ChildItem $root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latest) {
    Write-Host "No low-memory logs found." -ForegroundColor Yellow
    exit 1
}
Write-Host "Latest log dir: $($latest.FullName)" -ForegroundColor Cyan
$mem = Join-Path $latest.FullName "memory_samples.csv"
if (Test-Path $mem) {
    Write-Host "Top memory samples (by working set MB):" -ForegroundColor Cyan
    Import-Csv $mem |
        Sort-Object {[double]$_.ws_mb} -Descending |
        Select-Object -First 15 timestamp,phase,proc,pid,ws_mb,pm_mb,sys_free_mb |
        Format-Table -AutoSize
} else {
    Write-Host "No memory_samples.csv found in latest log." -ForegroundColor Yellow
}
