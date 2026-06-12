$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$seedPath = Join-Path $repoRoot 'build-clang\tests\next_missing_thunks.txt'
$countPath = Join-Path $repoRoot 'build-clang\tests\next_missing_thunks_live_count.txt'
$listPath = Join-Path $repoRoot 'build-clang\tests\next_missing_thunks_live.txt'
$objPath = Join-Path $repoRoot 'build-clang\CMakeFiles\faster-blaster.dir\src\core\conv_thunks.c.obj'

if (-not (Test-Path $seedPath)) {
    Write-Error "Seed file not found: $seedPath"
    exit 1
}

if (-not (Test-Path $objPath)) {
    Write-Error "Object file not found: $objPath"
    exit 1
}

$ops = Get-Content $seedPath | Where-Object { $_ -and $_.Trim().Length -gt 0 }
$symbolText = cmd /c "dumpbin /symbols $objPath"

$missing = @()
foreach ($op in $ops) {
    $stem = $op.ToLower()
    $hasForward = $symbolText -match "thunk_${stem}_fortran_to_cblas"
    $hasReverse = $symbolText -match "thunk_${stem}_cblas_to_fortran"
    if (-not ($hasForward -and $hasReverse)) {
        $missing += $op
    }
}

New-Item -Path $listPath -ItemType File -Force | Out-Null
if ($missing.Count -eq 0) {
    Clear-Content $listPath
} else {
    $missing | Set-Content $listPath
}
Set-Content $countPath ($missing.Count.ToString())

if (-not (Test-Path $countPath)) {
    Write-Error "Count artifact not found: $countPath"
    exit 1
}

$count = (Get-Content $countPath | Select-Object -First 1).Trim()
Write-Output "Live missing thunk count: $count"
Write-Output "Seed file: $seedPath"
Write-Output "Object file: $objPath"
Write-Output "Count file: $countPath"
Write-Output "List file: $listPath"

if ($count -ne '0') {
    Write-Output 'Missing ops:'
    Get-Content $listPath | Where-Object { $_ -and $_.Trim().Length -gt 0 }
}
