#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Validate hipSOLVER installation and run correctness tests

.DESCRIPTION
    Compiles and runs validation tests for hipSOLVER on NVIDIA RTX 4070.
    Tests verify:
    - LU factorization (GETRF)
    - Cholesky factorization (POTRF)
    - QR factorization (GEQRF)
    - Numerical correctness
    - Zero-cost abstraction (performance)

.EXAMPLE
    .\validate_hipsolver.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       hipSOLVER Validation - NVIDIA RTX 4070              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check for CUDA and hipSOLVER
Write-Host "[1/4] Checking dependencies..." -ForegroundColor Yellow

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    Write-Host "ERROR: CUDA_PATH not set" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ CUDA: $cudaPath" -ForegroundColor Green

$hipsolverPath = $env:HIPSOLVER_PATH
if (-not $hipsolverPath) {
    $hipsolverPath = "$PSScriptRoot\..\build\hipsolver_install"
    if (-not (Test-Path $hipsolverPath)) {
        Write-Host "ERROR: hipSOLVER not found. Run .\build_hipsolver.ps1 first" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  ✓ hipSOLVER: $hipsolverPath" -ForegroundColor Green

# Compile test
Write-Host ""
Write-Host "[2/4] Compiling validation tests..." -ForegroundColor Yellow

$testSource = "$PSScriptRoot\..\tests\test_hipsolver_validation.c"
$testExe = "$PSScriptRoot\..\build\test_hipsolver_validation.exe"

$buildDir = Split-Path $testExe
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
}

$nvccArgs = @(
    $testSource,
    "-o", $testExe,
    "-I$hipsolverPath\include",
    "-I$cudaPath\include",
    "-L$hipsolverPath\lib",
    "-L$cudaPath\lib\x64",
    "-lhipsolver",
    "-lcudart",
    "-lcusolver",
    "-lcublas",
    "-std=c++14"
)

Write-Host "  Compiling: nvcc $($nvccArgs -join ' ')" -ForegroundColor Gray

$nvcc = "$cudaPath\bin\nvcc.exe"
& $nvcc $nvccArgs 2>&1 | ForEach-Object {
    if ($_ -match "error") {
        Write-Host $_ -ForegroundColor Red
    } elseif ($_ -match "warning") {
        Write-Host $_ -ForegroundColor Yellow
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Compilation failed" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ Compilation successful" -ForegroundColor Green

# Set library paths
Write-Host ""
Write-Host "[3/4] Setting up runtime environment..." -ForegroundColor Yellow

$env:PATH = "$hipsolverPath\bin;$cudaPath\bin;$env:PATH"

Write-Host "  ✓ Library paths configured" -ForegroundColor Green

# Run tests
Write-Host ""
Write-Host "[4/4] Running validation tests..." -ForegroundColor Yellow
Write-Host ""

& $testExe

$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║         VALIDATION SUCCESSFUL - RTX 4070 READY            ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Build faster-blaster with hipSOLVER support" -ForegroundColor White
    Write-Host "  2. Run faster-blaster benchmarks to compare cuBLAS vs hipSOLVER" -ForegroundColor White
    Write-Host "  3. Measure zero-cost abstraction overhead (should be <1%)" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║              VALIDATION FAILED                             ║" -ForegroundColor Red
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above and ensure:" -ForegroundColor Yellow
    Write-Host "  - NVIDIA driver is up to date" -ForegroundColor White
    Write-Host "  - CUDA Toolkit is properly installed" -ForegroundColor White
    Write-Host "  - hipSOLVER was built correctly for CUDA backend" -ForegroundColor White
    Write-Host ""
}

exit $exitCode
