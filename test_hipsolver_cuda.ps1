#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build and run hipSOLVER LAPACK tests using CUDA/cuSOLVER backend

.DESCRIPTION
    Since hipSOLVER is compatible with cuSOLVER, we can test the LAPACK
    functions using NVIDIA CUDA toolchain.
#>

param(
    [ValidateSet('Debug', 'Release')]
    [string]$BuildType = 'Release'
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  hipSOLVER LAPACK Tests (via CUDA)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for CUDA
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
}

if (-not (Test-Path "$cudaPath\bin\nvcc.exe")) {
    Write-Host "ERROR: CUDA SDK not found at $cudaPath" -ForegroundColor Red
    exit 1
}

Write-Host "Using CUDA SDK: $cudaPath" -ForegroundColor Green
Write-Host "NOTE: Compiling hipSOLVER code targeting cuSOLVER (HIP→CUDA)" -ForegroundColor Yellow
Write-Host ""

# Source files
$testFile = "tests\gpu_backend_tests.cpp"
$hipsolverTrait = "src\backends\gpu\hipsolver_trait_impl.cpp"

# Output
$outputDir = "build\tests"
$outputExe = "$outputDir\gpu_backend_tests_hipsolver_cuda.exe"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# Compile flags for hipSOLVER→cuSOLVER mode
$includeFlags = @(
    "-I.\include",
    "-I.\include\faster-blaster",
    "-I`"$cudaPath\include`""
)

$libFlags = @(
    "-L`"$cudaPath\lib\x64`"",
    "-lcusolver",
    "-lcublas",
    "-lcudart"
)

$compileFlags = @(
    "-DWITH_HIP",
    "-DHIPSOLVER_TARGET_CUDA",  # Tell hipSOLVER to target cuSOLVER
    "-std=c++14"
)

if ($BuildType -eq "Debug") {
    $compileFlags += "-g"
    $compileFlags += "-G"
} else {
    $compileFlags += "-O3"
}

# Build command
Write-Host "Compiling hipSOLVER LAPACK tests (CUDA backend)..." -ForegroundColor Yellow
Write-Host "  Test: $testFile" -ForegroundColor Gray
Write-Host "  Trait: $hipsolverTrait" -ForegroundColor Gray
Write-Host "  Output: $outputExe" -ForegroundColor Gray
Write-Host ""

$nvccCmd = @(
    $testFile,
    $hipsolverTrait,
    "-o", $outputExe
) + $includeFlags + $libFlags + $compileFlags

try {
    & "$cudaPath\bin\nvcc.exe" @nvccCmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Compilation failed" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    Write-Host "✓ Build successful!" -ForegroundColor Green
    Write-Host ""
    
    # Run tests
    Write-Host "Running hipSOLVER LAPACK tests..." -ForegroundColor Yellow
    Write-Host ""
    
    & $outputExe
    $testResult = $LASTEXITCODE
    
    Write-Host ""
    if ($testResult -eq 0) {
        Write-Host "✓ All tests PASSED!" -ForegroundColor Green
    } else {
        Write-Host "✗ Some tests FAILED!" -ForegroundColor Red
    }
    
    exit $testResult
    
} catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    exit 1
}
