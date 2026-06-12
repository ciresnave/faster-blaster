#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build and run GPU backend tests on NVIDIA hardware

.DESCRIPTION
    Compiles gpu_backend_tests.cpp with CUDA support and runs comprehensive
    tests of cuBLAS backend.
#>

param(
    [ValidateSet('Debug', 'Release')]
    [string]$BuildType = 'Release'
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GPU Backend Tests - NVIDIA cuBLAS" -ForegroundColor Cyan
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

# Source files
$testFile = "tests\gpu_backend_tests.cpp"
$cublasTrait = "src\backends\gpu\cublas_trait_impl.c"
$gpuBackend = "src\backends\gpu\gpu_backend.h"

# Output
$outputDir = "build\tests"
$outputExe = "$outputDir\gpu_backend_tests_cublas.exe"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# Compile flags
$includeFlags = @(
    "-I.\include",
    "-I.\include\faster-blaster",
    "-I`"$cudaPath\include`""
)

$libFlags = @(
    "-L`"$cudaPath\lib\x64`"",
    "-lcublas",
    "-lcusolver",
    "-lcudart"
)

$compileFlags = @(
    "-DWITH_CUDA",
    "-std=c++14"
)

if ($BuildType -eq "Debug") {
    $compileFlags += "-g"
    $compileFlags += "-G"
} else {
    $compileFlags += "-O3"
}

# Build command
Write-Host "Compiling GPU backend tests (CUDA)..." -ForegroundColor Yellow
Write-Host "  Source: $testFile" -ForegroundColor Gray
Write-Host "  Backend: $cublasTrait" -ForegroundColor Gray
Write-Host "  Output: $outputExe" -ForegroundColor Gray
Write-Host ""

$nvccCmd = @(
    $testFile,
    $cublasTrait,
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
    Write-Host "Running tests..." -ForegroundColor Yellow
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
