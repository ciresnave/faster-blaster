#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build and run GPU backend tests on AMD hardware

.DESCRIPTION
    Compiles gpu_backend_tests.cpp with HIP support and runs comprehensive
    tests of hipSOLVER and rocBLAS backends.
#>

param(
    [ValidateSet('Debug', 'Release')]
    [string]$BuildType = 'Release',
    
    [ValidateSet('hipsolver', 'rocblas', 'both')]
    [string]$Backend = 'both'
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GPU Backend Tests - AMD HIP/ROCm" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for HIP
$hipPath = $env:HIP_PATH
if (-not $hipPath) {
    $hipPath = "C:\Program Files\AMD\ROCm\6.4"
}

if (-not (Test-Path "$hipPath\bin\hipcc.exe")) {
    Write-Host "ERROR: HIP SDK not found at $hipPath" -ForegroundColor Red
    exit 1
}

Write-Host "Using HIP SDK: $hipPath" -ForegroundColor Green

# Source files
$testFile = "tests\gpu_backend_tests.cpp"
$outputDir = "build\tests"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# Common flags
$includeFlags = @(
    "-I.\include",
    "-I.\include\faster-blaster",
    "-I`"$hipPath\include`""
)

$libFlags = @(
    "-L`"$hipPath\lib`"",
    "-lhipsolver",
    "-lrocblas",
    "-lamdhip64"
)

$compileFlags = @(
    "-DWITH_HIP",
    "-D__HIP_PLATFORM_AMD__",
    "-std=c++14"
)

if ($BuildType -eq "Debug") {
    $compileFlags += "-g"
} else {
    $compileFlags += "-O3"
}

# Build and test function
function Build-And-Test {
    param([string]$BackendName, [string]$SourceFile)
    
    $outputExe = "$outputDir\gpu_backend_tests_$BackendName.exe"
    
    Write-Host "Compiling GPU backend tests ($BackendName)..." -ForegroundColor Yellow
    Write-Host "  Source: $testFile" -ForegroundColor Gray
    Write-Host "  Backend: $SourceFile" -ForegroundColor Gray
    Write-Host "  Output: $outputExe" -ForegroundColor Gray
    Write-Host ""
    
    $hipccCmd = @(
        $testFile,
        $SourceFile,
        "-o", $outputExe
    ) + $includeFlags + $libFlags + $compileFlags
    
    & "$hipPath\bin\hipcc.exe" @hipccCmd 2>&1 | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Compilation failed for $BackendName" -ForegroundColor Red
        return $false
    }
    
    Write-Host "✓ Build successful!" -ForegroundColor Green
    Write-Host ""
    
    # Run tests
    Write-Host "Running tests ($BackendName)..." -ForegroundColor Yellow
    Write-Host ""
    
    & $outputExe
    $testResult = $LASTEXITCODE
    
    Write-Host ""
    if ($testResult -eq 0) {
        Write-Host "✓ All $BackendName tests PASSED!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ Some $BackendName tests FAILED!" -ForegroundColor Red
        return $false
    }
}

# Run tests based on selection
$allPassed = $true

if ($Backend -eq 'hipsolver' -or $Backend -eq 'both') {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Testing hipSOLVER Backend" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    if (-not (Build-And-Test "hipsolver" "src\backends\gpu\hipsolver_trait_impl.cpp")) {
        $allPassed = $false
    }
}

if ($Backend -eq 'rocblas' -or $Backend -eq 'both') {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Testing rocBLAS Backend" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    if (-not (Build-And-Test "rocblas" "src\backends\gpu\rocblas_trait_impl.c")) {
        $allPassed = $false
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Test Suite Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($allPassed) {
    Write-Host "✓ All backends PASSED!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "✗ Some backends FAILED!" -ForegroundColor Red
    exit 1
}
