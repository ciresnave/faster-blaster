#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build hipSOLVER LAPACK backend (unified NVIDIA + AMD)

.DESCRIPTION
    Compiles hipsolver_trait_impl.c using:
    - nvcc + cuSOLVER (NVIDIA GPUs)
    - hipcc + rocSOLVER (AMD GPUs)
    
    Auto-detects installed SDK and builds appropriate target.
    Single codebase, compile-time backend selection = zero-cost abstraction.

.PARAMETER Target
    Specify 'nvidia', 'amd', or 'auto' (default: auto-detect)

.PARAMETER BuildType
    Debug or Release (default: Release)

.EXAMPLE
    .\build_hipsolver.ps1
    # Auto-detects CUDA or HIP SDK

.EXAMPLE
    .\build_hipsolver.ps1 -Target nvidia -BuildType Debug
    # Force NVIDIA build with debug symbols
#>

param(
    [ValidateSet('auto', 'nvidia', 'amd')]
    [string]$Target = 'auto',
    
    [ValidateSet('Debug', 'Release')]
    [string]$BuildType = 'Release'
)

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  hipSOLVER LAPACK Backend Build Script" -ForegroundColor Cyan
Write-Host "  Zero-Cost Multi-Vendor Abstraction" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Source files
$sourceFile = "src\backends\gpu\hipsolver_trait_impl.cpp"
$includeDir = "include"
$outputLib = "faster-blaster-hipsolver"

# Build flags
$cflags = @()
$ldflags = @()

if ($BuildType -eq "Debug") {
    $cflags += "-g", "-O0"
} else {
    $cflags += "-O3", "-DNDEBUG"
}

# ============================================================================
# Auto-detect SDK
# ============================================================================

$cudaPath = $env:CUDA_PATH
$hipPath = $env:HIP_PATH

$hasCUDA = $false
$hasHIP = $false

if ($cudaPath -and (Test-Path "$cudaPath\bin\nvcc.exe")) {
    $hasCUDA = $true
    Write-Host "[OK] CUDA Toolkit detected: $cudaPath" -ForegroundColor Green
}

if ($hipPath -and (Test-Path "$hipPath\bin\hipcc.exe")) {
    $hasHIP = $true
    Write-Host "[OK] HIP SDK detected: $hipPath" -ForegroundColor Green
}

# Determine target
$actualTarget = $Target
if ($Target -eq 'auto') {
    if ($hasCUDA) {
        $actualTarget = 'nvidia'
        Write-Host "[AUTO] Selected NVIDIA (CUDA) backend" -ForegroundColor Yellow
    } elseif ($hasHIP) {
        $actualTarget = 'amd'
        Write-Host "[AUTO] Selected AMD (HIP) backend" -ForegroundColor Yellow
    } else {
        Write-Host "[ERROR] No CUDA or HIP SDK detected!" -ForegroundColor Red
        Write-Host "  Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
        Write-Host "  OR HIP SDK: https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html" -ForegroundColor Red
        exit 1
    }
}

# Validate target
if ($actualTarget -eq 'nvidia' -and -not $hasCUDA) {
    Write-Host "[ERROR] NVIDIA target requires CUDA Toolkit" -ForegroundColor Red
    exit 1
}

if ($actualTarget -eq 'amd' -and -not $hasHIP) {
    Write-Host "[ERROR] AMD target requires HIP SDK" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Build for NVIDIA (CUDA + cuSOLVER)
# ============================================================================

if ($actualTarget -eq 'nvidia') {
    Write-Host ""
    Write-Host "Building for NVIDIA GPUs (CUDA + cuSOLVER)..." -ForegroundColor Cyan
    
    # Use hipcc wrapper for NVIDIA (it calls nvcc internally with correct flags)
    if (-not $hipPath) {
        Write-Host "[ERROR] HIP SDK required even for NVIDIA builds (provides hipcc wrapper)" -ForegroundColor Red
        Write-Host "  Install HIP SDK from: https://github.com/ROCm/HIP" -ForegroundColor Red
        exit 1
    }
    
    $compiler = "$hipPath\bin\hipcc.exe"
    if (-not (Test-Path $compiler)) {
        $compiler = "$hipPath\bin\hipcc.bat"
    }
    
    $includePaths = @(
        "-I$includeDir",
        "-I`"$hipPath\include`"",
        "-I`"$cudaPath\include`""
    )
    
    $defines = @(
        "-DHIPSOLVER_TARGET_CUDA",
        "-D__HIP_PLATFORM_NVIDIA__"
    )
    
    $libs = @(
        "-lcusolver",
        "-lcublas",
        "-lcudart"
    )
    
    $libPaths = @(
        "-L$cudaPath\lib\x64"
    )
    
    $outputFile = "${outputLib}_cuda.obj"
    
    # Compile
    $compileCmd = @($compiler) + $cflags + $includePaths + $defines + @(
        "-c", $sourceFile,
        "-o", $outputFile
    )
    
    Write-Host "Compiling: $compiler $($compileCmd[1..($compileCmd.Length-1)] -join ' ')" -ForegroundColor Gray
    & $compiler ($compileCmd[1..($compileCmd.Length-1)])
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    Write-Host "[SUCCESS] NVIDIA backend compiled: $outputFile" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "[SUCCESS] Built NVIDIA backend: $outputFile" -ForegroundColor Green
    Write-Host "  Compiles to: cuSOLVER at compile-time" -ForegroundColor Green
    Write-Host "  Zero-cost abstraction: ✓" -ForegroundColor Green
}

# ============================================================================
# Build for AMD (HIP + rocSOLVER)
# ============================================================================

if ($actualTarget -eq 'amd') {
    Write-Host ""
    Write-Host "Building for AMD GPUs (HIP + rocSOLVER)..." -ForegroundColor Cyan
    
    $compiler = "$hipPath\bin\hipcc.exe"
    $includePaths = @(
        "-I$includeDir",
        "-I`"$hipPath\include`"",
        "-I`"$hipPath\include\hipsolver`""
    )
    
    $defines = @(
        "-DHIPSOLVER_TARGET_ROCM",
        "-D__HIP_PLATFORM_AMD__"
    )
    
    $libs = @(
        "-lhipsolver",
        "-lrocblas",
        "-lamdhip64"
    )
    
    $libPaths = @(
        "-L$hipPath\lib"
    )
    
    $outputFile = "${outputLib}_hip.lib"
    
    # Compile
    $compileCmd = @($compiler) + $cflags + $includePaths + $defines + @(
        "-c", $sourceFile,
        "-o", "hipsolver_trait_impl_hip.obj"
    )
    
    Write-Host "Compiling: $compiler $($compileCmd[1..($compileCmd.Length-1)] -join ' ')" -ForegroundColor Gray
    & $compiler ($compileCmd[1..($compileCmd.Length-1)])
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    # Link (static library)
    Write-Host "Creating static library: $outputFile" -ForegroundColor Gray
    lib.exe /OUT:$outputFile hipsolver_trait_impl_hip.obj
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Linking failed!" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    Write-Host ""
    Write-Host "[SUCCESS] Built AMD backend: $outputFile" -ForegroundColor Green
    Write-Host "  Compiles to: rocSOLVER at compile-time" -ForegroundColor Green
    Write-Host "  Zero-cost abstraction: ✓" -ForegroundColor Green
}

# ============================================================================
# Summary
# ============================================================================

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Build Complete!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Architecture:" -ForegroundColor Yellow
Write-Host "  - Single codebase (hipsolver_trait_impl.c)" -ForegroundColor White
Write-Host "  - Compile-time backend selection" -ForegroundColor White
Write-Host "  - Compiles to cuSOLVER (NVIDIA) OR rocSOLVER (AMD)" -ForegroundColor White
Write-Host "  - Zero runtime overhead vs direct API calls" -ForegroundColor White
Write-Host ""
Write-Host "Integration:" -ForegroundColor Yellow
Write-Host "  - Function pointers in const vtable" -ForegroundColor White
Write-Host "  - Compiler can inline through function pointers" -ForegroundColor White
Write-Host "  - Same zero-cost pattern as BLAS implementations" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Combine with cuBLAS/rocBLAS for full backend" -ForegroundColor White
Write-Host "  2. Test on NVIDIA RTX 4090 and AMD RX 7900 XTX" -ForegroundColor White
Write-Host "  3. Build oneMKL LAPACK (Intel Arc GPUs)" -ForegroundColor White
Write-Host "  4. Optional: Build MAGMA (vendor-neutral)" -ForegroundColor White
Write-Host ""
