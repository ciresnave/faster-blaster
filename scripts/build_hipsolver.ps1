#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build hipSOLVER for NVIDIA CUDA backend

.DESCRIPTION
    This script builds hipSOLVER configured to use NVIDIA cuSOLVER as the backend.
    hipSOLVER provides a portable API that works across AMD and NVIDIA GPUs.
    
    On NVIDIA: hipSOLVER → cuSOLVER (via HIP CUDA backend)
    On AMD:    hipSOLVER → rocSOLVER (native)

.PARAMETER Target
    Target backend: 'cuda' for NVIDIA, 'rocm' for AMD (default: cuda)

.PARAMETER BuildType
    Build configuration: Debug, Release, RelWithDebInfo (default: Release)

.PARAMETER InstallPrefix
    Installation directory (default: .\build\hipsolver_install)

.EXAMPLE
    .\build_hipsolver.ps1 -Target cuda
    
.EXAMPLE
    .\build_hipsolver.ps1 -Target cuda -BuildType Debug
#>

param(
    [Parameter(Position=0)]
    [ValidateSet('cuda', 'rocm')]
    [string]$Target = 'cuda',
    
    [Parameter()]
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')]
    [string]$BuildType = 'Release',
    
    [Parameter()]
    [string]$InstallPrefix = "$PSScriptRoot\..\build\hipsolver_install"
)

$ErrorActionPreference = "Stop"

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       hipSOLVER Build Script for NVIDIA RTX 4070          ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Yellow

if ($Target -eq 'cuda') {
    # Check for CUDA Toolkit
    $cudaPath = $env:CUDA_PATH
    if (-not $cudaPath) {
        Write-Host "ERROR: CUDA Toolkit not found. Set CUDA_PATH environment variable." -ForegroundColor Red
        Write-Host "Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  ✓ CUDA Toolkit found: $cudaPath" -ForegroundColor Green
    
    # Check for HIP SDK (includes CUDA backend)
    $hipPath = $env:HIP_PATH
    if (-not $hipPath) {
        Write-Host "WARNING: HIP_PATH not set. Checking default locations..." -ForegroundColor Yellow
        
        $possiblePaths = @(
            "C:\Program Files\AMD\ROCm\*\bin",
            "$env:ProgramFiles\hip\bin"
        )
        
        foreach ($path in $possiblePaths) {
            if (Test-Path $path) {
                $hipPath = Split-Path $path
                $env:HIP_PATH = $hipPath
                Write-Host "  ✓ HIP SDK found: $hipPath" -ForegroundColor Green
                break
            }
        }
        
        if (-not $hipPath) {
            Write-Host "ERROR: HIP SDK not found. Please install HIP with CUDA backend support." -ForegroundColor Red
            Write-Host "Download from: https://github.com/ROCm-Developer-Tools/HIP" -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Host "  ✓ HIP SDK found: $hipPath" -ForegroundColor Green
    }
}

# Check for CMake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Host "ERROR: CMake not found. Install from https://cmake.org/download/" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ CMake found: $($cmake.Version)" -ForegroundColor Green

# Check for NVIDIA GPU
Write-Host ""
Write-Host "[2/6] Detecting NVIDIA GPU..." -ForegroundColor Yellow

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    $gpuInfo = & nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>$null
    if ($gpuInfo) {
        $parts = $gpuInfo -split ','
        $gpuName = $parts[0].Trim()
        $driverVersion = $parts[1].Trim()
        $computeCap = $parts[2].Trim()
        
        Write-Host "  ✓ GPU: $gpuName" -ForegroundColor Green
        Write-Host "  ✓ Driver: $driverVersion" -ForegroundColor Green
        Write-Host "  ✓ Compute Capability: $computeCap" -ForegroundColor Green
        
        if ($gpuName -match "RTX 4070") {
            Write-Host "  ✓ RTX 4070 detected - perfect for validation!" -ForegroundColor Green
        }
    }
} else {
    Write-Host "  WARNING: nvidia-smi not found. Cannot detect GPU." -ForegroundColor Yellow
}

# Create build directory
Write-Host ""
Write-Host "[3/6] Setting up build directory..." -ForegroundColor Yellow

$buildDir = "$PSScriptRoot\..\build\hipsolver_build"
$sourceDir = "$PSScriptRoot\..\external\hipSOLVER"

if (Test-Path $buildDir) {
    Write-Host "  Cleaning existing build directory..." -ForegroundColor Gray
    Remove-Item $buildDir -Recurse -Force
}

New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
Write-Host "  ✓ Build directory: $buildDir" -ForegroundColor Green

# Clone hipSOLVER if needed
Write-Host ""
Write-Host "[4/6] Obtaining hipSOLVER source..." -ForegroundColor Yellow

if (-not (Test-Path $sourceDir)) {
    Write-Host "  Cloning hipSOLVER from GitHub..." -ForegroundColor Gray
    
    $externalDir = "$PSScriptRoot\..\external"
    if (-not (Test-Path $externalDir)) {
        New-Item -ItemType Directory -Path $externalDir -Force | Out-Null
    }
    
    Push-Location $externalDir
    try {
        git clone --depth 1 --branch master https://github.com/ROCmSoftwarePlatform/hipSOLVER.git
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to clone hipSOLVER"
        }
        Write-Host "  ✓ hipSOLVER source cloned" -ForegroundColor Green
    } finally {
        Pop-Location
    }
} else {
    Write-Host "  ✓ hipSOLVER source already available" -ForegroundColor Green
}

# Configure with CMake
Write-Host ""
Write-Host "[5/6] Configuring hipSOLVER build..." -ForegroundColor Yellow

Push-Location $buildDir
try {
    $cmakeArgs = @(
        "-S", $sourceDir,
        "-B", ".",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_INSTALL_PREFIX=$InstallPrefix"
    )
    
    if ($Target -eq 'cuda') {
        $cmakeArgs += @(
            "-DUSE_CUDA=ON",
            "-DCMAKE_CUDA_COMPILER=$cudaPath\bin\nvcc.exe",
            "-DHIP_PLATFORM=nvidia"
        )
    } else {
        $cmakeArgs += @(
            "-DUSE_CUDA=OFF",
            "-DHIP_PLATFORM=amd"
        )
    }
    
    Write-Host "  Running: cmake $($cmakeArgs -join ' ')" -ForegroundColor Gray
    
    & cmake $cmakeArgs
    
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    
    Write-Host "  ✓ Configuration complete" -ForegroundColor Green
    
    # Build
    Write-Host ""
    Write-Host "[6/6] Building hipSOLVER..." -ForegroundColor Yellow
    
    & cmake --build . --config $BuildType --parallel
    
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    
    Write-Host "  ✓ Build complete" -ForegroundColor Green
    
    # Install
    Write-Host ""
    Write-Host "Installing to: $InstallPrefix" -ForegroundColor Yellow
    
    & cmake --install .
    
    if ($LASTEXITCODE -ne 0) {
        throw "Installation failed"
    }
    
    Write-Host "  ✓ Installation complete" -ForegroundColor Green
    
} finally {
    Pop-Location
}

# Summary
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                 BUILD SUCCESSFUL                           ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "hipSOLVER installed to: $InstallPrefix" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run validation tests: .\scripts\validate_hipsolver.ps1" -ForegroundColor White
Write-Host "  2. Build faster-blaster with hipSOLVER support" -ForegroundColor White
Write-Host ""
Write-Host "Environment variables to set:" -ForegroundColor Yellow
Write-Host "  `$env:HIPSOLVER_PATH = '$InstallPrefix'" -ForegroundColor White
Write-Host ""
