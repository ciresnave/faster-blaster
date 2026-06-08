# Configure and Build faster-blaster with Source-Built Backends
# This script builds all backend libraries from source with maximum performance optimizations

param(
    [switch]$SkipBLIS,
    [switch]$SkipOpenBLAS,
    [switch]$SkipCLBlast,
    [switch]$Clean,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Configure and Build faster-blaster with Optimized Backends

Usage: .\configure_optimized.ps1 [options]

Options:
  -SkipBLIS      Don't build BLIS from source
  -SkipOpenBLAS  Don't build OpenBLAS from source
  -SkipCLBlast   Don't build CLBlast from source
  -Clean         Clean build directory before configuring
  -Help          Show this help message

Examples:
  # Build everything with max optimizations:
  .\configure_optimized.ps1

  # Clean build with everything:
  .\configure_optimized.ps1 -Clean

  # Build only BLIS and OpenBLAS:
  .\configure_optimized.ps1 -SkipCLBlast

Description:
  This script configures faster-blaster to build all backend libraries
  from source with CPU/GPU-specific optimizations for maximum performance.
  
  The build process will:
  - Auto-detect your CPU (AMD Zen2/3/4, Intel Skylake/Ice Lake, etc.)
  - Auto-detect your GPU (NVIDIA CUDA, AMD ROCm, OpenCL)
  - Build BLIS with architecture-specific kernels
  - Build OpenBLAS with optimal targets
  - Build CLBlast for GPU acceleration
  - Apply maximum compiler optimizations (-march=native, /arch:AVX2, etc.)
  
  All built libraries are installed to: build/backends-install/
"@
    exit 0
}

$ErrorActionPreference = "Stop"

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  faster-blaster - Maximum Performance Configuration" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check for CMake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Host "ERROR: CMake not found. Please install CMake 3.15 or later." -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ CMake: $($cmake.Version)" -ForegroundColor Green

# Check for WSL (needed for BLIS on Windows)
if ($PSVersionTable.Platform -ne "Unix") {
    $wsl = Get-Command wsl -ErrorAction SilentlyContinue
    if (-not $wsl) {
        Write-Host "  ⚠ WSL not found - BLIS build will be skipped on Windows" -ForegroundColor Yellow
        $SkipBLIS = $true
    } else {
        Write-Host "  ✓ WSL: Available" -ForegroundColor Green
    }
}

# Check for make/build tools
if ($PSVersionTable.Platform -eq "Unix") {
    $make = Get-Command make -ErrorAction SilentlyContinue
    if (-not $make) {
        Write-Host "  ⚠ make not found - install build-essential" -ForegroundColor Yellow
    } else {
        Write-Host "  ✓ make: Available" -ForegroundColor Green
    }
}

# Check for OpenCL (optional)
$opencl = Get-Command clinfo -ErrorAction SilentlyContinue
if ($opencl) {
    Write-Host "  ✓ OpenCL: Available" -ForegroundColor Green
} else {
    Write-Host "  ⚠ OpenCL not detected - CLBlast will be skipped" -ForegroundColor Yellow
    $SkipCLBlast = $true
}

Write-Host ""

# Clean build directory if requested
if ($Clean) {
    if (Test-Path "build") {
        Write-Host "Cleaning build directory..." -ForegroundColor Yellow
        Remove-Item -Path "build" -Recurse -Force
        Write-Host "  ✓ Build directory cleaned" -ForegroundColor Green
    }
}

# Create build directory
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Set-Location "build"

# Configure CMake options
$cmakeArgs = @(
    "..",
    "-G", "Ninja",
    "-DCMAKE_C_COMPILER=clang-cl",
    "-DCMAKE_CXX_COMPILER=clang-cl",
    "-DCMAKE_LINKER_TYPE=LLD",
    "-DCMAKE_CUDA_HOST_COMPILER=clang-cl",
    "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler",
    "-DFB_BUILD_BACKENDS_FROM_SOURCE=ON"
)

if ($SkipBLIS) {
    $cmakeArgs += "-DFB_BUILD_BLIS_FROM_SOURCE=OFF"
} else {
    Write-Host "BLIS will be built from source with CPU-specific optimizations" -ForegroundColor Cyan
    $cmakeArgs += "-DFB_BUILD_BLIS_FROM_SOURCE=ON"
}

if ($SkipOpenBLAS) {
    $cmakeArgs += "-DFB_BUILD_OPENBLAS_FROM_SOURCE=OFF"
} else {
    Write-Host "OpenBLAS will be built from source with CPU-specific optimizations" -ForegroundColor Cyan
    $cmakeArgs += "-DFB_BUILD_OPENBLAS_FROM_SOURCE=ON"
}

if ($SkipCLBlast) {
    $cmakeArgs += "-DFB_BUILD_CLBLAST_FROM_SOURCE=OFF"
} else {
    Write-Host "CLBlast will be built from source for GPU acceleration" -ForegroundColor Cyan
    $cmakeArgs += "-DFB_BUILD_CLBLAST_FROM_SOURCE=ON"
}

Write-Host ""
Write-Host "Configuring faster-blaster..." -ForegroundColor Yellow
Write-Host "CMake command: cmake $($cmakeArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Run CMake configure
& cmake @cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Configuration failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "  Configuration complete!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Build backends: cmake --build . --target blis-backend openblas-backend" -ForegroundColor White
Write-Host "  2. Build faster-blaster: cmake --build . --config Release" -ForegroundColor White
Write-Host "  3. Run tests: ctest -C Release" -ForegroundColor White
Write-Host ""
Write-Host "Or build everything at once:" -ForegroundColor Cyan
Write-Host "  cmake --build . --config Release -j" -ForegroundColor White
Write-Host ""
Write-Host "Built libraries will be in: build/backends-install/" -ForegroundColor Yellow
Write-Host ""

Set-Location ..
