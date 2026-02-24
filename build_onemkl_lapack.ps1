#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build Intel oneMKL LAPACK backend

.DESCRIPTION
    Compiles onemkl_lapack_impl.cpp using Intel DPC++ compiler (icpx) with SYCL
    and oneMKL libraries for LAPACK operations on Intel GPUs.

.PARAMETER BuildType
    Debug or Release (default: Release)

.EXAMPLE
    .\build_onemkl_lapack.ps1
    # Build Release version

.EXAMPLE
    .\build_onemkl_lapack.ps1 -BuildType Debug
    # Build with debug symbols
#>

param(
    [ValidateSet('Debug', 'Release')]
    [string]$BuildType = 'Release'
)

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Intel oneMKL LAPACK Backend Build Script" -ForegroundColor Cyan
Write-Host "  Zero-Cost SYCL/DPC++ Abstraction" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Detect Intel oneAPI
# ============================================================================

$oneAPIPath = $null
$possiblePaths = @(
    "${env:ONEAPI_ROOT}",
    "C:\Program Files (x86)\Intel\oneAPI",
    "${env:ProgramFiles(x86)}\Intel\oneAPI",
    "${env:ProgramFiles}\Intel\oneAPI"
)

foreach ($path in $possiblePaths) {
    if ($path -and (Test-Path $path)) {
        $oneAPIPath = $path
        Write-Host "[OK] Intel oneAPI detected: $oneAPIPath" -ForegroundColor Green
        break
    }
}

if (-not $oneAPIPath) {
    Write-Host "[ERROR] Intel oneAPI not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Intel oneAPI Base Toolkit:" -ForegroundColor Yellow
    Write-Host "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After installation, run:" -ForegroundColor Yellow
    Write-Host '  & "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"' -ForegroundColor Cyan
    exit 1
}

# Check for compiler
$compilerPath = "$oneAPIPath\compiler\latest\bin\icpx.exe"
if (-not (Test-Path $compilerPath)) {
    # Try alternate path
    $compilerPath = "$oneAPIPath\compiler\latest\windows\bin\icpx.exe"
}

if (-not (Test-Path $compilerPath)) {
    Write-Host "[ERROR] Intel DPC++ compiler (icpx) not found!" -ForegroundColor Red
    Write-Host "Expected: $compilerPath" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Intel DPC++ compiler: $compilerPath" -ForegroundColor Green

# Check for oneMKL
$mklPath = "$oneAPIPath\mkl\latest"
if (-not (Test-Path $mklPath)) {
    Write-Host "[ERROR] Intel oneMKL not found!" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Intel oneMKL: $mklPath" -ForegroundColor Green

# ============================================================================
# Build Configuration
# ============================================================================

$sourceFile = "src\backends\gpu\onemkl_lapack_impl.cpp"
$includeDir = "include"
$outputLib = "faster-blaster-onemkl-lapack"

if (-not (Test-Path $sourceFile)) {
    Write-Host "[ERROR] Source file not found: $sourceFile" -ForegroundColor Red
    exit 1
}

# Build flags
$cxxflags = @(
    "-fsycl",                    # Enable SYCL compilation
    "-std=c++17",                # C++17 standard
    "-I$includeDir",             # Project headers
    "-I$mklPath\include",        # oneMKL headers
    "-fPIC"                      # Position independent code
)

if ($BuildType -eq "Debug") {
    $cxxflags += "-g", "-O0", "-DDEBUG"
    $outputFile = "${outputLib}_debug.lib"
} else {
    $cxxflags += "-O3", "-DNDEBUG"
    $outputFile = "${outputLib}.lib"
}

# Linker flags
$ldflags = @(
    "-L$mklPath\lib\intel64",
    "-lmkl_sycl",
    "-lmkl_intel_ilp64",
    "-lmkl_tbb_thread",
    "-lmkl_core",
    "-ltbb",
    "-lsycl",
    "-lOpenCL"
)

# ============================================================================
# Compile
# ============================================================================

Write-Host ""
Write-Host "Building oneMKL LAPACK backend..." -ForegroundColor Cyan
Write-Host "  Source: $sourceFile" -ForegroundColor Gray
Write-Host "  Output: $outputFile" -ForegroundColor Gray
Write-Host "  Build type: $BuildType" -ForegroundColor Gray
Write-Host ""

$compileCmd = @($compilerPath) + $cxxflags + @(
    "-c", $sourceFile,
    "-o", "onemkl_lapack_impl.obj"
)

Write-Host "Compiling: $($compileCmd -join ' ')" -ForegroundColor Gray
& $compileCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[OK] Compilation successful" -ForegroundColor Green

# ============================================================================
# Create Static Library
# ============================================================================

Write-Host ""
Write-Host "Creating static library: $outputFile" -ForegroundColor Cyan

lib.exe /OUT:$outputFile onemkl_lapack_impl.obj

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Library creation failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "[OK] Library created successfully" -ForegroundColor Green

# ============================================================================
# Summary
# ============================================================================

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Build Complete!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output: $outputFile" -ForegroundColor Green
Write-Host ""
Write-Host "Architecture:" -ForegroundColor Yellow
Write-Host "  - SYCL/DPC++ based implementation" -ForegroundColor White
Write-Host "  - oneMKL LAPACK operations (26 total)" -ForegroundColor White
Write-Host "  - Direct queue calls = zero-cost abstraction" -ForegroundColor White
Write-Host "  - Scratchpad allocation handled automatically" -ForegroundColor White
Write-Host ""
Write-Host "Supported Hardware:" -ForegroundColor Yellow
Write-Host "  - Intel Arc GPUs (A770, A750, A380)" -ForegroundColor White
Write-Host "  - Intel Data Center GPUs (Flex 170, Max 1550)" -ForegroundColor White
Write-Host "  - Intel Iris Xe (11th gen+)" -ForegroundColor White
Write-Host ""
Write-Host "Multi-Vendor LAPACK Status:" -ForegroundColor Yellow
Write-Host "  ✅ hipSOLVER: NVIDIA + AMD (unified)" -ForegroundColor Green
Write-Host "  ✅ oneMKL:    Intel (this build)" -ForegroundColor Green
Write-Host "  📝 MAGMA:     All vendors (future)" -ForegroundColor Gray
Write-Host ""
Write-Host "Integration:" -ForegroundColor Yellow
Write-Host "  - Combine with onemkl_trait_impl.cpp (BLAS)" -ForegroundColor White
Write-Host "  - Use fb_onemkl_lapack_trait vtable" -ForegroundColor White
Write-Host "  - Same zero-cost pattern as hipSOLVER" -ForegroundColor White
Write-Host ""
