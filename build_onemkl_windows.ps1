#!/usr/bin/env pwsh
# Build script for Intel oneMKL GPU backend on Windows
# Requires: Intel oneAPI Base Toolkit 2024.0+

param(
    [switch]$Clean,
    [switch]$Verbose,
    [switch]$Test
)

$ErrorActionPreference = "Stop"

# Colors for output
$SuccessColor = "Green"
$ErrorColor = "Red"
$InfoColor = "Cyan"
$WarningColor = "Yellow"

function Write-Info { param($Message) Write-Host $Message -ForegroundColor $InfoColor }
function Write-Success { param($Message) Write-Host $Message -ForegroundColor $SuccessColor }
function Write-Fail { param($Message) Write-Host $Message -ForegroundColor $ErrorColor }
function Write-Warn { param($Message) Write-Host $Message -ForegroundColor $WarningColor }

Write-Info "======================================"
Write-Info "  Intel oneMKL Backend Builder"
Write-Info "======================================"
Write-Info ""

# Project structure
$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build"
$ObjDir = Join-Path $BuildDir "obj"
$BinDir = Join-Path $BuildDir "bin"
$IncludeDir = Join-Path $ProjectRoot "include"
$SrcFile = Join-Path $ProjectRoot "src\backends\gpu\onemkl_trait_impl.cpp"
$ObjFile = Join-Path $ObjDir "onemkl_trait_impl.obj"

# Create build directories
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Info "Cleaning build directory..."
    Remove-Item $BuildDir -Recurse -Force
}

@($BuildDir, $ObjDir, $BinDir) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
        Write-Info "Created directory: $_"
    }
}

# Detect Intel oneAPI installation
Write-Info "Detecting Intel oneAPI installation..."

$OneAPIRoot = $null
$PossiblePaths = @(
    "C:\Program Files (x86)\Intel\oneAPI",
    "C:\Program Files\Intel\oneAPI",
    "$env:ONEAPI_ROOT"
)

foreach ($Path in $PossiblePaths) {
    if ($Path -and (Test-Path $Path)) {
        $OneAPIRoot = $Path
        Write-Success "Found oneAPI at: $OneAPIRoot"
        break
    }
}

if (-not $OneAPIRoot) {
    Write-Fail "ERROR: Intel oneAPI Base Toolkit not found!"
    Write-Info ""
    Write-Info "Please install Intel oneAPI Base Toolkit from:"
    Write-Info "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
    Write-Info ""
    Write-Info "Installation includes:"
    Write-Info "  - Intel DPC++ Compiler (icpx)"
    Write-Info "  - oneMKL library (BLAS/LAPACK)"
    Write-Info "  - SYCL runtime"
    Write-Info "  - Support for Intel GPUs (Arc, Flex, Max, Iris Xe)"
    Write-Info ""
    exit 1
}

# Find compiler version
$CompilerRoot = Join-Path $OneAPIRoot "compiler\latest"
if (-not (Test-Path $CompilerRoot)) {
    Write-Fail "ERROR: Compiler not found at $CompilerRoot"
    exit 1
}

$CompilerBin = Join-Path $CompilerRoot "bin"
$IcpxExe = Join-Path $CompilerBin "icpx.exe"

if (-not (Test-Path $IcpxExe)) {
    Write-Fail "ERROR: icpx.exe not found at $IcpxExe"
    exit 1
}

Write-Success "Found Intel C++ compiler: $IcpxExe"

# Check compiler version
$CompilerVersion = & $IcpxExe --version 2>&1 | Select-Object -First 1
Write-Info "Compiler: $CompilerVersion"

# Find oneMKL
$MKLRoot = Join-Path $OneAPIRoot "mkl\latest"
if (-not (Test-Path $MKLRoot)) {
    Write-Fail "ERROR: oneMKL not found at $MKLRoot"
    exit 1
}

$MKLInclude = Join-Path $MKLRoot "include"
$MKLLib = Join-Path $MKLRoot "lib"

Write-Success "Found oneMKL at: $MKLRoot"

# Verify critical files
$RequiredIncludes = @(
    (Join-Path $MKLInclude "oneapi\mkl.hpp"),
    (Join-Path $MKLInclude "oneapi\mkl\blas.hpp")
)

$MissingIncludes = $RequiredIncludes | Where-Object { -not (Test-Path $_) }
if ($MissingIncludes) {
    Write-Fail "ERROR: Missing oneMKL include files:"
    $MissingIncludes | ForEach-Object { Write-Fail "  - $_" }
    exit 1
}

# Find SYCL runtime
$SYCLInclude = Join-Path $CompilerRoot "include\sycl"
if (-not (Test-Path $SYCLInclude)) {
    Write-Fail "ERROR: SYCL headers not found at $SYCLInclude"
    exit 1
}

Write-Success "Found SYCL runtime: $SYCLInclude"

# Detect Intel GPUs
Write-Info ""
Write-Info "Detecting Intel GPU devices..."

try {
    # Use sycl-ls if available
    $SyclLs = Join-Path $CompilerBin "sycl-ls.exe"
    if (Test-Path $SyclLs) {
        $Devices = & $SyclLs 2>&1
        Write-Info "Available SYCL devices:"
        $Devices | ForEach-Object { Write-Info "  $_" }
    } else {
        Write-Warn "sycl-ls not found, skipping device detection"
    }
} catch {
    Write-Warn "Could not enumerate SYCL devices: $_"
}

# Build flags
$CompileFlags = @(
    # SYCL compilation
    "-fsycl",
    
    # C++ standard
    "-std=c++17",
    
    # Include paths
    "-I$IncludeDir",
    "-I$MKLInclude",
    "-I$SYCLInclude",
    
    # Preprocessor defines
    "-DMKL_ILP64",              # Use 64-bit integers in oneMKL
    "-D_CRT_SECURE_NO_WARNINGS", # Suppress Windows CRT warnings
    
    # Optimization
    "-O2",
    
    # Warnings
    "-Wall",
    "-Wno-deprecated-declarations",
    
    # Output
    "-c",
    "-o", $ObjFile,
    $SrcFile
)

if ($Verbose) {
    $CompileFlags += "-v"
}

# Compile
Write-Info ""
Write-Info "Compiling oneMKL backend..."
Write-Info "Source: $SrcFile"
Write-Info "Output: $ObjFile"

if ($Verbose) {
    Write-Info ""
    Write-Info "Compile command:"
    Write-Info "  $IcpxExe $($CompileFlags -join ' ')"
}

Write-Info ""
try {
    & $IcpxExe @CompileFlags
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✓ oneMKL backend compiled successfully!"
        
        if (Test-Path $ObjFile) {
            $ObjSize = (Get-Item $ObjFile).Length / 1KB
            Write-Success "  Object file: $ObjFile ($([math]::Round($ObjSize, 2)) KB)"
        }
    } else {
        Write-Fail "✗ Compilation failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
} catch {
    Write-Fail "✗ Compilation error: $_"
    exit 1
}

# Optional: Build test program
if ($Test) {
    Write-Info ""
    Write-Info "Building test program..."
    
    $TestSrc = Join-Path $ProjectRoot "test_onemkl_basic.cpp"
    $TestExe = Join-Path $BinDir "test_onemkl_basic.exe"
    
    if (-not (Test-Path $TestSrc)) {
        Write-Warn "Test source not found: $TestSrc"
        Write-Info "Skipping test build"
    } else {
        # Link flags for test executable
        $LinkFlags = @(
            "-fsycl",
            "-I$IncludeDir",
            "-I$MKLInclude",
            "-L$MKLLib",
            "-lmkl_sycl",
            "-lmkl_intel_ilp64",
            "-lmkl_tbb_thread",
            "-lmkl_core",
            "-ltbb",
            "-o", $TestExe,
            $TestSrc,
            $ObjFile
        )
        
        Write-Info "Linking test executable..."
        try {
            & $IcpxExe @LinkFlags
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "✓ Test executable built: $TestExe"
                
                # Run test
                Write-Info ""
                Write-Info "Running tests..."
                & $TestExe
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "✓ All tests passed!"
                } else {
                    Write-Fail "✗ Tests failed with exit code $LASTEXITCODE"
                }
            } else {
                Write-Fail "✗ Linking failed with exit code $LASTEXITCODE"
            }
        } catch {
            Write-Fail "✗ Linking error: $_"
        }
    }
}

# Summary
Write-Info ""
Write-Info "======================================"
Write-Info "  Build Summary"
Write-Info "======================================"
Write-Success "Backend: Intel oneMKL (SYCL)"
Write-Success "Compiler: Intel DPC++ (icpx)"
Write-Success "Status: Compilable (hardware not required)"
Write-Info ""
Write-Info "Next steps:"
Write-Info "  1. Implement remaining BLAS operations (220+ total)"
Write-Info "  2. Test on Intel Arc/Iris Xe GPU"
Write-Info "  3. Benchmark against cuBLAS/rocBLAS"
Write-Info ""
Write-Info "Note: oneMKL has MORE operations than cuBLAS/rocBLAS!"
Write-Info "  - Includes cspmv, zspmv, cspr, zspr, cspr2, zspr2"
Write-Info "  - Total: 220+ BLAS operations vs 198 in NVIDIA/AMD"
Write-Info ""
