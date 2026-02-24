# Build OpenBLAS from source using Clang-cl (C99/VLA support with MSVC ABI)
# Standard MSVC (cl.exe) lacks C99 VLA support needed by OpenBLAS
# clang-cl provides C99 support while maintaining Windows ABI compatibility

param(
    [string]$SourceDir,
    [string]$InstallDir,
    [string]$Target = "ZEN",
    [int]$NumCores = 32
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Building OpenBLAS with Clang-cl (C99 VLA support)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Source:  $SourceDir"
Write-Host "Install: $InstallDir"
Write-Host "Target:  $Target"
Write-Host "Cores:   $NumCores"
Write-Host "Compiler: clang-cl (Clang with MSVC ABI compatibility)"
Write-Host ""

# Create build directory
$BuildDir = Join-Path $env:TEMP "openblas-msvc-conly-build"
if (Test-Path $BuildDir) {
    Write-Host "Cleaning old build directory..."
    Remove-Item -Recurse -Force $BuildDir
}
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null

Push-Location $BuildDir

try {
    # Verify clang-cl is available
    $clangCl = Get-Command clang-cl -ErrorAction SilentlyContinue
    if (-not $clangCl) {
        throw "clang-cl not found! Please install LLVM or 'C++ Clang tools for Windows' via Visual Studio Installer."
    }
    
    Write-Host "Found clang-cl: $($clangCl.Source)" -ForegroundColor Green
    
    Write-Host "Configuring with CMake..." -ForegroundColor Yellow
    
    # Find LLVM's OpenMP library (libomp)
    $llvmPath = Split-Path -Parent $clangCl.Source
    $llvmRoot = Split-Path -Parent $llvmPath
    $llvmLibPath = Join-Path $llvmRoot "lib"
    
    Write-Host "LLVM Root: $llvmRoot" -ForegroundColor Cyan
    Write-Host "LLVM Lib:  $llvmLibPath" -ForegroundColor Cyan
    
    # Configure with CMake using clang-cl with OpenMP support
    # clang-cl provides C99/VLA support while maintaining MSVC ABI compatibility
    # DYNAMIC_ARCH=OFF: Only build for target architecture (ZEN), not all CPUs
    # USE_OPENMP=ON: Enable OpenMP for multithreading (clang has OpenMP 5.0+ support)
    
    # Add LLVM lib path to LIB environment variable for libomp.lib
    $env:LIB = "$llvmLibPath;$env:LIB"

    # Let CMake's FindOpenMP locate the runtime by hinting the OpenMP library path
    $openmpLib = Join-Path $llvmLibPath "libomp.lib"

    cmake -G "Ninja" `
        -DCMAKE_C_COMPILER=clang-cl `
        -DCMAKE_CXX_COMPILER=clang-cl `
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_INSTALL_PREFIX="$InstallDir" `
        "-DOpenMP_C_LIBRARY=`"$openmpLib`"" `
        "-DOpenMP_omp_LIBRARY=`"$openmpLib`"" `
        -DTARGET="$Target" `
        -DUSE_OPENMP=ON `
        -DUSE_THREAD=ON `
        -DNUM_THREADS=128 `
        -DDYNAMIC_ARCH=OFF `
        -DBUILD_SHARED_LIBS=ON `
        -DBUILD_STATIC_LIBS=OFF `
        -DNO_LAPACKE=OFF `
        -DNOFORTRAN=ON `
        -DNO_AVX512=OFF `
        -DBUILD_TESTING=OFF `
        "$SourceDir"
    
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed!"
    }
    
    Write-Host ""
    Write-Host "Building OpenBLAS (this will take 10-15 minutes)..." -ForegroundColor Yellow
    cmake --build . --config Release --parallel $NumCores
    
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed!"
    }
    
    Write-Host ""
    Write-Host "Installing to $InstallDir..." -ForegroundColor Yellow
    cmake --install . --config Release
    
    if ($LASTEXITCODE -ne 0) {
        throw "Installation failed!"
    }
    
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  OpenBLAS built successfully!" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "Installed to: $InstallDir"
    
    # Verify installation
    if (Test-Path "$InstallDir\bin\openblas.dll") {
        Write-Host "Shared library: openblas.dll" -ForegroundColor Green
    }
    if (Test-Path "$InstallDir\lib\openblas.lib") {
        Write-Host "Import library: openblas.lib" -ForegroundColor Green
    }
    
} catch {
    Write-Host ""
    Write-Host "ERROR: $_" -ForegroundColor Red
    Write-Host "Build directory preserved for debugging: $BuildDir" -ForegroundColor Yellow
    Pop-Location
    exit 1
} finally {
    if ($LASTEXITCODE -eq 0) {
        Pop-Location
        # Only cleanup on success
        if (Test-Path $BuildDir) {
            Write-Host "Cleaning up build directory..."
            Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue
        }
    }
}
