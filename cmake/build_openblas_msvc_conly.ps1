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
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Building OpenBLAS with Clang-cl (C99 VLA support)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Source:  $SourceDir"
Write-Host "Install: $InstallDir"
Write-Host "Target:  $Target"
Write-Host "Cores:   $NumCores"
Write-Host "Compiler: clang-cl (Clang with MSVC ABI compatibility)"
Write-Host "Script: build_openblas_msvc_conly.ps1 v2026-06-09c"
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

    # ---- Ensure Windows SDK + MSVC libs/includes are set ----
    # When running outside a VS Developer Command Prompt, LIB and INCLUDE may
    # not contain both Windows SDK and MSVC CRT paths. Source vcvars64.bat
    # when either side is missing.
    $libEnv = if ($env:LIB) { $env:LIB } else { "" }
    $hasWindowsSdkLibs = $libEnv -match 'Windows Kits|WindowsKits'
    $hasMsvcLibs = $libEnv -match '\\MSVC\\[^\\]+\\lib\\x64'

    if (-not $hasWindowsSdkLibs -or -not $hasMsvcLibs) {
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswhere) {
            $vsInstallerDir = Split-Path -Parent $vswhere
            if ($vsInstallerDir -and -not (($env:Path -split ';') -contains $vsInstallerDir)) {
                $env:Path = "$vsInstallerDir;$env:Path"
            }
            $vsPath = & $vswhere -latest -property installationPath 2>$null
            $vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
            $launchVsDevShell = "$vsPath\Common7\Tools\Launch-VsDevShell.ps1"
            if (Test-Path $launchVsDevShell) {
                Write-Host "Initializing VS environment from: $launchVsDevShell" -ForegroundColor Yellow
                try {
                    & $launchVsDevShell -Arch amd64 -HostArch amd64 | Out-Null
                    Write-Host "VS environment initialized via Launch-VsDevShell." -ForegroundColor Green
                } catch {
                    Write-Host "Warning: Launch-VsDevShell failed. Continuing with current environment." -ForegroundColor Yellow
                }
                $env:LIB = "$llvmLibPath;$env:LIB"
            }
        }
    }

    # Let CMake's FindOpenMP locate the runtime by hinting the OpenMP library path
    $openmpLib = Join-Path $llvmLibPath "libomp.lib"

    function Invoke-OpenBlasConfigure {
        param(
            [string]$ConfigTarget,
            [string]$DynamicArch,
            [string]$NoAvx512
        )

        if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
            $PSNativeCommandUseErrorActionPreference = $false
        }

        cmake -G "Ninja" `
            -DCMAKE_C_COMPILER=clang-cl `
            -DCMAKE_CXX_COMPILER=clang-cl `
            -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY `
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
            -DCMAKE_BUILD_TYPE=Release `
            -DCMAKE_INSTALL_PREFIX="$InstallDir" `
            "-DOpenMP_C_LIBRARY=`"$openmpLib`"" `
            "-DOpenMP_omp_LIBRARY=`"$openmpLib`"" `
            -DTARGET="$ConfigTarget" `
            -DUSE_OPENMP=ON `
            -DUSE_THREAD=ON `
            -DNUM_THREADS=128 `
            "-DDYNAMIC_ARCH=$DynamicArch" `
            -DBUILD_SHARED_LIBS=OFF `
            -DBUILD_STATIC_LIBS=ON `
            -DNO_LAPACKE=OFF `
            -DNOFORTRAN=ON `
            "-DNO_AVX512=$NoAvx512" `
            -DBUILD_TESTING=OFF `
            "$SourceDir"

        return $LASTEXITCODE
    }

    $configureExit = Invoke-OpenBlasConfigure -ConfigTarget $Target -DynamicArch "OFF" -NoAvx512 "OFF"
    if ($configureExit -ne 0) {
        Write-Host "Initial OpenBLAS configure failed for TARGET=$Target. Retrying with GENERIC + DYNAMIC_ARCH=ON..." -ForegroundColor Yellow
        $configureExit = Invoke-OpenBlasConfigure -ConfigTarget "GENERIC" -DynamicArch "ON" -NoAvx512 "ON"
    }

    if ($configureExit -ne 0) {
        Write-Host "OpenBLAS CMake path failed. Trying upstream make-based build fallback..." -ForegroundColor Yellow

        $makeCmd = Get-Command make -ErrorAction SilentlyContinue
        if (-not $makeCmd) {
            $makeCmd = Get-Command mingw32-make -ErrorAction SilentlyContinue
        }
        if (-not $makeCmd) {
            $makeCmd = Get-Command gmake -ErrorAction SilentlyContinue
        }
        if (-not $makeCmd) {
            throw "CMake configuration failed and no GNU make executable (make/mingw32-make/gmake) is available for fallback build."
        }

        Push-Location $SourceDir
        try {
            & $makeCmd.Source "BINARY=64" "CC=clang" "FC=" "TARGET=$Target" "DYNAMIC_ARCH=1" "USE_OPENMP=1" "USE_THREAD=1" "NOFORTRAN=1" "NO_SHARED=1" "NO_STATIC=0" "-j$NumCores"
            if ($LASTEXITCODE -ne 0) {
                throw "OpenBLAS make build failed."
            }

            & $makeCmd.Source "PREFIX=$InstallDir" install
            if ($LASTEXITCODE -ne 0) {
                throw "OpenBLAS make install failed."
            }
        }
        finally {
            Pop-Location
        }

        Write-Host "OpenBLAS make-based fallback build succeeded." -ForegroundColor Green
    }
    
    if ($configureExit -eq 0) {
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
