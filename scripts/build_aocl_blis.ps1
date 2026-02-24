#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build AOCL-BLIS from source for maximum performance
.DESCRIPTION
    Downloads and builds AMD AOCL-BLIS library with optimizations for current hardware
.PARAMETER InstallDir
    Installation directory (default: C:\AOCL-BLIS-Custom)
.PARAMETER BuildType
    Build type: Release or Debug (default: Release)
#>

param(
    [string]$InstallDir = "C:\AOCL-BLIS-Custom",
    [string]$BuildType = "Release",
    [string]$AOCLVersion = "4.2"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Building AOCL-BLIS from Source" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Detect CPU architecture
$cpuInfo = Get-WmiObject -Class Win32_Processor | Select-Object -First 1
$cpuName = $cpuInfo.Name
Write-Host "Detected CPU: $cpuName" -ForegroundColor Yellow

# Determine optimal configuration
$blisConfig = "zen3"  # Default for AMD Zen3+
if ($cpuName -match "Zen 4|Ryzen 7[0-9]{3}|Ryzen 9 7[0-9]{3}") {
    $blisConfig = "zen4"
    Write-Host "Using Zen4 optimized configuration" -ForegroundColor Green
} elseif ($cpuName -match "Zen 3|Ryzen [59] 5[0-9]{3}|Ryzen [59] 6[0-9]{3}") {
    $blisConfig = "zen3"
    Write-Host "Using Zen3 optimized configuration" -ForegroundColor Green
} elseif ($cpuName -match "Zen 2|Ryzen [357] 3[0-9]{3}|Ryzen [357] 4[0-9]{3}") {
    $blisConfig = "zen2"
    Write-Host "Using Zen2 optimized configuration" -ForegroundColor Green
} elseif ($cpuName -match "Intel") {
    Write-Warning "Intel CPU detected. AOCL-BLIS is optimized for AMD CPUs."
    Write-Warning "Consider using Intel MKL instead for better Intel performance."
    $blisConfig = "haswell"  # Generic Intel fallback
} else {
    Write-Host "Using generic Zen3 configuration" -ForegroundColor Yellow
}

# Check for required tools
$requiredTools = @("git", "python", "cmake")
foreach ($tool in $requiredTools) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error "$tool is required but not found in PATH. Please install it first."
        exit 1
    }
}

# Create working directory
$workDir = Join-Path $env:TEMP "aocl-blis-build"
if (Test-Path $workDir) {
    Write-Host "Removing existing build directory..." -ForegroundColor Yellow
    Remove-Item -Path $workDir -Recurse -Force
}
New-Item -ItemType Directory -Path $workDir | Out-Null

Push-Location $workDir

try {
    # Clone BLIS repository (AMD fork)
    Write-Host "`nCloning BLIS repository..." -ForegroundColor Cyan
    git clone --depth 1 --branch master https://github.com/amd/blis.git
    
    Set-Location blis
    
    # Configure BLIS with optimal settings
    Write-Host "`nConfiguring BLIS..." -ForegroundColor Cyan
    Write-Host "Configuration: $blisConfig" -ForegroundColor Yellow
    Write-Host "Install directory: $InstallDir" -ForegroundColor Yellow
    
    $configureArgs = @(
        "--prefix=$InstallDir",
        "--enable-cblas",
        "--enable-threading=openmp",
        "--enable-shared",
        "--enable-static",
        $blisConfig
    )
    
    Write-Host "`nRunning: ./configure $($configureArgs -join ' ')" -ForegroundColor Gray
    & python configure @configureArgs
    
    if ($LASTEXITCODE -ne 0) {
        throw "Configuration failed with exit code $LASTEXITCODE"
    }
    
    # Build with maximum parallelism
    Write-Host "`nBuilding BLIS..." -ForegroundColor Cyan
    $numCores = (Get-WmiObject -Class Win32_ComputerSystem).NumberOfLogicalProcessors
    Write-Host "Using $numCores parallel jobs" -ForegroundColor Yellow
    
    & make -j $numCores
    
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
    
    # Install
    Write-Host "`nInstalling BLIS to $InstallDir..." -ForegroundColor Cyan
    & make install
    
    if ($LASTEXITCODE -ne 0) {
        throw "Installation failed with exit code $LASTEXITCODE"
    }
    
    Write-Host "`n============================================" -ForegroundColor Green
    Write-Host " AOCL-BLIS Build Successful!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Installation location: $InstallDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Add to system PATH: $InstallDir\bin" -ForegroundColor White
    Write-Host "2. Set BLIS_NUM_THREADS environment variable (optional)" -ForegroundColor White
    Write-Host "3. Rebuild faster-blaster to link against custom BLIS" -ForegroundColor White
    Write-Host ""
    
    # Show library info
    $libDir = Join-Path $InstallDir "lib"
    if (Test-Path $libDir) {
        Write-Host "Built libraries:" -ForegroundColor Cyan
        Get-ChildItem -Path $libDir -Filter "*.dll" | ForEach-Object {
            Write-Host "  - $($_.Name)" -ForegroundColor Gray
        }
    }
    
} catch {
    Write-Error "Build failed: $_"
    exit 1
} finally {
    Pop-Location
}

Write-Host "`nBuild directory preserved at: $workDir" -ForegroundColor Gray
Write-Host "You can safely delete it after testing the installation." -ForegroundColor Gray
