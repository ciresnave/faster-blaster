# Build OpenBLAS from source using MinGW-w64 cross-compiler in WSL
# Produces Windows PE/COFF binaries compatible with MSVC
# MSVC doesn't support variable-length arrays needed by OpenBLAS

param(
    [string]$SourceDir = "C:\Users\cires\OneDrive\Documents\projects\faster-blaster\build\openblas-backend-prefix\src\openblas-backend",
    [string]$InstallDir = "C:\Users\cires\OneDrive\Documents\projects\faster-blaster\build\backends-install\openblas",
    [string]$Target = "ZEN",
    [int]$NumCores = 32
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Building OpenBLAS from source (WSL)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Source:  $SourceDir"
Write-Host "Install: $InstallDir"
Write-Host "Target:  $Target"
Write-Host "Cores:   $NumCores"
Write-Host ""

# Convert Windows paths to WSL format
function ConvertTo-WSLPath {
    param([string]$WinPath)
    # Normalize to backslashes first
    $WinPath = $WinPath -replace '/', '\'
    if ($WinPath -match "^([A-Z]):\\(.*)") {
        $drive = $matches[1].ToLower()
        $path = $matches[2] -replace '\\', '/'
        return "/mnt/$drive/$path"
    }
    return $WinPath
}

$wslSource = ConvertTo-WSLPath $SourceDir
$wslInstall = ConvertTo-WSLPath $InstallDir

Write-Host "WSL paths:"
Write-Host "  Source:  $wslSource"
Write-Host "  Install: $wslInstall"
Write-Host ""

# Check WSL availability
Write-Host "Checking WSL..." -ForegroundColor Yellow
$wslCheck = wsl echo "OK" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: WSL is not available!" -ForegroundColor Red
    exit 1
}
Write-Host "WSL is available" -ForegroundColor Green
Write-Host ""

# Build with make in WSL using dedicated bash script
Write-Host "Building OpenBLAS with make (this will take 10-15 minutes)..." -ForegroundColor Green

# Call the bash script to build in WSL's native /tmp filesystem
$wslScript = ConvertTo-WSLPath "$PSScriptRoot\build_openblas_wsl.sh"
wsl bash "$wslScript" "$wslSource" "$wslInstall" "$Target" "$NumCores"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  OpenBLAS build complete!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Installed to: $InstallDir"
Write-Host ""

# Verify installation
if (Test-Path "$InstallDir\lib\openblas.lib") {
    $fileSize = (Get-Item "$InstallDir\lib\openblas.lib").Length / 1MB
    $fileSizeRounded = [math]::Round($fileSize, 2)
    Write-Host "[OK] Library file: openblas.lib ($fileSizeRounded MB)" -ForegroundColor Green
} else {
    Write-Host "[WARN] openblas.lib not found!" -ForegroundColor Yellow
}

if (Test-Path "$InstallDir\include\openblas_config.h") {
    Write-Host "[OK] Headers installed" -ForegroundColor Green
} else {
    Write-Host "[WARN] Headers not found!" -ForegroundColor Yellow
}

exit 0
