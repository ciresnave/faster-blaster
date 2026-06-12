# Build script for cuBLAS backend tests on Windows
# Requires: CUDA Toolkit 11.0+ installed

param(
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building cuBLAS Backend Tests" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if CUDA is installed
if (-not (Test-Path $CudaPath)) {
    Write-Host "ERROR: CUDA not found at $CudaPath" -ForegroundColor Red
    Write-Host "Please install CUDA Toolkit or specify correct path with -CudaPath parameter" -ForegroundColor Yellow
    exit 1
}

$nvcc = Join-Path $CudaPath "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) {
    Write-Host "ERROR: nvcc.exe not found at $nvcc" -ForegroundColor Red
    exit 1
}

Write-Host "Using CUDA at: $CudaPath" -ForegroundColor Green
Write-Host "NVCC: $nvcc" -ForegroundColor Green

# Set paths
$projectRoot = Split-Path -Parent $PSScriptRoot
$includeDir = Join-Path $projectRoot "include"
$srcDir = Join-Path $projectRoot "src\backends\gpu"
$testDir = $PSScriptRoot
$buildDir = Join-Path $projectRoot "build\tests"

# Create build directory
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Write-Host "`nCompiling cuBLAS trait implementation..." -ForegroundColor Yellow

# Compile cublas_trait_impl.c
$traitSrc = Join-Path $srcDir "cublas_trait_impl.c"
$traitObj = Join-Path $buildDir "cublas_trait_impl.obj"

$compileArgs = @(
    "-c",
    "-I", $includeDir,
    "-I", (Join-Path $CudaPath "include"),
    $traitSrc,
    "-o", $traitObj
)

Write-Host "Command: $nvcc $($compileArgs -join ' ')" -ForegroundColor DarkGray

& $nvcc @compileArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to compile cublas_trait_impl.c" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Compiled cublas_trait_impl.c" -ForegroundColor Green

Write-Host "`nCompiling test_cublas_basic.c..." -ForegroundColor Yellow

# Compile test_cublas_basic.c
$testSrc = Join-Path $testDir "test_cublas_basic.c"
$testObj = Join-Path $buildDir "test_cublas_basic.obj"

$compileArgs = @(
    "-c",
    "-I", $includeDir,
    "-I", (Join-Path $CudaPath "include"),
    $testSrc,
    "-o", $testObj
)

Write-Host "Command: $nvcc $($compileArgs -join ' ')" -ForegroundColor DarkGray

& $nvcc @compileArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to compile test_cublas_basic.c" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Compiled test_cublas_basic.c" -ForegroundColor Green

Write-Host "`nLinking executable..." -ForegroundColor Yellow

# Link executable
$exePath = Join-Path $buildDir "test_cublas_basic.exe"

$linkArgs = @(
    $traitObj,
    $testObj,
    "-lcublas",
    "-lcusolver",
    "-o", $exePath
)

Write-Host "Command: $nvcc $($linkArgs -join ' ')" -ForegroundColor DarkGray

& $nvcc @linkArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to link executable" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Linked test_cublas_basic.exe" -ForegroundColor Green

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Executable: $exePath" -ForegroundColor Cyan
Write-Host "`nTo run tests:" -ForegroundColor Yellow
Write-Host "  cd $buildDir" -ForegroundColor White
Write-Host "  .\test_cublas_basic.exe" -ForegroundColor White
Write-Host ""
