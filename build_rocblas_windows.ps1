param(
    [string]$RocmPath = "C:\Program Files\AMD\ROCm\6.4"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building rocBLAS Backend (Windows HIP SDK)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Using ROCm at: $RocmPath"
Write-Host ""

# Verify HIP SDK installation
$hipcc = "$RocmPath\bin\hipcc.exe"
if (-not (Test-Path $hipcc)) {
    Write-Host "ERROR: hipcc not found at $hipcc" -ForegroundColor Red
    Write-Host ""
    Write-Host "Searching for HIP SDK installation..."
    $found = $false
    $searchPaths = @(
        "C:\Program Files\AMD\ROCm",
        "C:\AMD\ROCm",
        "C:\Program Files\AMD"
    )
    
    foreach ($searchPath in $searchPaths) {
        if (Test-Path $searchPath) {
            $hipccFiles = Get-ChildItem $searchPath -Recurse -Filter "hipcc.exe" -ErrorAction SilentlyContinue
            if ($hipccFiles) {
                Write-Host "Found hipcc at:" -ForegroundColor Yellow
                $hipccFiles | ForEach-Object { Write-Host "  $($_.FullName)" -ForegroundColor Green }
                $found = $true
            }
        }
    }
    
    if (-not $found) {
        Write-Host ""
        Write-Host "HIP SDK not installed. Please install from:" -ForegroundColor Red
        Write-Host "  https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"
    }
    exit 1
}

Write-Host "✓ Found hipcc: $hipcc" -ForegroundColor Green

# Check HIP version
Write-Host ""
Write-Host "HIP SDK Version:" -ForegroundColor Yellow
& $hipcc --version

# Verify rocBLAS library
Write-Host ""
Write-Host "Checking for rocBLAS library..." -ForegroundColor Yellow
$libDir = "$RocmPath\lib"
$rocblasFiles = @(
    "$libDir\rocblas.lib",
    "$libDir\rocblas64.lib",
    "$libDir\rocblas.dll.lib"
)

$rocblasFound = $false
foreach ($lib in $rocblasFiles) {
    if (Test-Path $lib) {
        Write-Host "✓ Found rocBLAS: $lib" -ForegroundColor Green
        $rocblasFound = $true
        break
    }
}

if (-not $rocblasFound) {
    Write-Host "WARNING: rocBLAS library not found in expected locations" -ForegroundColor Yellow
    Write-Host "Searching for rocBLAS files..."
    Get-ChildItem $libDir -Filter "*rocblas*" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  Found: $($_.FullName)" -ForegroundColor Cyan
    }
}

# Verify rocSOLVER library
$rocsolverFiles = @(
    "$libDir\rocsolver.lib",
    "$libDir\rocsolver64.lib",
    "$libDir\rocsolver.dll.lib"
)

$rocsolverFound = $false
foreach ($lib in $rocsolverFiles) {
    if (Test-Path $lib) {
        Write-Host "✓ Found rocSOLVER: $lib" -ForegroundColor Green
        $rocsolverFound = $true
        break
    }
}

if (-not $rocsolverFound) {
    Write-Host "WARNING: rocSOLVER library not found" -ForegroundColor Yellow
}

# Create build directory
Write-Host ""
Write-Host "Creating build directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "build\tests" | Out-Null
Write-Host "✓ Created build\tests" -ForegroundColor Green

# Compile rocBLAS trait implementation
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Compiling rocBLAS Trait Implementation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$includeArgs = @(
    "-I", "include",
    "-I", "`"$RocmPath\include`""
)

$compileArgs = @(
    "-c",
    "-D__HIP_PLATFORM_AMD__",
    "-D_CRT_SECURE_NO_WARNINGS",
    $includeArgs,
    "src\backends\gpu\rocblas_trait_impl.c",
    "-o", "build\tests\rocblas_trait_impl.obj"
)

Write-Host "Command: $hipcc $($compileArgs -join ' ')" -ForegroundColor Cyan

& $hipcc $compileArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to compile rocblas_trait_impl.c" -ForegroundColor Red
    Write-Host "Exit code: $LASTEXITCODE"
    exit 1
}

Write-Host ""
Write-Host "✓ Compiled rocblas_trait_impl.c" -ForegroundColor Green

# Compile test file
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Compiling Test File" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$compileArgs = @(
    "-c",
    "-D__HIP_PLATFORM_AMD__",
    "-D_CRT_SECURE_NO_WARNINGS",
    $includeArgs,
    "tests\test_rocblas_basic.c",
    "-o", "build\tests\test_rocblas_basic.obj"
)

Write-Host "Command: $hipcc $($compileArgs -join ' ')" -ForegroundColor Cyan

& $hipcc $compileArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to compile test_rocblas_basic.c" -ForegroundColor Red
    Write-Host "Exit code: $LASTEXITCODE"
    exit 1
}

Write-Host ""
Write-Host "✓ Compiled test_rocblas_basic.c" -ForegroundColor Green

# Link executable
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Linking Executable" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$linkArgs = @(
    "build\tests\rocblas_trait_impl.obj",
    "build\tests\test_rocblas_basic.obj",
    "-L`"$libDir`"",
    "-lrocblas",
    "-lrocsolver",
    "-o", "build\tests\test_rocblas_basic.exe"
)

Write-Host "Command: $hipcc $($linkArgs -join ' ')" -ForegroundColor Cyan

& $hipcc $linkArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to link executable" -ForegroundColor Red
    Write-Host "Exit code: $LASTEXITCODE"
    Write-Host ""
    Write-Host "Note: Windows linking may require different syntax." -ForegroundColor Yellow
    Write-Host "Try checking library names in: $libDir"
    exit 1
}

Write-Host ""
Write-Host "✓ Linked test_rocblas_basic.exe" -ForegroundColor Green

# Check if DLLs need to be copied
Write-Host ""
Write-Host "Checking runtime dependencies..." -ForegroundColor Yellow
$binDir = "$RocmPath\bin"
$dllsToCopy = @(
    "rocblas64.dll",
    "rocsolver64.dll",
    "amdhip64.dll"
)

foreach ($dll in $dllsToCopy) {
    $srcDll = "$binDir\$dll"
    $dstDll = "build\tests\$dll"
    
    if (Test-Path $srcDll) {
        if (-not (Test-Path $dstDll)) {
            Write-Host "Copying $dll to build\tests\" -ForegroundColor Cyan
            Copy-Item $srcDll $dstDll -ErrorAction SilentlyContinue
        }
    }
}

# Success summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Build Completed Successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Executable: build\tests\test_rocblas_basic.exe" -ForegroundColor White
Write-Host ""
Write-Host "To run tests:" -ForegroundColor Yellow
Write-Host "  cd build\tests" -ForegroundColor White
Write-Host "  .\test_rocblas_basic.exe" -ForegroundColor White
Write-Host ""
Write-Host "Note: Ensure your AMD RX 7900 XTX is connected and drivers are installed." -ForegroundColor Cyan
Write-Host ""
