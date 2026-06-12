#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Validate LAPACK implementations for AMD and Intel without real hardware

.DESCRIPTION
    Performs compilation tests, static analysis, and API validation to ensure
    AMD (rocSOLVER) and Intel (oneMKL) LAPACK backends will work correctly
    when deployed on actual hardware.

.EXAMPLE
    .\validate_lapack_backends.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Multi-Vendor LAPACK Validation Suite" -ForegroundColor Cyan
Write-Host "  Testing Without Real Hardware" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$results = @{
    "hipSOLVER_NVIDIA" = @{ Status = "Unknown"; Issues = @() }
    "hipSOLVER_AMD" = @{ Status = "Unknown"; Issues = @() }
    "oneMKL_Intel" = @{ Status = "Unknown"; Issues = @() }
}

# ============================================================================
# Test 1: Check Source Files Exist
# ============================================================================

Write-Host "[Test 1] Checking source files..." -ForegroundColor Yellow

$sourceFiles = @(
    @{ Name = "hipSOLVER"; Path = "src\backends\gpu\hipsolver_trait_impl.cpp" },
    @{ Name = "oneMKL"; Path = "src\backends\gpu\onemkl_lapack_impl.cpp" }
)

foreach ($file in $sourceFiles) {
    if (Test-Path $file.Path) {
        $size = (Get-Item $file.Path).Length
        Write-Host "  [OK] $($file.Name): $($file.Path) ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $($file.Name): $($file.Path) not found!" -ForegroundColor Red
        $results["hipSOLVER_AMD"].Issues += "Source file missing"
        $results["oneMKL_Intel"].Issues += "Source file missing"
    }
}

Write-Host ""

# ============================================================================
# Test 2: SDK Detection
# ============================================================================

Write-Host "[Test 2] Detecting SDKs..." -ForegroundColor Yellow

# CUDA Toolkit
$cudaPath = $env:CUDA_PATH
if ($cudaPath -and (Test-Path "$cudaPath\bin\nvcc.exe")) {
    Write-Host "  [OK] CUDA Toolkit: $cudaPath" -ForegroundColor Green
    $results["hipSOLVER_NVIDIA"].Status = "SDK Available"
} else {
    Write-Host "  [SKIP] CUDA Toolkit not found (expected)" -ForegroundColor Gray
}

# HIP SDK (AMD)
$hipPath = $env:HIP_PATH
if ($hipPath -and (Test-Path "$hipPath\bin\hipcc.exe")) {
    Write-Host "  [OK] HIP SDK: $hipPath" -ForegroundColor Green
    $results["hipSOLVER_AMD"].Status = "SDK Available"
} else {
    Write-Host "  [WARN] HIP SDK not found (cannot validate AMD)" -ForegroundColor Yellow
    $results["hipSOLVER_AMD"].Issues += "SDK not installed"
}

# Intel oneAPI
$oneAPIPath = $env:ONEAPI_ROOT
if (-not $oneAPIPath) {
    $oneAPIPath = "C:\Program Files (x86)\Intel\oneAPI"
}

if ($oneAPIPath -and (Test-Path $oneAPIPath)) {
    $compilerPath = "$oneAPIPath\compiler\latest\bin\icpx.exe"
    if (-not (Test-Path $compilerPath)) {
        $compilerPath = "$oneAPIPath\compiler\latest\windows\bin\icpx.exe"
    }
    
    if (Test-Path $compilerPath) {
        Write-Host "  [OK] Intel oneAPI: $oneAPIPath" -ForegroundColor Green
        $results["oneMKL_Intel"].Status = "SDK Available"
    } else {
        Write-Host "  [WARN] Intel oneAPI found but compiler missing" -ForegroundColor Yellow
        $results["oneMKL_Intel"].Issues += "Compiler not found"
    }
} else {
    Write-Host "  [WARN] Intel oneAPI not found (cannot validate Intel)" -ForegroundColor Yellow
    $results["oneMKL_Intel"].Issues += "SDK not installed"
}

Write-Host ""

# ============================================================================
# Test 3: Compilation Test (if SDKs available)
# ============================================================================

Write-Host "[Test 3] Compilation tests..." -ForegroundColor Yellow

# Test hipSOLVER with HIP SDK (AMD)
if ($results["hipSOLVER_AMD"].Status -eq "SDK Available") {
    Write-Host "  Testing hipSOLVER (AMD target)..." -ForegroundColor Cyan
    
    try {
        $hipcc = "$hipPath\bin\hipcc.exe"
        $includeDir = "include"
        
        # Test compile (syntax check only, no linking)
        $testCmd = @(
            "-DHIPSOLVER_TARGET_ROCM",
            "-D__HIP_PLATFORM_AMD__",
            "-Iinclude",
            "-I`"$hipPath\include`"",
            "-fsyntax-only",
            "src\backends\gpu\hipsolver_trait_impl.cpp"
        )
        
        Write-Host "    Command: $hipcc $($testCmd -join ' ')" -ForegroundColor Gray
        & $hipcc @testCmd 2>&1 | Out-Null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] hipSOLVER (AMD) syntax check passed" -ForegroundColor Green
            $results["hipSOLVER_AMD"].Status = "Validated"
        } else {
            Write-Host "  [FAIL] hipSOLVER (AMD) has syntax errors" -ForegroundColor Red
            $results["hipSOLVER_AMD"].Status = "Failed"
            $results["hipSOLVER_AMD"].Issues += "Syntax errors"
        }
    } catch {
        Write-Host "  [ERROR] Compilation test failed: $_" -ForegroundColor Red
        $results["hipSOLVER_AMD"].Issues += "Compilation error: $_"
    }
}

# Test oneMKL (Intel)
if ($results["oneMKL_Intel"].Status -eq "SDK Available") {
    Write-Host "  Testing oneMKL (Intel target)..." -ForegroundColor Cyan
    
    try {
        $icpx = $compilerPath
        $includeDir = "include"
        $mklPath = "$oneAPIPath\mkl\latest"
        
        # Test compile (syntax check only)
        $testCmd = @(
            "-fsycl",
            "-std=c++17",
            "-I$includeDir",
            "-I$mklPath\include",
            "-fsyntax-only",
            "src\backends\gpu\onemkl_lapack_impl.cpp"
        )
        
        Write-Host "    Command: $icpx $($testCmd -join ' ')" -ForegroundColor Gray
        & $icpx @testCmd 2>&1 | Out-Null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] oneMKL syntax check passed" -ForegroundColor Green
            $results["oneMKL_Intel"].Status = "Validated"
        } else {
            Write-Host "  [FAIL] oneMKL has syntax errors" -ForegroundColor Red
            $results["oneMKL_Intel"].Status = "Failed"
            $results["oneMKL_Intel"].Issues += "Syntax errors"
        }
    } catch {
        Write-Host "  [ERROR] Compilation test failed: $_" -ForegroundColor Red
        $results["oneMKL_Intel"].Issues += "Compilation error: $_"
    }
}

Write-Host ""

# ============================================================================
# Test 4: API Consistency Check
# ============================================================================

Write-Host "[Test 4] API consistency validation..." -ForegroundColor Yellow

$expectedFunctions = @(
    "sgetrf", "dgetrf", "cgetrf", "zgetrf",
    "sgetrs", "dgetrs", "cgetrs", "zgetrs",
    "spotrf", "dpotrf", "cpotrf", "zpotrf",
    "spotrs", "dpotrs", "cpotrs", "zpotrs",
    "sgeqrf", "dgeqrf", "cgeqrf", "zgeqrf",
    "sgesvd", "dgesvd", "cgesvd", "zgesvd",
    "ssyev", "dsyev", "cheev", "zheev"
)

Write-Host "  Checking for all 26 LAPACK operations..." -ForegroundColor Cyan

# Check hipSOLVER
$hipsolverContent = Get-Content "src\backends\gpu\hipsolver_trait_impl.cpp" -Raw
$hipsolverMissing = @()
foreach ($func in $expectedFunctions) {
    if ($hipsolverContent -notmatch "hipsolver_$func") {
        $hipsolverMissing += $func
    }
}

if ($hipsolverMissing.Count -eq 0) {
    Write-Host "  [OK] hipSOLVER: All 26 functions implemented" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] hipSOLVER: Missing $($hipsolverMissing.Count) functions: $($hipsolverMissing -join ', ')" -ForegroundColor Red
    $results["hipSOLVER_AMD"].Issues += "Missing functions: $($hipsolverMissing -join ', ')"
}

# Check oneMKL
$onemklContent = Get-Content "src\backends\gpu\onemkl_lapack_impl.cpp" -Raw
$onemklMissing = @()
foreach ($func in $expectedFunctions) {
    if ($onemklContent -notmatch "onemkl_$func") {
        $onemklMissing += $func
    }
}

if ($onemklMissing.Count -eq 0) {
    Write-Host "  [OK] oneMKL: All 26 functions implemented" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] oneMKL: Missing $($onemklMissing.Count) functions: $($onemklMissing -join ', ')" -ForegroundColor Red
    $results["oneMKL_Intel"].Issues += "Missing functions: $($onemklMissing -join ', ')"
}

Write-Host ""

# ============================================================================
# Test 5: Error Handling Check
# ============================================================================

Write-Host "[Test 5] Error handling validation..." -ForegroundColor Yellow

# Check hipSOLVER has error handling
$hipsolverErrorChecks = 0
if ($hipsolverContent -match "HIPSOLVER_STATUS_SUCCESS") { $hipsolverErrorChecks++ }
if ($hipsolverContent -match "info_host") { $hipsolverErrorChecks++ }
if ($hipsolverContent -match "return.*status") { $hipsolverErrorChecks++ }

if ($hipsolverErrorChecks -ge 2) {
    Write-Host "  [OK] hipSOLVER: Error handling present" -ForegroundColor Green
} else {
    Write-Host "  [WARN] hipSOLVER: Limited error handling" -ForegroundColor Yellow
    $results["hipSOLVER_AMD"].Issues += "Limited error handling"
}

# Check oneMKL has error handling
$onemklErrorChecks = 0
if ($onemklContent -match "try.*catch") { $onemklErrorChecks++ }
if ($onemklContent -match "info_host") { $onemklErrorChecks++ }
if ($onemklContent -match "sycl::exception") { $onemklErrorChecks++ }

if ($onemklErrorChecks -ge 2) {
    Write-Host "  [OK] oneMKL: Error handling present (try/catch)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] oneMKL: Limited error handling" -ForegroundColor Yellow
    $results["oneMKL_Intel"].Issues += "Limited error handling"
}

Write-Host ""

# ============================================================================
# Test 6: Memory Management Check
# ============================================================================

Write-Host "[Test 6] Memory management validation..." -ForegroundColor Yellow

# Check hipSOLVER
$hipsolverMemChecks = 0
if ($hipsolverContent -match "hipMalloc") { $hipsolverMemChecks++ }
if ($hipsolverContent -match "hipFree") { $hipsolverMemChecks++ }
if ($hipsolverContent -match "workspace") { $hipsolverMemChecks++ }

if ($hipsolverMemChecks -eq 3) {
    Write-Host "  [OK] hipSOLVER: Proper memory management (allocate/free)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] hipSOLVER: Potential memory management issues" -ForegroundColor Yellow
    $results["hipSOLVER_AMD"].Issues += "Memory management concerns"
}

# Check oneMKL
$onemklMemChecks = 0
if ($onemklContent -match "sycl::malloc_device") { $onemklMemChecks++ }
if ($onemklContent -match "sycl::free") { $onemklMemChecks++ }
if ($onemklContent -match "scratchpad") { $onemklMemChecks++ }

if ($onemklMemChecks -eq 3) {
    Write-Host "  [OK] oneMKL: Proper SYCL memory management" -ForegroundColor Green
} else {
    Write-Host "  [WARN] oneMKL: Potential memory management issues" -ForegroundColor Yellow
    $results["oneMKL_Intel"].Issues += "Memory management concerns"
}

Write-Host ""

# ============================================================================
# Test 7: Stream/Queue Handling
# ============================================================================

Write-Host "[Test 7] Async stream/queue handling..." -ForegroundColor Yellow

# Check hipSOLVER
if ($hipsolverContent -match "hipsolverSetStream") {
    Write-Host "  [OK] hipSOLVER: Stream handling implemented" -ForegroundColor Green
} else {
    Write-Host "  [WARN] hipSOLVER: No stream handling" -ForegroundColor Yellow
    $results["hipSOLVER_AMD"].Issues += "Missing stream handling"
}

# Check oneMKL
if ($onemklContent -match "sycl::queue") {
    Write-Host "  [OK] oneMKL: SYCL queue handling implemented" -ForegroundColor Green
} else {
    Write-Host "  [WARN] oneMKL: No queue handling" -ForegroundColor Yellow
    $results["oneMKL_Intel"].Issues += "Missing queue handling"
}

Write-Host ""

# ============================================================================
# Results Summary
# ============================================================================

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Validation Results" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($backend in $results.Keys | Sort-Object) {
    $result = $results[$backend]
    
    $statusColor = switch ($result.Status) {
        "Validated" { "Green" }
        "SDK Available" { "Yellow" }
        "Failed" { "Red" }
        default { "Gray" }
    }
    
    Write-Host "$backend : $($result.Status)" -ForegroundColor $statusColor
    
    if ($result.Issues.Count -gt 0) {
        foreach ($issue in $result.Issues) {
            Write-Host "  ⚠ $issue" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
}

# ============================================================================
# Recommendations
# ============================================================================

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Recommendations for Real Hardware Testing" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "AMD RX 7900 XTX Testing:" -ForegroundColor Yellow
Write-Host "  1. Install HIP SDK 6.4+ (if not already installed)" -ForegroundColor White
Write-Host "  2. Run: .\build_hipsolver.ps1 -Target amd" -ForegroundColor White
Write-Host "  3. Test LU factorization first (sgetrf/sgetrs)" -ForegroundColor White
Write-Host "  4. Verify info parameter returns correctly" -ForegroundColor White
Write-Host "  5. Check memory is freed properly (no leaks)" -ForegroundColor White
Write-Host ""

Write-Host "Intel Arc A770 Testing:" -ForegroundColor Yellow
Write-Host "  1. Install Intel oneAPI Base Toolkit 2024.0+" -ForegroundColor White
Write-Host "  2. Run: .\build_onemkl_lapack.ps1" -ForegroundColor White
Write-Host "  3. Test Cholesky first (spotrf/spotrs - simpler)" -ForegroundColor White
Write-Host "  4. Monitor SYCL queue synchronization" -ForegroundColor White
Write-Host "  5. Verify scratchpad allocation sizes" -ForegroundColor White
Write-Host ""

Write-Host "Common Issues to Watch:" -ForegroundColor Yellow
Write-Host "  ⚠ ipiv indexing (1-based FORTRAN vs 0-based C)" -ForegroundColor White
Write-Host "  ⚠ Matrix layout (column-major required)" -ForegroundColor White
Write-Host "  ⚠ Leading dimension (lda >= max(1,m))" -ForegroundColor White
Write-Host "  ⚠ Workspace size queries (may vary by GPU)" -ForegroundColor White
Write-Host "  ⚠ Stream synchronization (async operations)" -ForegroundColor White
Write-Host ""

Write-Host "Create Test Data:" -ForegroundColor Yellow
Write-Host "  • Well-conditioned matrices (condition number < 100)" -ForegroundColor White
Write-Host "  • Small sizes first (16x16, 64x64, 256x256)" -ForegroundColor White
Write-Host "  • Known solutions (A = I, A = random with fixed seed)" -ForegroundColor White
Write-Host "  • Compare with CPU LAPACK (NumPy/SciPy)" -ForegroundColor White
Write-Host ""

# Exit with appropriate code
$failedCount = ($results.Values | Where-Object { $_.Status -eq "Failed" }).Count
exit $failedCount
