#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run all faster-blaster backend tests
    
.DESCRIPTION
    Builds and runs tests for all available backends (cuBLAS, AOCL, BLIS, oneMKL).
    Reports pass/fail status for each backend.
    
.PARAMETER BuildDir
    Build directory (default: build)
    
.PARAMETER Config
    Build configuration: Debug or Release (default: Release)
    
.PARAMETER Backends
    Comma-separated list of backends to test (default: all)
    Valid values: cublas,aocl,blis,onemkl
    
.EXAMPLE
    .\test_all_backends.ps1
    
.EXAMPLE
    .\test_all_backends.ps1 -Backends cublas,aocl -Config Debug
#>

param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$Backends = "all"
)

# Color output functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║ $Message" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Failure {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

# Test result tracking
$script:TestResults = @{}
$script:TotalTests = 0
$script:PassedTests = 0
$script:FailedTests = 0
$script:SkippedTests = 0

# Parse backend list
$BackendList = @()
$skipBuild = $false
if ($Backends -eq "all") {
    $BackendList = @("cublas", "aocl", "blis", "onemkl", "reference")
} elseif ($Backends -eq "reference") {
    $BackendList = @("reference")
    $skipBuild = $true  # Reference test is always pre-built
} else {
    $BackendList = $Backends -split "," | ForEach-Object { $_.Trim() }
}

Write-Header "Faster-Blaster Backend Test Runner"

Write-Info "Configuration:"
Write-Host "  Build Directory: $BuildDir"
Write-Host "  Configuration: $Config"
Write-Host "  Backends: $($BackendList -join ', ')"

# Check if build directory exists
if (-not (Test-Path $BuildDir)) {
    Write-Failure "Build directory '$BuildDir' not found!"
    Write-Info "Run 'cmake -B $BuildDir -DCMAKE_BUILD_TYPE=$Config' first"
    exit 1
}

# Change to build directory
Push-Location $BuildDir

try {
    # ========================================================================
    # Build the tests (unless only running reference)
    # ========================================================================
    if (-not $skipBuild) {
        Write-Header "Building Tests"
        
        Write-Info "Running CMake build..."
        $buildResult = cmake --build . --config $Config 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Failure "Build failed!"
            Write-Host $buildResult
            exit 1
        }
        
        Write-Success "Build completed successfully"
    } else {
        Write-Header "Test Execution (Reference Only - No Build Required)"
        Write-Info "Reference implementation is always available in source tree"
    }
    
    # ========================================================================
    # Run cuBLAS tests
    # ========================================================================
    if ($BackendList -contains "cublas") {
        Write-Header "Testing cuBLAS Backend (NVIDIA GPU)"
        $script:TotalTests++
        
        $testExe = Join-Path "tests" $Config "test_cublas_basic.exe"
        
        if (Test-Path $testExe) {
            Write-Info "Running $testExe..."
            & $testExe
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "cuBLAS tests PASSED"
                $script:TestResults["cuBLAS"] = "PASSED"
                $script:PassedTests++
            } else {
                Write-Failure "cuBLAS tests FAILED (exit code: $LASTEXITCODE)"
                $script:TestResults["cuBLAS"] = "FAILED"
                $script:FailedTests++
            }
        } else {
            Write-Warning "cuBLAS test executable not found - skipping"
            Write-Info "Enable with: cmake -DENABLE_CUBLAS=ON"
            $script:TestResults["cuBLAS"] = "SKIPPED"
            $script:SkippedTests++
        }
    }
    
    # ========================================================================
    # Run AOCL tests
    # ========================================================================
    if ($BackendList -contains "aocl") {
        Write-Header "Testing AOCL Backend (AMD CPU)"
        $script:TotalTests++
        
        $testExe = Join-Path "tests" $Config "test_aocl_backend.exe"
        
        if (Test-Path $testExe) {
            Write-Info "Running $testExe..."
            & $testExe
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "AOCL tests PASSED"
                $script:TestResults["AOCL"] = "PASSED"
                $script:PassedTests++
            } else {
                Write-Failure "AOCL tests FAILED (exit code: $LASTEXITCODE)"
                $script:TestResults["AOCL"] = "FAILED"
                $script:FailedTests++
            }
        } else {
            Write-Warning "AOCL test executable not found - skipping"
            Write-Info "Enable with: cmake -DENABLE_AOCL=ON"
            $script:TestResults["AOCL"] = "SKIPPED"
            $script:SkippedTests++
        }
    }
    
    # ========================================================================
    # Run BLIS tests
    # ========================================================================
    if ($BackendList -contains "blis") {
        Write-Header "Testing BLIS Backend (Portable CPU)"
        $script:TotalTests++
        
        $testExe = Join-Path "tests" $Config "test_blis_backend.exe"
        
        if (Test-Path $testExe) {
            Write-Info "Running $testExe..."
            & $testExe
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "BLIS tests PASSED"
                $script:TestResults["BLIS"] = "PASSED"
                $script:PassedTests++
            } else {
                Write-Failure "BLIS tests FAILED (exit code: $LASTEXITCODE)"
                $script:TestResults["BLIS"] = "FAILED"
                $script:FailedTests++
            }
        } else {
            Write-Warning "BLIS test executable not found - skipping"
            Write-Info "Enable with: cmake -DENABLE_BLIS=ON"
            $script:TestResults["BLIS"] = "SKIPPED"
            $script:SkippedTests++
        }
    }
    
    # ========================================================================
    # Run oneMKL tests
    # ========================================================================
    if ($BackendList -contains "onemkl") {
        Write-Header "Testing oneMKL Backend (Intel GPU / CUDA)"
        $script:TotalTests++
        
        $testExe = Join-Path "tests" $Config "test_onemkl_backend.exe"
        
        if (Test-Path $testExe) {
            Write-Info "Running $testExe..."
            & $testExe
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "oneMKL tests PASSED"
                $script:TestResults["oneMKL"] = "PASSED"
                $script:PassedTests++
            } else {
                Write-Failure "oneMKL tests FAILED (exit code: $LASTEXITCODE)"
                $script:TestResults["oneMKL"] = "FAILED"
                $script:FailedTests++
            }
        } else {
            Write-Warning "oneMKL test executable not found - skipping"
            Write-Info "Enable with: cmake -DENABLE_ONEMKL=ON"
            $script:TestResults["oneMKL"] = "SKIPPED"
            $script:SkippedTests++
        }
    }
    
    # ========================================================================
    # Run Reference Implementation Tests (Correctness Baseline)
    # ========================================================================
    Write-Header "Testing Reference Backend (Correctness Validation)"
    $script:TotalTests++
    
    $testExe = Join-Path "tests" $Config "test_blas_level1_reference_expanded.exe"
    
    if (Test-Path $testExe) {
        Write-Info "Running $testExe..."
        Write-Info "This test validates all 54 BLAS Level 1 operations with inline implementations"
        Write-Info "Reference backend acts as a correctness oracle for size-aware selection"
        Write-Host ""
        & $testExe
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Reference backend tests PASSED (54/54 operations validated)"
            $script:TestResults["Reference"] = "PASSED"
            $script:PassedTests++
        } else {
            Write-Failure "Reference backend tests FAILED (exit code: $LASTEXITCODE)"
            $script:TestResults["Reference"] = "FAILED"
            $script:FailedTests++
        }
    } else {
        Write-Warning "Reference test executable not found - skipping"
        Write-Info "Reference implementation is always available in source (reference.c)"
        $script:TestResults["Reference"] = "SKIPPED"
        $script:SkippedTests++
    }
    
    # ========================================================================
    # Print Summary
    # ========================================================================
    Write-Header "Test Summary"
    
    Write-Host "`nBackend Results:"
    foreach ($backend in $script:TestResults.Keys | Sort-Object) {
        $result = $script:TestResults[$backend]
        $icon = switch ($result) {
            "PASSED" { "✅"; break }
            "FAILED" { "❌"; break }
            "SKIPPED" { "⏭️ "; break }
        }
        
        $color = switch ($result) {
            "PASSED" { "Green"; break }
            "FAILED" { "Red"; break }
            "SKIPPED" { "Yellow"; break }
        }
        
        Write-Host "  $icon $backend : " -NoNewline
        Write-Host $result -ForegroundColor $color
    }
    
    Write-Host "`nOverall Statistics:"
    Write-Host "  Total Backends Tested: $script:TotalTests"
    Write-Host "  Passed: " -NoNewline
    Write-Host $script:PassedTests -ForegroundColor Green
    Write-Host "  Failed: " -NoNewline
    Write-Host $script:FailedTests -ForegroundColor Red
    Write-Host "  Skipped: " -NoNewline
    Write-Host $script:SkippedTests -ForegroundColor Yellow
    
    $successRate = if ($script:TotalTests -gt 0) {
        [math]::Round(($script:PassedTests / $script:TotalTests) * 100, 1)
    } else { 0 }
    
    Write-Host "`n  Success Rate: " -NoNewline
    $rateColor = if ($successRate -ge 80) { "Green" } 
                 elseif ($successRate -ge 50) { "Yellow" } 
                 else { "Red" }
    Write-Host "$successRate%" -ForegroundColor $rateColor
    
    # ========================================================================
    # Exit with appropriate code
    # ========================================================================
    if ($script:FailedTests -gt 0) {
        Write-Host "`n" -NoNewline
        Write-Failure "Some tests failed!"
        exit 1
    } elseif ($script:PassedTests -eq 0) {
        Write-Host "`n" -NoNewline
        Write-Warning "No tests were run"
        exit 1
    } else {
        Write-Host "`n" -NoNewline
        Write-Success "All tests passed!"
        exit 0
    }
    
} finally {
    Pop-Location
}
