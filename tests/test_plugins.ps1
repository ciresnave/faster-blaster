#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test the plugin architecture

.DESCRIPTION
    Runs the plugin architecture test which validates all 9 plugins register
    correctly and the best backend is selected based on hardware.

.PARAMETER BuildDir
    Build directory (default: build)

.PARAMETER Config
    Build configuration: Debug or Release (default: Release)

.EXAMPLE
    .\test_plugins.ps1

.EXAMPLE
    .\test_plugins.ps1 -Config Debug
#>

param(
    [string]$BuildDir = "build",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

# Color output functions
function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Failure {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║ $($Message.PadRight(58)) ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

# Change to project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Header "Faster-Blaster Plugin Architecture Test"

Write-Info "Configuration:"
Write-Host "  Build Directory: $BuildDir"
Write-Host "  Configuration: $Config"
Write-Host ""

# Build test
Write-Header "Building Plugin Test"
Write-Info "Running CMake build..."

$BuildOutput = cmake --build $BuildDir --config $Config --target test_plugin_architecture 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Failure "Build failed!"
    Write-Host $BuildOutput
    exit 1
}

Write-Success "Build completed successfully"

# Run test
Write-Header "Running Plugin Architecture Test"

$TestExe = Join-Path $BuildDir "tests\$Config\test_plugin_architecture.exe"

if (-not (Test-Path $TestExe)) {
    Write-Failure "Test executable not found: $TestExe"
    exit 1
}

Write-Info "Executing: $TestExe"
Write-Host ""

& $TestExe

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Success "Plugin architecture test PASSED!"
    Write-Host ""
    Write-Host "✨ All 9 plugins registered correctly:" -ForegroundColor Green
    Write-Host "   CPU Plugins (5): AOCL, BLIS, OpenBLAS, MKL, Accelerate" -ForegroundColor Gray
    Write-Host "   GPU Plugins (4): cuBLAS, rocBLAS, oneMKL, Metal" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🎯 Hardware-aware selection working!" -ForegroundColor Green
    Write-Host ""
    exit 0
} else {
    Write-Host ""
    Write-Failure "Plugin architecture test FAILED (exit code: $LASTEXITCODE)"
    exit 1
}
