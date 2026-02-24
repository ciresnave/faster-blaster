<#
.SYNOPSIS
    Generate a new BLAS/LAPACK backend by adapting an existing one.

.DESCRIPTION
    This script copies an existing backend implementation and performs
    systematic search-and-replace to adapt it for a new backend library.
    
    This dramatically reduces the effort of adding new backends, as the
    CBLAS/LAPACKE API is standardized across implementations.

.PARAMETER SourceBackend
    The source backend to copy from (e.g., "openblas", "mkl")

.PARAMETER TargetBackend
    The new backend to generate (e.g., "blis", "aocl", "accelerate")

.PARAMETER SourceFile
    Path to the source backend .c file

.PARAMETER OutputFile
    Path where the new backend .c file will be written

.EXAMPLE
    .\generate_backend.ps1 -SourceBackend mkl -TargetBackend blis -SourceFile ..\src\backends\mkl_backend.c -OutputFile ..\src\backends\blis_backend.c
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$SourceBackend,
    
    [Parameter(Mandatory=$true)]
    [string]$TargetBackend,
    
    [Parameter(Mandatory=$true)]
    [string]$SourceFile,
    
    [Parameter(Mandatory=$true)]
    [string]$OutputFile
)

# Validate input file exists
if (-not (Test-Path $SourceFile)) {
    Write-Error "Source file not found: $SourceFile"
    exit 1
}

Write-Host "Generating $TargetBackend backend from $SourceBackend backend..." -ForegroundColor Cyan

# Read the source file
$content = Get-Content $SourceFile -Raw

# Perform systematic replacements
# 1. Replace lowercase backend names
$content = $content -replace $SourceBackend.ToLower(), $TargetBackend.ToLower()

# 2. Replace uppercase backend names
$content = $content -replace $SourceBackend.ToUpper(), $TargetBackend.ToUpper()

# 3. Replace mixed-case function prefixes (e.g., mkl_ -> blis_)
$srcPrefix = $SourceBackend.ToLower() + "_"
$tgtPrefix = $TargetBackend.ToLower() + "_"
$content = $content -replace $srcPrefix, $tgtPrefix

# 4. Replace structure names (e.g., g_mkl -> g_blis)
$srcStruct = "g_" + $SourceBackend.ToLower()
$tgtStruct = "g_" + $TargetBackend.ToLower()
$content = $content -replace $srcStruct, $tgtStruct

# 5. Replace capitalized names in comments/docs
$srcCap = $SourceBackend.Substring(0,1).ToUpper() + $SourceBackend.Substring(1).ToLower()
$tgtCap = $TargetBackend.Substring(0,1).ToUpper() + $TargetBackend.Substring(1).ToLower()
$content = $content -replace $srcCap, $tgtCap

# Backend-specific library loading adjustments
$libraryMappings = @{
    "blis" = @{
        "windows" = "libblis.dll"
        "linux" = "libblis.so"
        "macos" = "libblis.dylib"
    }
    "aocl" = @{
        "windows" = "AOCL-LibBlis-Win-MT-dll.dll"
        "linux" = "libblis.so"
        "macos" = "libblis.dylib"
    }
    "accelerate" = @{
        "windows" = $null  # N/A on Windows
        "linux" = $null    # N/A on Linux
        "macos" = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
    }
    "atlas" = @{
        "windows" = "libatlas.dll"
        "linux" = "libatlas.so"
        "macos" = "libatlas.dylib"
    }
}

# Apply library-specific mappings if available
if ($libraryMappings.ContainsKey($TargetBackend.ToLower())) {
    $libs = $libraryMappings[$TargetBackend.ToLower()]
    
    # Update Windows library name pattern
    if ($content -match 'windows_library\[\]\s*=\s*"([^"]+)"') {
        if ($libs.windows) {
            $content = $content -replace 'windows_library\[\]\s*=\s*"[^"]+"', "windows_library[] = `"$($libs.windows)`""
        }
    }
    
    # Update Linux library name pattern
    if ($content -match 'linux_library\[\]\s*=\s*"([^"]+)"') {
        if ($libs.linux) {
            $content = $content -replace 'linux_library\[\]\s*=\s*"[^"]+"', "linux_library[] = `"$($libs.linux)`""
        }
    }
    
    # Update macOS library name pattern
    if ($content -match 'macos_library\[\]\s*=\s*"([^"]+)"') {
        if ($libs.macos) {
            $content = $content -replace 'macos_library\[\]\s*=\s*"[^"]+"', "macos_library[] = `"$($libs.macos)`""
        }
    }
}

# Write the output file
$content | Out-File -FilePath $OutputFile -Encoding UTF8 -NoNewline

Write-Host "✓ Successfully generated $OutputFile" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review the generated file for any backend-specific adjustments"
Write-Host "  2. Update library loading paths if needed"
Write-Host "  3. Add the backend to backend_loader.c"
Write-Host "  4. Test compilation and runtime behavior"
Write-Host ""
Write-Host "Backend-specific notes for ${TargetBackend}:" -ForegroundColor Cyan

# Provide backend-specific guidance
switch ($TargetBackend.ToLower()) {
    "blis" {
        Write-Host "  - BLIS uses standard CBLAS/LAPACKE APIs (same as OpenBLAS/MKL)"
        Write-Host "  - No changes needed to function calls"
    }
    "aocl" {
        Write-Host "  - AMD AOCL uses standard CBLAS/LAPACKE APIs"
        Write-Host "  - Library name on Windows may vary by version"
    }
    "accelerate" {
        Write-Host "  - macOS only - uses vecLib CBLAS/LAPACKE"
        Write-Host "  - Framework path: /System/Library/Frameworks/Accelerate.framework"
        Write-Host "  - Disable Windows/Linux code paths"
    }
    "atlas" {
        Write-Host "  - ATLAS uses standard CBLAS API"
        Write-Host "  - May have limited LAPACK support - verify availability"
    }
    default {
        Write-Host "  - Review API compatibility (most use CBLAS/LAPACKE)"
        Write-Host "  - Check library naming conventions for your platform"
    }
}

Write-Host ""
Write-Host "Estimated time saved: ~4-6 hours of manual coding" -ForegroundColor Green
