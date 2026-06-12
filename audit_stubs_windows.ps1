# Audit Script for Stub Detection (Windows PowerShell)
# Purpose: Identify all stub/placeholder/partial implementations in faster-blaster-reference
# Usage: .\audit_stubs_windows.ps1

param(
    [string]$SourceDir = ".\faster-blaster-reference\src",
    [string]$OutputFile = "IMPLEMENTATION_STATUS_AUDIT.csv"
)

Write-Host "🔍 Scanning for stubs in: $SourceDir" -ForegroundColor Cyan

# Initialize results array
$results = @()
$results += "operation,file,status,completeness_percent,category,priority"

# Search patterns for stubs
$stubPatterns = @(
    "TODO",
    "FIXME",
    "NOT_IMPLEMENTED",
    "STUB",
    "not yet",
    "not implemented",
    "placeholder",
    "implement me"
)

# Category mappings
$categoryMap = @{
    "l1" = "BLAS L1"
    "l2" = "BLAS L2"
    "l3" = "BLAS L3"
    "driver" = "LAPACK Driver"
    "computational" = "LAPACK Computational"
    "auxiliary" = "LAPACK Auxiliary"
}

# Get all .c files in src directory
$files = Get-ChildItem -Path $SourceDir -Filter "*.c" -Recurse

Write-Host "Found $($files.Count) source files" -ForegroundColor Green

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    $lines = @(Get-Content $file.FullName)
    
    # Determine category from path
    $category = "Unknown"
    $relPath = $file.FullName -replace [regex]::Escape($SourceDir), ""
    
    if ($relPath -match "blas.*l1") { $category = "BLAS L1"; $priority = 1 }
    elseif ($relPath -match "blas.*l2") { $category = "BLAS L2"; $priority = 2 }
    elseif ($relPath -match "blas.*l3") { $category = "BLAS L3"; $priority = 3 }
    elseif ($relPath -match "lapack.*driver") { $category = "LAPACK Driver"; $priority = 4 }
    elseif ($relPath -match "lapack.*computational") { $category = "LAPACK Computational"; $priority = 5 }
    elseif ($relPath -match "lapack.*auxiliary") { $category = "LAPACK Auxiliary"; $priority = 6 }
    else { $category = "Other"; $priority = 7 }
    
    # Extract function name from filename
    $operationName = $file.BaseName
    
    # Determine status and completeness
    $status = "COMPLETE"
    $completeness = 100
    
    # Check for empty function bodies (just opening brace, closing brace)
    if ($content -match "^[^{]*{[\s\n]*}") {
        $status = "EMPTY"
        $completeness = 0
    }
    # Check for stub comments
    else {
        foreach ($pattern in $stubPatterns) {
            if ($content -match $pattern) {
                if ($status -eq "COMPLETE") {
                    $status = "PARTIAL"
                    $completeness = 30
                }
                break
            }
        }
    }
    
    # Check function size (very small = likely stub)
    $lineCount = $lines.Count
    if ($lineCount -lt 20 -and $status -eq "COMPLETE") {
        # Might be stub, but could be simple operation
        # Mark as REVIEW if under 15 lines
        if ($lineCount -lt 15) {
            $status = "REVIEW"
            $completeness = 50
        }
    }
    
    # Add to results
    $results += "$operationName,$($file.FullName),$status,$completeness,$category,$priority"
    
    Write-Host "  $operationName : $status ($completeness%)" -ForegroundColor $(
        if ($status -eq "COMPLETE") { "Green" } 
        elseif ($status -eq "PARTIAL") { "Yellow" } 
        else { "Red" }
    )
}

# Save results
$results | Out-File -FilePath $OutputFile -Encoding UTF8

Write-Host "`n✅ Audit complete! Results saved to: $OutputFile" -ForegroundColor Green

# Summary statistics
$complete = ($results -match ",COMPLETE,").Count
$partial = ($results -match ",PARTIAL,").Count
$empty = ($results -match ",EMPTY,").Count
$review = ($results -match ",REVIEW,").Count

Write-Host "`n📊 Summary:" -ForegroundColor Cyan
Write-Host "  Complete: $complete" -ForegroundColor Green
Write-Host "  Partial:  $partial" -ForegroundColor Yellow
Write-Host "  Empty:    $empty" -ForegroundColor Red
Write-Host "  Review:   $review" -ForegroundColor Yellow

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  1. Review $OutputFile"
Write-Host "  2. Sort by completeness_percent to prioritize work"
Write-Host "  3. Assign operations to developers"
Write-Host "  4. Begin implementation with highest priority items"
