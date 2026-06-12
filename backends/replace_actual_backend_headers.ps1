$root = "c:\Users\cires\OneDrive\Documents\projects\faster-blaster"

function Replace-BackendHeaders {
    param(
        [string]$BackendName,
        [string]$SourceDir,
        [string[]]$Patterns
    )

    $destDir = Join-Path $root "backends\headers\$BackendName"
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir | Out-Null
    }

    Get-ChildItem -Path $destDir -Force -Recurse | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue

    foreach ($pattern in $Patterns) {
        $files = Get-ChildItem -Path $SourceDir -Filter $pattern -File -Recurse -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            Copy-Item -Path $file.FullName -Destination $destDir -Force
        }
    }

    Write-Host "Replaced backend headers for $BackendName from $SourceDir"
}

$openblasSrc = Join-Path $root "build-clang\backends-install\openblas\include"
if (Test-Path $openblasSrc) {
    Replace-BackendHeaders -BackendName openblas -SourceDir $openblasSrc -Patterns @('cblas.h','openblas_config.h','*.h')
} else {
    Write-Warning "OpenBLAS headers not found at $openblasSrc"
}

$blisSrc = Join-Path $root "build-clang\backends-install\blis\include\blis"
if (Test-Path $blisSrc) {
    Replace-BackendHeaders -BackendName blis -SourceDir $blisSrc -Patterns @('*.h')
    $blisCblas = Join-Path $root "build-clang\backends-install\blis\include\cblas.h"
    if (Test-Path $blisCblas) {
        Copy-Item -Path $blisCblas -Destination (Join-Path $root "backends\headers\blis") -Force
    }
} else {
    Write-Warning "BLIS headers not found at $blisSrc"
}

$mklsrc = Join-Path $root "build-clang\backends-install\mkl\include"
if (Test-Path $mklsrc) {
    Replace-BackendHeaders -BackendName mkl -SourceDir $mklsrc -Patterns @('*.h')
} else {
    Write-Warning "MKL headers not found at $mklsrc; leaving existing MKL header set in place"
}
