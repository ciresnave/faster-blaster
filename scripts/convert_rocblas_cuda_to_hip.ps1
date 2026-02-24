# Convert rocBLAS generated file from CUDA to HIP types
# This script performs systematic find/replace for CUDA→HIP conversion

$file = "src\backends\gpu\rocblas_trait_impl_generated.c"

if (-not (Test-Path $file)) {
    Write-Error "File not found: $file"
    exit 1
}

Write-Host "Converting $file from CUDA to HIP..." -ForegroundColor Cyan

# Read file content
$content = Get-Content $file -Raw

# Count original occurrences
Write-Host "`nOriginal CUDA references:" -ForegroundColor Yellow
Write-Host "  cusolverStatus_t: $(([regex]::Matches($content, 'cusolverStatus_t')).Count)"
Write-Host "  CUSOLVER_STATUS_SUCCESS: $(([regex]::Matches($content, 'CUSOLVER_STATUS_SUCCESS')).Count)"
Write-Host "  cusolverDn: $(([regex]::Matches($content, 'cusolverDn')).Count)"
Write-Host "  cublasCreate: $(([regex]::Matches($content, 'cublasCreate')).Count)"
Write-Host "  cublasDestroy: $(([regex]::Matches($content, 'cublasDestroy')).Count)"

# Phase 1: Type conversions
Write-Host "`nPhase 1: Type conversions..." -ForegroundColor Green
$content = $content -replace 'cusolverStatus_t', 'rocblas_status'
$content = $content -replace 'CUSOLVER_STATUS_SUCCESS', 'rocblas_status_success'

# Phase 2: Handle creation/destruction
Write-Host "Phase 2: Handle management..." -ForegroundColor Green
$content = $content -replace 'cublasCreate', 'rocblas_create_handle'
$content = $content -replace 'cublasDestroy', 'rocblas_destroy_handle'

# Phase 3: Set stream
Write-Host "Phase 3: Stream operations..." -ForegroundColor Green
$content = $content -replace 'cusolverDnSetStream', 'rocblas_set_stream'

# Phase 4: LU Factorization (getrf)
Write-Host "Phase 4: LU factorization functions..." -ForegroundColor Green
$content = $content -replace 'cusolverDnSgetrf_bufferSize', 'rocsolver_sgetrf'  # Will need manual cleanup
$content = $content -replace 'cusolverDnDgetrf_bufferSize', 'rocsolver_dgetrf'
$content = $content -replace 'cusolverDnCgetrf_bufferSize', 'rocsolver_cgetrf'
$content = $content -replace 'cusolverDnZgetrf_bufferSize', 'rocsolver_zgetrf'
$content = $content -replace 'cusolverDnSgetrf', 'rocsolver_sgetrf'
$content = $content -replace 'cusolverDnDgetrf', 'rocsolver_dgetrf'
$content = $content -replace 'cusolverDnCgetrf', 'rocsolver_cgetrf'
$content = $content -replace 'cusolverDnZgetrf', 'rocsolver_zgetrf'

# Phase 5: Linear Solve (getrs)
Write-Host "Phase 5: Linear solve functions..." -ForegroundColor Green
$content = $content -replace 'cusolverDnSgetrs', 'rocsolver_sgetrs'
$content = $content -replace 'cusolverDnDgetrs', 'rocsolver_dgetrs'
$content = $content -replace 'cusolverDnCgetrs', 'rocsolver_cgetrs'
$content = $content -replace 'cusolverDnZgetrs', 'rocsolver_zgetrs'

# Phase 6: Cholesky (potrf/potrs)
Write-Host "Phase 6: Cholesky functions..." -ForegroundColor Green
$content = $content -replace 'cusolverDnSpotrf_bufferSize', 'rocsolver_spotrf'
$content = $content -replace 'cusolverDnDpotrf_bufferSize', 'rocsolver_dpotrf'
$content = $content -replace 'cusolverDnCpotrf_bufferSize', 'rocsolver_cpotrf'
$content = $content -replace 'cusolverDnZpotrf_bufferSize', 'rocsolver_zpotrf'
$content = $content -replace 'cusolverDnSpotrf', 'rocsolver_spotrf'
$content = $content -replace 'cusolverDnDpotrf', 'rocsolver_dpotrf'
$content = $content -replace 'cusolverDnCpotrf', 'rocsolver_cpotrf'
$content = $content -replace 'cusolverDnZpotrf', 'rocsolver_zpotrf'
$content = $content -replace 'cusolverDnSpotrs', 'rocsolver_spotrs'
$content = $content -replace 'cusolverDnDpotrs', 'rocsolver_dpotrs'
$content = $content -replace 'cusolverDnCpotrs', 'rocsolver_cpotrs'
$content = $content -replace 'cusolverDnZpotrs', 'rocsolver_zpotrs'

# Phase 7: QR Factorization (geqrf)
Write-Host "Phase 7: QR factorization functions..." -ForegroundColor Green
$content = $content -replace 'cusolverDnSgeqrf_bufferSize', 'rocsolver_sgeqrf'
$content = $content -replace 'cusolverDnDgeqrf_bufferSize', 'rocsolver_dgeqrf'
$content = $content -replace 'cusolverDnCgeqrf_bufferSize', 'rocsolver_cgeqrf'
$content = $content -replace 'cusolverDnZgeqrf_bufferSize', 'rocsolver_zgeqrf'
$content = $content -replace 'cusolverDnSgeqrf', 'rocsolver_sgeqrf'
$content = $content -replace 'cusolverDnDgeqrf', 'rocsolver_dgeqrf'
$content = $content -replace 'cusolverDnCgeqrf', 'rocsolver_cgeqrf'
$content = $content -replace 'cusolverDnZgeqrf', 'rocsolver_zgeqrf'

# Phase 8: SVD (gesvd)
Write-Host "Phase 8: SVD functions..." -ForegroundColor Green
$content = $content -replace 'cusolverDnSgesvd_bufferSize', 'rocsolver_sgesvd'
$content = $content -replace 'cusolverDnDgesvd_bufferSize', 'rocsolver_dgesvd'
$content = $content -replace 'cusolverDnCgesvd_bufferSize', 'rocsolver_cgesvd'
$content = $content -replace 'cusolverDnZgesvd_bufferSize', 'rocsolver_zgesvd'
$content = $content -replace 'cusolverDnSgesvd', 'rocsolver_sgesvd'
$content = $content -replace 'cusolverDnDgesvd', 'rocsolver_dgesvd'
$content = $content -replace 'cusolverDnCgesvd', 'rocsolver_cgesvd'
$content = $content -replace 'cusolverDnZgesvd', 'rocsolver_zgesvd'

# Phase 9: Eigenvalues (syevd/heevd)
Write-Host "Phase 9: Eigenvalue functions..." -ForegroundColor Green
$content = $content -replace 'cusolverDnSsyevd_bufferSize', 'rocsolver_ssyevd'
$content = $content -replace 'cusolverDnDsyevd_bufferSize', 'rocsolver_dsyevd'
$content = $content -replace 'cusolverDnCheevd_bufferSize', 'rocsolver_cheevd'
$content = $content -replace 'cusolverDnZheevd_bufferSize', 'rocsolver_zheevd'
$content = $content -replace 'cusolverDnSsyevd', 'rocsolver_ssyevd'
$content = $content -replace 'cusolverDnDsyevd', 'rocsolver_dsyevd'
$content = $content -replace 'cusolverDnCheevd', 'rocsolver_cheevd'
$content = $content -replace 'cusolverDnZheevd', 'rocsolver_zheevd'

# Phase 10: Handle references
Write-Host "Phase 10: Handle references..." -ForegroundColor Green
$content = $content -replace 'ctx->cusolver_handle', 'ctx->rocblas_handle'

# Phase 11: Comments
Write-Host "Phase 11: Update comments..." -ForegroundColor Green
$content = $content -replace 'cuSOLVER', 'rocSOLVER'
$content = $content -replace 'cuBLAS returns 1-based', 'rocBLAS returns 1-based'

# Backup original
$backup = "$file.cuda_backup"
Copy-Item $file $backup -Force
Write-Host "`nBackup created: $backup" -ForegroundColor Magenta

# Write converted content
$content | Set-Content $file -NoNewline

Write-Host "`nConversion complete!" -ForegroundColor Green
Write-Host "`nRemaining manual fixes needed:" -ForegroundColor Yellow
Write-Host "  1. Remove cusolverDnCreate/cusolverDnDestroy from init/shutdown"
Write-Host "  2. Add rocsolver_handle field to rocblas_context_t struct"
Write-Host "  3. Remove workspace buffer size queries and allocations"
Write-Host "  4. Update function signatures (remove workspace parameters)"
Write-Host "`nRun 'git diff $file' to review changes"
