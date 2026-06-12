# Automated rocSOLVER API conversion
# This script fixes all remaining _bufferSize calls

$file = "src\backends\gpu\rocblas_trait_impl.c"

# Read file
$content = Get-Content $file -Raw

# Count before
$before = ([regex]::Matches($content, '_bufferSize')).Count
Write-Host "Before: $before _bufferSize calls"

# Fix spotrf - remove workspace, keep devInfo
$content = $content -replace '(?ms)(static int rocblas_spotrf_impl.*?\{.*?)int lwork;\s+float\* workspace;\s+(int\* devInfo.*?stream.*?\})\s+status = rocsolver_spotrf_bufferSize.*?;.*?if.*?return -1;.*?hipMalloc.*?workspace.*?;.*?hipMalloc.*?devInfo.*?;.*?status = rocsolver_spotrf\(ctx->solver_handle, fill, n, \(float\*\)a, lda,\s+workspace, lwork, devInfo\);(.*?hipFree\(workspace\);)', '$1$2    hipMalloc((void**)&devInfo, sizeof(int));    status = rocsolver_spotrf(ctx->solver_handle, fill, n, (float*)a, lda, devInfo);$3'

# Fix dpotrf
$content = $content -replace '(?ms)(static int rocblas_dpotrf_impl.*?\{.*?)int lwork;\s+double\* workspace;\s+(int\* devInfo.*?stream.*?\})\s+status = rocsolver_dpotrf_bufferSize.*?;.*?if.*?return -1;.*?hipMalloc.*?workspace.*?;.*?hipMalloc.*?devInfo.*?;.*?status = rocsolver_dpotrf\(ctx->solver_handle, fill, n, \(double\*\)a, lda,\s+workspace, lwork, devInfo\);(.*?hipFree\(workspace\);)', '$1$2    hipMalloc((void**)&devInfo, sizeof(int));    status = rocsolver_dpotrf(ctx->solver_handle, fill, n, (double*)a, lda, devInfo);$3'

# Fix cpotrf
$content = $content -replace '(?ms)(static int rocblas_cpotrf_impl.*?\{.*?)int lwork;\s+rocblas_float_complex\* workspace;\s+(int\* devInfo.*?stream.*?\})\s+status = rocsolver_cpotrf_bufferSize.*?;.*?if.*?return -1;.*?hipMalloc.*?workspace.*?;.*?hipMalloc.*?devInfo.*?;.*?status = rocsolver_cpotrf\(ctx->solver_handle, fill, n, \(rocblas_float_complex\*\)a, lda,\s+workspace, lwork, devInfo\);(.*?hipFree\(workspace\);)', '$1$2    hipMalloc((void**)&devInfo, sizeof(int));    status = rocsolver_cpotrf(ctx->solver_handle, fill, n, (rocblas_float_complex*)a, lda, devInfo);$3'

# Fix zpotrf
$content = $content -replace '(?ms)(static int rocblas_zpotrf_impl.*?\{.*?)int lwork;\s+rocblas_double_complex\* workspace;\s+(int\* devInfo.*?stream.*?\})\s+status = rocsolver_zpotrf_bufferSize.*?;.*?if.*?return -1;.*?hipMalloc.*?workspace.*?;.*?hipMalloc.*?devInfo.*?;.*?status = rocsolver_zpotrf\(ctx->solver_handle, fill, n, \(rocblas_double_complex\*\)a, lda,\s+workspace, lwork, devInfo\);(.*?hipFree\(workspace\);)', '$1$2    hipMalloc((void**)&devInfo, sizeof(int));    status = rocsolver_zpotrf(ctx->solver_handle, fill, n, (rocblas_double_complex*)a, lda, devInfo);$3'

# Fix potrs functions (no devInfo)
$content = $content -replace '(?s)(rocsolver_spotrs\(ctx->solver_handle, fill, n, nrhs,\s+\(const float\*\)a, lda, \(float\*\)b, ldb), devInfo\)', '$1)'
$content = $content -replace '(?s)(rocsolver_dpotrs\(ctx->solver_handle, fill, n, nrhs,\s+\(const double\*\)a, lda, \(double\*\)b, ldb), devInfo\)', '$1)'
$content = $content -replace '(?s)(rocsolver_cpotrs\(ctx->solver_handle, fill, n, nrhs,\s+\(const rocblas_float_complex\*\)a, lda, \(rocblas_float_complex\*\)b, ldb), devInfo\)', '$1)'
$content = $content -replace '(?s)(rocsolver_zpotrs\(ctx->solver_handle, fill, n, nrhs,\s+\(const rocblas_double_complex\*\)a, lda,\s+\(rocblas_double_complex\*\)b, ldb), devInfo\)', '$1)'

# Remove devInfo malloc/free from potrs functions
$content = $content -replace '(?ms)(static int rocblas_spotrs_impl.*?\{.*?rocblas_fill fill.*?stream.*?\})\s+hipMalloc\(\(void\*\*\)&devInfo, sizeof\(int\)\);(.*?rocsolver_spotrs.*?\);)\s+int info;\s+hipMemcpy\(&info, devInfo, sizeof\(int\), hipMemcpyDeviceToHost\);\s+hipFree\(devInfo\);(.*?return \(status == rocblas_status_success) && info == 0\)', '$1$2$3 == rocblas_status_success'

# Similar for d/c/z potrs...

# Write back
$content | Set-Content $file

$after = ([regex]::Matches($content, '_bufferSize')).Count
Write-Host "After: $after _bufferSize calls"
Write-Host "Fixed: $($before - $after) functions"
