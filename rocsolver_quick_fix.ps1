# rocSOLVER API Fix Script
# Automatically fixes remaining _bufferSize calls to use correct rocSOLVER API

$file = "src\backends\gpu\rocblas_trait_impl.c"
$content = Get-Content $file -Raw

# Pattern 1: getrf-style (has devInfo) - already fixed
# Pattern 2: getrs-style (no devInfo)  - need to fix d/c/z versions  
# Pattern 3: potrf-style (has devInfo) - need to fix all
# Pattern 4: potrs-style (no devInfo) - need to fix all  
# Pattern 5: geqrf-style (has devInfo) - need to fix all
# Pattern 6: gesvd-style (complex) - need to fix all
# Pattern 7: syevd/heevd-style - need to fix all

Write-Host "Fixing remaining rocSOLVER API calls..."
Write-Host "Before: $((Select-String -Path $file -Pattern '_bufferSize').Count) _bufferSize calls"

# Fix dgetrs
$content = $content -replace '(?s)static int rocblas_dgetrs_impl\(void\* handle, fb_gpu_stream_t stream, char trans,\s+int n, int nrhs, fb_gpu_ptr_t a, int lda,\s+fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb\) \{[^}]+\}','static int rocblas_dgetrs_impl(void* handle, fb_gpu_stream_t stream, char trans,
                               int n, int nrhs, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t ipiv, fb_gpu_ptr_t b, int ldb) {
    rocblas_context_t* ctx = (rocblas_context_t*)handle;
    rocblas_operation op = (rocblas_operation)rocblas_convert_transpose(trans);
    rocblas_status status;
    
    if (stream) {
        rocblas_set_stream(ctx->solver_handle, (hipStream_t)stream);
    }
    
    status = rocsolver_dgetrs(ctx->solver_handle, op, n, nrhs,
                              (double*)a, lda, (int*)ipiv,
                              (double*)b, ldb);
    
    return (status == rocblas_status_success) ? 0 : -1;
}'

# Continue with more replacements...
Write-Host "Note: Manual fixes needed for complex patterns. See rocsolver_complete_fix.ps1"
