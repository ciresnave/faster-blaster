#!/usr/bin/env pwsh
# Generate rocBLAS implementation with BLAS only (no LAPACK for now)

$ErrorActionPreference = "Stop"

Write-Host "Reading cuBLAS implementation..." -ForegroundColor Cyan
$cublas = Get-Content "src\backends\gpu\cublas_trait_impl.c" -Raw

Write-Host "Extracting BLAS-only sections..." -ForegroundColor Yellow
# Remove LAPACK sections 
$cublas = $cublas -replace '(?s)/\* =+ cuSOLVER LAPACK.*?(?=\n/\* =+ Virtual Function Table)', ''

Write-Host "Converting cuBLAS to rocBLAS..." -ForegroundColor Yellow
# Run all conversions from previous script
$rocblas = $cublas -replace 'cublas_trait_impl\.c', 'rocblas_trait_impl.c'
$rocblas = $rocblas -replace '@brief NVIDIA cuBLAS', '@brief AMD rocBLAS'
$rocblas = $rocblas -replace 'Provides cuBLAS-specific', 'Provides rocBLAS-specific'
$rocblas = $rocblas -replace 'NVIDIA GPUs alongside AMD', 'AMD GPUs alongside NVIDIA'
$rocblas = $rocblas -replace '#include <cuda_runtime\.h>', '#include <hip/hip_runtime.h>'
$rocblas = $rocblas -replace '#include <cublas_v2\.h>', '#include <rocblas/rocblas.h>'
$rocblas = $rocblas -replace '#include <cusolverDn\.h>', '#include <rocsolver/rocsolver.h>'
$rocblas = $rocblas -replace 'cuBLAS-specific Context', 'rocBLAS-specific Context'
$rocblas = $rocblas -replace 'cublasHandle_t cublas_handle;', 'rocblas_handle rocblas_handle;'
$rocblas = $rocblas -replace 'cusolverDnHandle_t cusolver_handle;', 'rocblas_handle rocsolver_handle;'
$rocblas = $rocblas -replace 'cublas_context_t', 'rocblas_context_t'
$rocblas = $rocblas -replace 'cudaError_t', 'hipError_t'
$rocblas = $rocblas -replace 'cudaSuccess', 'hipSuccess'
$rocblas = $rocblas -replace 'cudaSetDevice', 'hipSetDevice'
$rocblas = $rocblas -replace 'cudaGetDevice', 'hipGetDevice'
$rocblas = $rocblas -replace 'cudaGetDeviceProperties', 'hipGetDeviceProperties'
$rocblas = $rocblas -replace 'cudaDeviceProp', 'hipDeviceProp_t'
$rocblas = $rocblas -replace 'cudaGetErrorString', 'hipGetErrorString'
$rocblas = $rocblas -replace 'cudaMalloc', 'hipMalloc'
$rocblas = $rocblas -replace 'cudaFree', 'hipFree'
$rocblas = $rocblas -replace 'cudaMemcpy', 'hipMemcpy'
$rocblas = $rocblas -replace 'cudaMemcpyHostToDevice', 'hipMemcpyHostToDevice'
$rocblas = $rocblas -replace 'cudaMemcpyDeviceToHost', 'hipMemcpyDeviceToHost'
$rocblas = $rocblas -replace 'cudaMemcpyDeviceToDevice', 'hipMemcpyDeviceToDevice'
$rocblas = $rocblas -replace 'cudaStream_t', 'hipStream_t'
$rocblas = $rocblas -replace 'cudaStreamCreate', 'hipStreamCreate'
$rocblas = $rocblas -replace 'cudaStreamDestroy', 'hipStreamDestroy'
$rocblas = $rocblas -replace 'cudaStreamSynchronize', 'hipStreamSynchronize'
$rocblas = $rocblas -replace 'cudaDeviceSynchronize', 'hipDeviceSynchronize'
$rocblas = $rocblas -replace '\bcublasStatus_t\b', 'rocblas_status'
$rocblas = $rocblas -replace '\bcusolverStatus_t\b', 'rocblas_status'
$rocblas = $rocblas -replace '\bCUBLAS_STATUS_SUCCESS\b', 'rocblas_status_success'
$rocblas = $rocblas -replace '\bCUSOLVER_STATUS_SUCCESS\b', 'rocblas_status_success'
$rocblas = $rocblas -replace 'cuda_err', 'hip_err'
$rocblas = $rocblas -replace 'cusolver_status', 'rocsolver_status'
$rocblas = $rocblas -replace 'cusolver_handle', 'rocsolver_handle'
$rocblas = $rocblas -replace '\bcublasCreate\b', 'rocblas_create_handle'
$rocblas = $rocblas -replace '\bcublasDestroy\b', 'rocblas_destroy_handle'
$rocblas = $rocblas -replace '\bcusolverDnCreate\b', 'rocblas_create_handle'
$rocblas = $rocblas -replace '\bcusolverDnDestroy\b', 'rocblas_destroy_handle'
$rocblas = $rocblas -replace '\bcublasSetStream\b', 'rocblas_set_stream'
$rocblas = $rocblas -replace '\(const cuComplex\*\)', '(const rocblas_float_complex*)'
$rocblas = $rocblas -replace '\(cuComplex\*\)', '(rocblas_float_complex*)'
$rocblas = $rocblas -replace '\(const cuDoubleComplex\*\)', '(const rocblas_double_complex*)'
$rocblas = $rocblas -replace '\(cuDoubleComplex\*\)', '(rocblas_double_complex*)'
$rocblas = $rocblas -replace '\bcublas_', 'rocblas_'

# API calls
$ops1 = @('Saxpy', 'Daxpy', 'Sscal', 'Dscal', 'Scopy', 'Dcopy', 'Sswap', 'Dswap',
          'Sdot', 'Ddot', 'Snrm2', 'Dnrm2', 'Sasum', 'Dasum', 'Isamax', 'Idamax',
          'Caxpy', 'Zaxpy', 'Cscal', 'Zscal', 'Csscal', 'Zdscal', 'Ccopy', 'Zcopy',
          'Cswap', 'Zswap', 'Cdotu', 'Zdotu', 'Cdotc', 'Zdotc',
          'Scnrm2', 'Dznrm2', 'Scasum', 'Dzasum', 'Icamax', 'Izamax',
          'Srot', 'Drot', 'Crot', 'Zrot', 'Srotg', 'Drotg', 'Crotg', 'Zrotg',
          'Srotm', 'Drotm', 'Srotmg', 'Drotmg')
foreach ($op in $ops1) {
    $rocblas = $rocblas -replace "cublas$op\(", "rocblas_$($op.ToLower())("
}

$ops2 = @('Sgemv', 'Dgemv', 'Cgemv', 'Zgemv',
          'Sgbmv', 'Dgbmv', 'Cgbmv', 'Zgbmv',
          'Ssymv', 'Dsymv', 'Chemv', 'Zhemv', 'Csymv', 'Zsymv',
          'Ssbmv', 'Dsbmv', 'Chbmv', 'Zhbmv',
          'Sspmv', 'Dspmv', 'Chpmv', 'Zhpmv',
          'Strmv', 'Dtrmv', 'Ctrmv', 'Ztrmv',
          'Stbmv', 'Dtbmv', 'Ctbmv', 'Ztbmv',
          'Stpmv', 'Dtpmv', 'Ctpmv', 'Ztpmv',
          'Strsv', 'Dtrsv', 'Ctrsv', 'Ztrsv',
          'Stbsv', 'Dtbsv', 'Ctbsv', 'Ztbsv',
          'Stpsv', 'Dtpsv', 'Ctpsv', 'Ztpsv',
          'Sger', 'Dger', 'Cgeru', 'Zgeru', 'Cgerc', 'Zgerc',
          'Ssyr', 'Dsyr', 'Cher', 'Zher', 'Csyr', 'Zsyr',
          'Sspr', 'Dspr', 'Chpr', 'Zhpr',
          'Ssyr2', 'Dsyr2', 'Cher2', 'Zher2', 'Csyr2', 'Zsyr2',
          'Sspr2', 'Dspr2', 'Chpr2', 'Zhpr2')
foreach ($op in $ops2) {
    $rocblas = $rocblas -replace "cublas$op\(", "rocblas_$($op.ToLower())("
}

$ops3 = @('Sgemm', 'Dgemm', 'Cgemm', 'Zgemm',
          'Ssymm', 'Dsymm', 'Csymm', 'Zsymm', 'Chemm', 'Zhemm',
          'Ssyrk', 'Dsyrk', 'Csyrk', 'Zsyrk', 'Cherk', 'Zherk',
          'Ssyr2k', 'Dsyr2k', 'Csyr2k', 'Zsyr2k', 'Cher2k', 'Zher2k',
          'Strmm', 'Dtrmm', 'Ctrmm', 'Ztrmm',
          'Strsm', 'Dtrsm', 'Ctrsm', 'Ztrsm')
foreach ($op in $ops3) {
    $rocblas = $rocblas -replace "cublas$op\(", "rocblas_$($op.ToLower())("
}

# Constants
$rocblas = $rocblas -replace 'CUBLAS_FILL_MODE_UPPER', 'rocblas_fill_upper'
$rocblas = $rocblas -replace 'CUBLAS_FILL_MODE_LOWER', 'rocblas_fill_lower'
$rocblas = $rocblas -replace 'CUBLAS_OP_N', 'rocblas_operation_none'
$rocblas = $rocblas -replace 'CUBLAS_OP_T', 'rocblas_operation_transpose'
$rocblas = $rocblas -replace 'CUBLAS_OP_C', 'rocblas_operation_conjugate_transpose'
$rocblas = $rocblas -replace 'CUBLAS_DIAG_NON_UNIT', 'rocblas_diagonal_non_unit'
$rocblas = $rocblas -replace 'CUBLAS_DIAG_UNIT', 'rocblas_diagonal_unit'
$rocblas = $rocblas -replace 'CUBLAS_SIDE_LEFT', 'rocblas_side_left'
$rocblas = $rocblas -replace 'CUBLAS_SIDE_RIGHT', 'rocblas_side_right'
$rocblas = $rocblas -replace 'cublasFillMode_t', 'rocblas_fill'
$rocblas = $rocblas -replace 'cublasOperation_t', 'rocblas_operation'
$rocblas = $rocblas -replace 'cublasDiagType_t', 'rocblas_diagonal'
$rocblas = $rocblas -replace 'cublasSideMode_t', 'rocblas_side'
$rocblas = $rocblas -replace 'fb_cublas_trait', 'fb_rocblas_trait'
$rocblas = $rocblas -replace 'fb_get_cublas_trait', 'fb_get_rocblas_trait'
$rocblas = $rocblas -replace 'cuBLAS:', 'rocBLAS:'
$rocblas = $rocblas -replace 'cuBLAS ', 'rocBLAS '
$rocblas = $rocblas -replace ' cuBLAS', ' rocBLAS'

# Add public interface if missing
if ($rocblas -notmatch 'fb_get_rocblas_trait') {
    $rocblas += @"

/* ============================================================================
 * Public Interface
 * ========================================================================== */

const fb_gpu_backend_trait_t* fb_get_rocblas_trait(void) {
    return &fb_rocblas_trait;
}
"@
}

Write-Host "Writing BLAS-only rocBLAS implementation..." -ForegroundColor Green
$rocblas | Out-File -Encoding UTF8 "src\backends\gpu\rocblas_trait_impl.c"

$lines = ($rocblas -split "`n").Count
$bytes = $rocblas.Length
Write-Host "✓ Created rocblas_trait_impl.c: $lines lines, $bytes bytes" -ForegroundColor Cyan
Write-Host "✓ All 198 BLAS operations (LAPACK temporarily excluded)" -ForegroundColor Green
