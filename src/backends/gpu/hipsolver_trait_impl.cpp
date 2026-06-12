/**
 * @file hipsolver_trait_impl.c
 * @brief hipSOLVER-based LAPACK implementation (NVIDIA + AMD unified)
 * 
 * This implementation uses AMD's hipSOLVER library, which provides a unified
 * API for LAPACK operations that compiles to:
 *   - cuSOLVER (NVIDIA GPUs)
 *   - rocSOLVER (AMD GPUs)
 * 
 * Zero-cost abstraction: Single codebase, compile-time backend selection.
 * No runtime overhead vs calling cuSOLVER/rocSOLVER directly.
 * 
 * Build:
 *   NVIDIA: nvcc with -DHIPSOLVER_TARGET_CUDA -lcusolver
 *   AMD:    hipcc with -DHIPSOLVER_TARGET_ROCM -lrocsolver
 */

#include "faster-blaster/gpu_backend_trait.h"
#include <hip/hip_runtime.h>
#include <hipsolver/hipsolver.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * LAPACK Operations - LU Factorization
 * ========================================================================== */

static int hipsolver_sgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    // Set stream
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    // Allocate workspace
    int lwork = 0;
    status = hipsolverSgetrf_bufferSize(solver, m, n, (float*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    float* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(float));
    }
    
    // Allocate device info
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    // Perform LU factorization
    status = hipsolverSgetrf(solver, m, n, (float*)a, lda, workspace, lwork, (int*)ipiv, info_device);
    
    // Cleanup
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverDgetrf_bufferSize(solver, m, n, (double*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    double* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(double));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverDgetrf(solver, m, n, (double*)a, lda, workspace, lwork,
                             (int*)ipiv, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_cgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverCgetrf_bufferSize(solver, m, n, (hipFloatComplex*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipFloatComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipFloatComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverCgetrf(solver, m, n, (hipFloatComplex*)a, lda, workspace, lwork,
                             (int*)ipiv, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_zgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverZgetrf_bufferSize(solver, m, n, (hipDoubleComplex*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipDoubleComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipDoubleComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverZgetrf(solver, m, n, (hipDoubleComplex*)a, lda, workspace, lwork,
                             (int*)ipiv, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Operations - LU Solve
 * ========================================================================== */

static int hipsolver_sgetrs(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv,
                             fb_gpu_ptr_t b, int ldb) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    // Convert transpose flag
    hipsolverOperation_t op = (trans == 'N') ? HIPSOLVER_OP_N :
                              (trans == 'T') ? HIPSOLVER_OP_T : HIPSOLVER_OP_C;
    
    // Query workspace size
    int lwork = 0;
    status = hipsolverSgetrs_bufferSize(solver, op, n, nrhs, (float*)a, lda, (int*)ipiv,
                                         (float*)b, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    // Allocate workspace
    float* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(float));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverSgetrs(solver, op, n, nrhs, (float*)a, lda,
                             (int*)ipiv, (float*)b, ldb, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dgetrs(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv,
                             fb_gpu_ptr_t b, int ldb) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverOperation_t op = (trans == 'N') ? HIPSOLVER_OP_N :
                              (trans == 'T') ? HIPSOLVER_OP_T : HIPSOLVER_OP_C;
    
    int lwork = 0;
    status = hipsolverDgetrs_bufferSize(solver, op, n, nrhs, (double*)a, lda, (int*)ipiv,
                                         (double*)b, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    double* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(double));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverDgetrs(solver, op, n, nrhs, (double*)a, lda,
                             (int*)ipiv, (double*)b, ldb, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_cgetrs(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv,
                             fb_gpu_ptr_t b, int ldb) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverOperation_t op = (trans == 'N') ? HIPSOLVER_OP_N :
                              (trans == 'T') ? HIPSOLVER_OP_T : HIPSOLVER_OP_C;
    
    int lwork = 0;
    status = hipsolverCgetrs_bufferSize(solver, op, n, nrhs, (hipFloatComplex*)a, lda, (int*)ipiv,
                                         (hipFloatComplex*)b, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipFloatComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipFloatComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverCgetrs(solver, op, n, nrhs, (hipFloatComplex*)a, lda,
                             (int*)ipiv, (hipFloatComplex*)b, ldb, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_zgetrs(void* handle, fb_gpu_stream_t stream, char trans, int n, int nrhs,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv,
                             fb_gpu_ptr_t b, int ldb) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverOperation_t op = (trans == 'N') ? HIPSOLVER_OP_N :
                              (trans == 'T') ? HIPSOLVER_OP_T : HIPSOLVER_OP_C;
    
    int lwork = 0;
    status = hipsolverZgetrs_bufferSize(solver, op, n, nrhs, (hipDoubleComplex*)a, lda, (int*)ipiv,
                                         (hipDoubleComplex*)b, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipDoubleComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipDoubleComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverZgetrs(solver, op, n, nrhs, (hipDoubleComplex*)a, lda,
                             (int*)ipiv, (hipDoubleComplex*)b, ldb, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Operations - Cholesky Factorization
 * ========================================================================== */

static int hipsolver_spotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                             fb_gpu_ptr_t a, int lda) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverSpotrf_bufferSize(solver, fill, n, (float*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    float* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(float));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverSpotrf(solver, fill, n, (float*)a, lda, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dpotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                             fb_gpu_ptr_t a, int lda) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverDpotrf_bufferSize(solver, fill, n, (double*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    double* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(double));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverDpotrf(solver, fill, n, (double*)a, lda, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_cpotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                             fb_gpu_ptr_t a, int lda) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverCpotrf_bufferSize(solver, fill, n, (hipFloatComplex*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipFloatComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipFloatComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverCpotrf(solver, fill, n, (hipFloatComplex*)a, lda, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_zpotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                             fb_gpu_ptr_t a, int lda) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverZpotrf_bufferSize(solver, fill, n, (hipDoubleComplex*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipDoubleComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipDoubleComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverZpotrf(solver, fill, n, (hipDoubleComplex*)a, lda, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Operations - Cholesky Solve
 * ========================================================================== */

static int hipsolver_spotrs(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverSpotrs_bufferSize(solver, fill, n, nrhs, (float*)a, lda, (float*)b, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    float* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(float));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverSpotrs(solver, fill, n, nrhs, (float*)a, lda, (float*)b, ldb,
                             workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dpotrs(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverDpotrs_bufferSize(solver, fill, n, nrhs, (double*)a, lda, (double*)b, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    double* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(double));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverDpotrs(solver, fill, n, nrhs, (double*)a, lda, (double*)b, ldb,
                             workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_cpotrs(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverCpotrs_bufferSize(solver, fill, n, nrhs, (hipFloatComplex*)a, lda,
                                         (hipFloatComplex*)b, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipFloatComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipFloatComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverCpotrs(solver, fill, n, nrhs, (hipFloatComplex*)a, lda,
                             (hipFloatComplex*)b, ldb, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_zpotrs(void* handle, fb_gpu_stream_t stream, char uplo, int n, int nrhs,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverZpotrs_bufferSize(solver, fill, n, nrhs, (hipDoubleComplex*)a, lda,
                                         (hipDoubleComplex*)b, ldb, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipDoubleComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipDoubleComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverZpotrs(solver, fill, n, nrhs, (hipDoubleComplex*)a, lda,
                             (hipDoubleComplex*)b, ldb, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Operations - QR Factorization
 * ========================================================================== */

static int hipsolver_sgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverSgeqrf_bufferSize(solver, m, n, (float*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    float* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(float));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverSgeqrf(solver, m, n, (float*)a, lda, (float*)tau, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverDgeqrf_bufferSize(solver, m, n, (double*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    double* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(double));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverDgeqrf(solver, m, n, (double*)a, lda, (double*)tau, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_cgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverCgeqrf_bufferSize(solver, m, n, (hipFloatComplex*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipFloatComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipFloatComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverCgeqrf(solver, m, n, (hipFloatComplex*)a, lda,
                             (hipFloatComplex*)tau, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_zgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverZgeqrf_bufferSize(solver, m, n, (hipDoubleComplex*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipDoubleComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipDoubleComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverZgeqrf(solver, m, n, (hipDoubleComplex*)a, lda,
                             (hipDoubleComplex*)tau, workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Operations - SVD
 * ========================================================================== */

static int hipsolver_sgesvd(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt,
                             int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                             fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverSgesvd_bufferSize(solver, jobu, jobvt, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    float* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(float));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverSgesvd(solver, jobu, jobvt, m, n, (float*)a, lda, (float*)s,
                             (float*)u, ldu, (float*)vt, ldvt, workspace, lwork,
                             NULL, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dgesvd(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt,
                             int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                             fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverDgesvd_bufferSize(solver, jobu, jobvt, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    double* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(double));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverDgesvd(solver, jobu, jobvt, m, n, (double*)a, lda, (double*)s,
                             (double*)u, ldu, (double*)vt, ldvt, workspace, lwork,
                             NULL, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_cgesvd(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt,
                             int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                             fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverCgesvd_bufferSize(solver, jobu, jobvt, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipFloatComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipFloatComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverCgesvd(solver, jobu, jobvt, m, n, (hipFloatComplex*)a, lda, (float*)s,
                             (hipFloatComplex*)u, ldu, (hipFloatComplex*)vt, ldvt,
                             workspace, lwork, NULL, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_zgesvd(void* handle, fb_gpu_stream_t stream, char jobu, char jobvt,
                             int m, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t s,
                             fb_gpu_ptr_t u, int ldu, fb_gpu_ptr_t vt, int ldvt) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    int lwork = 0;
    status = hipsolverZgesvd_bufferSize(solver, jobu, jobvt, m, n, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipDoubleComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipDoubleComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverZgesvd(solver, jobu, jobvt, m, n, (hipDoubleComplex*)a, lda, (double*)s,
                             (hipDoubleComplex*)u, ldu, (hipDoubleComplex*)vt, ldvt,
                             workspace, lwork, NULL, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Operations - Eigenvalues (Symmetric/Hermitian)
 * ========================================================================== */

static int hipsolver_ssyev(void* handle, fb_gpu_stream_t stream, char jobz, char uplo,
                            int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverEigMode_t jobz_mode = (jobz == 'V') ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverSsyevd_bufferSize(solver, jobz_mode, fill, n, (float*)a, lda, (float*)w, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    float* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(float));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverSsyevd(solver, jobz_mode, fill, n, (float*)a, lda, (float*)w,
                             workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dsyev(void* handle, fb_gpu_stream_t stream, char jobz, char uplo,
                            int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverEigMode_t jobz_mode = (jobz == 'V') ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverDsyevd_bufferSize(solver, jobz_mode, fill, n, (double*)a, lda, (double*)w, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    double* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(double));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverDsyevd(solver, jobz_mode, fill, n, (double*)a, lda, (double*)w,
                             workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_cheev(void* handle, fb_gpu_stream_t stream, char jobz, char uplo,
                            int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverEigMode_t jobz_mode = (jobz == 'V') ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverCheevd_bufferSize(solver, jobz_mode, fill, n, (hipFloatComplex*)a, lda,
                                        (float*)w, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipFloatComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipFloatComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverCheevd(solver, jobz_mode, fill, n, (hipFloatComplex*)a, lda, (float*)w,
                             workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_zheev(void* handle, fb_gpu_stream_t stream, char jobz, char uplo,
                            int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t w) {
    hipsolverHandle_t solver = (hipsolverHandle_t)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverEigMode_t jobz_mode = (jobz == 'V') ? HIPSOLVER_EIG_MODE_VECTOR : HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverFillMode_t fill = (uplo == 'U') ? HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverZheevd_bufferSize(solver, jobz_mode, fill, n, (hipDoubleComplex*)a, lda,
                                        (double*)w, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    hipDoubleComplex* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(hipDoubleComplex));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverZheevd(solver, jobz_mode, fill, n, (hipDoubleComplex*)a, lda, (double*)w,
                             workspace, lwork, info_device);
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Trait Vtable Partial (LAPACK operations only)
 * Note: Full trait includes BLAS operations - see cublas_trait_impl.c or rocblas_trait_impl.c
 * ========================================================================== */

/**
 * @brief hipSOLVER LAPACK operations vtable
 * 
 * This can be used standalone or combined with cuBLAS/rocBLAS for full backend.
 * Compile with -DHIPSOLVER_TARGET_CUDA for NVIDIA or -DHIPSOLVER_TARGET_ROCM for AMD.
 */

extern "C" {

const fb_gpu_backend_trait_t fb_hipsolver_lapack_trait = {
    .name = "hipSOLVER LAPACK (NVIDIA + AMD unified)",
    .type = FB_GPU_BACKEND_CUBLAS, // Can be CUBLAS or ROCBLAS depending on compile target
    
    /* LAPACK operations */
    .sgetrf = hipsolver_sgetrf,
    .dgetrf = hipsolver_dgetrf,
    .cgetrf = hipsolver_cgetrf,
    .zgetrf = hipsolver_zgetrf,
    
    .sgetrs = hipsolver_sgetrs,
    .dgetrs = hipsolver_dgetrs,
    .cgetrs = hipsolver_cgetrs,
    .zgetrs = hipsolver_zgetrs,
    
    .spotrf = hipsolver_spotrf,
    .dpotrf = hipsolver_dpotrf,
    .cpotrf = hipsolver_cpotrf,
    .zpotrf = hipsolver_zpotrf,
    
    .spotrs = hipsolver_spotrs,
    .dpotrs = hipsolver_dpotrs,
    .cpotrs = hipsolver_cpotrs,
    .zpotrs = hipsolver_zpotrs,
    
    .sgeqrf = hipsolver_sgeqrf,
    .dgeqrf = hipsolver_dgeqrf,
    .cgeqrf = hipsolver_cgeqrf,
    .zgeqrf = hipsolver_zgeqrf,
    
    .sgesvd = hipsolver_sgesvd,
    .dgesvd = hipsolver_dgesvd,
    .cgesvd = hipsolver_cgesvd,
    .zgesvd = hipsolver_zgesvd,
    
    .ssyev = hipsolver_ssyev,
    .dsyev = hipsolver_dsyev,
    .cheev = hipsolver_cheev,
    .zheev = hipsolver_zheev,
};

} // extern "C"
