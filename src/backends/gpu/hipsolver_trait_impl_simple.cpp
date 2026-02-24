/**
 * @file hipsolver_trait_impl_simple.cpp
 * @brief Simplified hipSOLVER-based LAPACK implementation (NVIDIA + AMD unified)
 * 
 * This demonstrates the zero-cost abstraction concept with minimal LAPACK operations.
 * Once this builds successfully, we can expand to the full API.
 */

#include "faster-blaster/gpu_backend_trait.h"
#include <hip/hip_runtime.h>
#include <hipsolver/hipsolver.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Lifecycle Management
 * ========================================================================== */

static int hipsolver_init(void** ctx) {
    hipsolverHandle_t* solver = (hipsolverHandle_t*)malloc(sizeof(hipsolverHandle_t));
    if (!solver) return -1;
    
    hipsolverStatus_t status = hipsolverCreate(solver);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
        free(solver);
        return (int)status;
    }
    
    *ctx = solver;
    return 0;
}

static int hipsolver_shutdown(void* ctx) {
    if (!ctx) return -1;
    
    hipsolverHandle_t* solver = (hipsolverHandle_t*)ctx;
    hipsolverStatus_t status = hipsolverDestroy(*solver);
    free(solver);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Operations - LU Factorization
 * ========================================================================== */

static int hipsolver_sgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, int* info_host) {
    hipsolverHandle_t solver = *(hipsolverHandle_t*)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    // Query workspace size
    int lwork = 0;
    status = hipsolverSgetrf_bufferSize(solver, m, n, (float*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    // Allocate workspace
    float* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(float));
    }
    
    // Allocate device info
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    // Perform LU factorization
    status = hipsolverSgetrf(solver, m, n, (float*)a, lda,
                             workspace, lwork, (int*)ipiv, info_device);
    
    // Copy info back to host
    if (info_host) {
        hipMemcpy(info_host, info_device, sizeof(int), hipMemcpyDeviceToHost);
    }
    
    // Cleanup
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dgetrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t ipiv, int* info_host) {
    hipsolverHandle_t solver = *(hipsolverHandle_t*)handle;
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
    
    status = hipsolverDgetrf(solver, m, n, (double*)a, lda,
                             workspace, lwork, (int*)ipiv, info_device);
    
    if (info_host) {
        hipMemcpy(info_host, info_device, sizeof(int), hipMemcpyDeviceToHost);
    }
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Operations - Cholesky Factorization
 * ========================================================================== */

static int hipsolver_spotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                             fb_gpu_ptr_t a, int lda, int* info_host) {
    hipsolverHandle_t solver = *(hipsolverHandle_t*)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U' || uplo == 'u') ? 
                                HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverSpotrf_bufferSize(solver, fill, n, (float*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    float* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(float));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverSpotrf(solver, fill, n, (float*)a, lda,
                             workspace, lwork, info_device);
    
    if (info_host) {
        hipMemcpy(info_host, info_device, sizeof(int), hipMemcpyDeviceToHost);
    }
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dpotrf(void* handle, fb_gpu_stream_t stream, char uplo, int n,
                             fb_gpu_ptr_t a, int lda, int* info_host) {
    hipsolverHandle_t solver = *(hipsolverHandle_t*)handle;
    hipsolverStatus_t status;
    
    if (stream) {
        hipsolverSetStream(solver, (hipStream_t)stream);
    }
    
    hipsolverFillMode_t fill = (uplo == 'U' || uplo == 'u') ? 
                                HIPSOLVER_FILL_MODE_UPPER : HIPSOLVER_FILL_MODE_LOWER;
    
    int lwork = 0;
    status = hipsolverDpotrf_bufferSize(solver, fill, n, (double*)a, lda, &lwork);
    if (status != HIPSOLVER_STATUS_SUCCESS) return (int)status;
    
    double* workspace = NULL;
    if (lwork > 0) {
        hipMalloc(&workspace, lwork * sizeof(double));
    }
    
    int* info_device = NULL;
    hipMalloc(&info_device, sizeof(int));
    
    status = hipsolverDpotrf(solver, fill, n, (double*)a, lda,
                             workspace, lwork, info_device);
    
    if (info_host) {
        hipMemcpy(info_host, info_device, sizeof(int), hipMemcpyDeviceToHost);
    }
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * LAPACK Operations - QR Factorization
 * ========================================================================== */

static int hipsolver_sgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau, int* info_host) {
    hipsolverHandle_t solver = *(hipsolverHandle_t*)handle;
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
    
    status = hipsolverSgeqrf(solver, m, n, (float*)a, lda, (float*)tau,
                             workspace, lwork, info_device);
    
    if (info_host) {
        hipMemcpy(info_host, info_device, sizeof(int), hipMemcpyDeviceToHost);
    }
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

static int hipsolver_dgeqrf(void* handle, fb_gpu_stream_t stream, int m, int n,
                             fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t tau, int* info_host) {
    hipsolverHandle_t solver = *(hipsolverHandle_t*)handle;
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
    
    status = hipsolverDgeqrf(solver, m, n, (double*)a, lda, (double*)tau,
                             workspace, lwork, info_device);
    
    if (info_host) {
        hipMemcpy(info_host, info_device, sizeof(int), hipMemcpyDeviceToHost);
    }
    
    if (workspace) hipFree(workspace);
    hipFree(info_device);
    
    return (status == HIPSOLVER_STATUS_SUCCESS) ? 0 : (int)status;
}

/* ============================================================================
 * Trait Export  (Note: This is a minimal subset for testing)
 * ========================================================================== */

extern "C" {
    // For demonstration purposes only - real trait would have full vtable
    int hipsolver_test_init(void** ctx) { return hipsolver_init(ctx); }
    int hipsolver_test_shutdown(void* ctx) { return hipsolver_shutdown(ctx); }
    int hipsolver_test_sgetrf(void* h, void* s, int m, int n, void* a, int lda, void* ipiv, int* info) {
        return hipsolver_sgetrf(h, s, m, n, a, lda, ipiv, info);
    }
    int hipsolver_test_spotrf(void* h, void* s, char uplo, int n, void* a, int lda, int* info) {
        return hipsolver_spotrf(h, s, uplo, n, a, lda, info);
    }
    int hipsolver_test_sgeqrf(void* h, void* s, int m, int n, void* a, int lda, void* tau, int* info) {
        return hipsolver_sgeqrf(h, s, m, n, a, lda, tau, info);
    }
}
