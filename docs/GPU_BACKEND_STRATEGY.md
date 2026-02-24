# GPU Backend Implementation Strategy

## Overview

GPU backends (cuBLAS, rocBLAS) require fundamentally different architecture than CPU backends due to:
- Device memory (VRAM) vs host memory (RAM)
- Asynchronous execution model
- Different API signatures
- Explicit memory management
- Stream/queue management

##  Key Architectural Differences

### CPU Backends (OpenBLAS, MKL, BLIS, etc.)
```c
// Direct host memory operations
void cpu_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    cblas_saxpy(n, alpha, x, incx, y, incy);  // Synchronous, host memory
}
```

### GPU Backends (cuBLAS, rocBLAS)
```c
// Requires device memory and async handling
void gpu_saxpy(int n, float alpha, const float* x_host, int incx, float* y_host, int incy) {
    float *x_dev, *y_dev;
    
    // 1. Allocate device memory
    cudaMalloc(&x_dev, n * sizeof(float));
    cudaMalloc(&y_dev, n * sizeof(float));
    
    // 2. Copy host -> device
    cudaMemcpy(x_dev, x_host, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y_host, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // 3. Execute on GPU (asynchronous)
    cublasSaxpy(handle, n, &alpha, x_dev, incx, y_dev, incy);
    
    // 4. Copy device -> host
    cudaMemcpy(y_host, y_dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 5. Free device memory
    cudaFree(x_dev);
    cudaFree(y_dev);
    
    // 6. Synchronize
    cudaDeviceSynchronize();
}
```

## Implementation Options

### Option 1: Automatic Memory Management (Transparent)
**Pros**: Easy to use, no API changes
**Cons**: Performance overhead from constant host ↔ device transfers

```c
// User calls with host pointers - library handles everything
float x[1000], y[1000];
fb_saxpy(1000, 2.0f, x, 1, y, 1);  // Works, but slow due to transfers
```

### Option 2: Explicit Device Memory (High Performance)
**Pros**: No unnecessary transfers, maximum performance
**Cons**: User must manage device memory explicitly

```c
// User manages device memory
fb_gpu_ptr_t x_dev = fb_gpu_malloc(1000 * sizeof(float));
fb_gpu_ptr_t y_dev = fb_gpu_malloc(1000 * sizeof(float));

fb_gpu_memcpy_h2d(x_dev, x_host, 1000 * sizeof(float));
fb_gpu_memcpy_h2d(y_dev, y_host, 1000 * sizeof(float));

fb_gpu_saxpy(1000, 2.0f, x_dev, 1, y_dev, 1);  // Fast - no transfers

fb_gpu_memcpy_d2h(y_host, y_dev, 1000 * sizeof(float));
fb_gpu_free(x_dev);
fb_gpu_free(y_dev);
```

### Option 3: Hybrid Approach (Recommended)
**Pros**: Flexible, allows both use cases
**Cons**: More complex API

```c
// Separate APIs for CPU and GPU
fb_cpu_saxpy(...);   // Uses CPU backend (OpenBLAS, MKL, etc.)
fb_gpu_saxpy(...);   // Uses GPU backend (cuBLAS, rocBLAS, etc.)

// Or backend selection
fb_set_backend(FB_BACKEND_CUBLAS);
fb_saxpy(...);  // Uses selected backend
```

## Recommended Implementation Strategy

Given the fundamental differences, we should implement GPU backends as a **separate but parallel** system:

### 1. Separate Header Structure
```
include/faster-blaster/
├── blas.h              # CPU BLAS operations
├── lapack.h            # CPU LAPACK operations  
├── gpu_blas.h          # GPU BLAS operations
├── gpu_lapack.h        # GPU LAPACK operations
└── backends/
    ├── cpu/            # CPU backend headers
    └── gpu/            # GPU backend headers
```

### 2. Explicit GPU API
```c
// GPU-specific operations that work with device pointers
typedef void* fb_gpu_ptr_t;

// Memory management
fb_gpu_ptr_t fb_gpu_malloc(size_t size);
void fb_gpu_free(fb_gpu_ptr_t ptr);
int fb_gpu_memcpy_h2d(fb_gpu_ptr_t dst, const void* src, size_t size);
int fb_gpu_memcpy_d2h(void* dst, fb_gpu_ptr_t src, size_t size);

// BLAS operations (work on device pointers)
void fb_gpu_saxpy(int n, float alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
void fb_gpu_sgemm(char transa, char transb, int m, int n, int k,
                  float alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb,
                  float beta, fb_gpu_ptr_t c, int ldc);
```

### 3. Stream/Async Support
```c
typedef void* fb_gpu_stream_t;

fb_gpu_stream_t fb_gpu_stream_create(void);
void fb_gpu_stream_destroy(fb_gpu_stream_t stream);
void fb_gpu_stream_synchronize(fb_gpu_stream_t stream);

// Async operations
void fb_gpu_saxpy_async(fb_gpu_stream_t stream, int n, float alpha, 
                        fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
```

## Implementation Plan

### Phase 1: Core Infrastructure (DONE ✅)
- [x] GPU backend loading system
- [x] Device enumeration
- [x] Context management
- [x] Memory allocation/deallocation
- [x] Host ↔ Device transfers

### Phase 2: cuBLAS Level 1 Operations (~54 operations)
Implement all Level 1 BLAS with device memory handling:
- SASUM, DASUM, CASUM, ZASUM
- SAXPY, DAXPY, CAXPY, ZAXPY
- SCOPY, DCOPY, CCOPY, ZCOPY
- SDOT, DDOT, CDOTU, ZDOTU, CDOTC, ZDOTC
- SSCAL, DSCAL, CSCAL, ZSCAL, CSSCAL, ZDSCAL
- SNRM2, DNRM2, SCNRM2, DZNRM2
- SSWAP, DSWAP, CSWAP, ZSWAP
- ISAMAX, IDAMAX, ICAMAX, IZAMAX
- SROT, DROT, CROT, ZROT
- SROTG, DROTG, CROTG, ZROTG
- SROTM, DROTM, SROTMG, DROTMG

### Phase 3: cuBLAS Level 2 Operations (~70 operations)
- SGEMV, DGEMV, CGEMV, ZGEMV
- SGBMV, DGBMV, CGBMV, ZGBMV
- SSYMV, DSYMV, CSYMV, ZSYMV
- CHEMV, ZHEMV, CHBMV, ZHBMV, CHPMV, ZHPMV
- STRMV, DTRMV, CTRMV, ZTRMV
- STRSV, DTRSV, CTRSV, ZTRSV
- SGER, DGER, CGERU, ZGERU, CGERC, ZGERC
- SSYR, DSYR, CHER, ZHER
- SSYR2, DSYR2, CHER2, ZHER2

### Phase 4: cuBLAS Level 3 Operations (~28 operations)
- SGEMM, DGEMM, CGEMM, ZGEMM
- SSYMM, DSYMM, CSYMM, ZSYMM
- CHEMM, ZHEMM
- SSYRK, DSYRK, CSYRK, ZSYRK
- CHERK, ZHERK
- SSYR2K, DSYR2K, CSYR2K, ZSYR2K
- CHER2K, ZHER2K
- STRMM, DTRMM, CTRMM, ZTRMM
- STRSM, DTRSM, CTRSM, ZTRSM

### Phase 5: cuSOLVER LAPACK Operations (~60 operations)
cuSOLVER has different API than LAPACKE:
- SGESV → cusolverDnSgesv
- SPOTRF → cusolverDnSpotrf
- SGEQRF → cusolverDnSgeqrf
- SGESVD → cusolverDnSgesvd
- SSYEV → cusolverDnSsyevd

**Key difference**: cuSOLVER requires workspace queries

### Phase 6: rocBLAS/rocSOLVER Backend
Can largely copy cuBLAS implementation with API adjustments:
- `cublas` → `rocblas`
- `cusolver` → `rocsolver`
- Similar enum/structure patterns

## Code Example: Proper cuBLAS Wrapper

```c
// File: src/backends/gpu/cublas_operations.c

#include "cublas_backend.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Global cuBLAS handle (initialized in fb_cublas_init)
static cublasHandle_t g_cublas_handle = NULL;

void fb_gpu_saxpy(int n, float alpha, fb_gpu_ptr_t x, int incx, 
                  fb_gpu_ptr_t y, int incy) {
    // x and y are already device pointers - no transfer needed
    cublasStatus_t status = cublasSaxpy(
        g_cublas_handle,
        n,
        &alpha,
        (const float*)x,
        incx,
        (float*)y,
        incy
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        // Error handling
        fprintf(stderr, "cuBLAS saxpy failed: %d\n", status);
    }
}

void fb_gpu_sgemm(char transa, char transb, int m, int n, int k,
                  float alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb,
                  float beta, fb_gpu_ptr_t c, int ldc) {
    cublasOperation_t transA = (transa == 'N') ? CUBLAS_OP_N : 
                               (transa == 'T') ? CUBLAS_OP_T : CUBLAS_OP_C;
    cublasOperation_t transB = (transb == 'N') ? CUBLAS_OP_N : 
                               (transb == 'T') ? CUBLAS_OP_T : CUBLAS_OP_C;
    
    cublasStatus_t status = cublasSgemm(
        g_cublas_handle,
        transA,
        transB,
        m, n, k,
        &alpha,
        (const float*)a, lda,
        (const float*)b, ldb,
        &beta,
        (float*)c, ldc
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS sgemm failed: %d\n", status);
    }
}
```

## Testing Strategy

### Unit Tests
```c
void test_gpu_saxpy(void) {
    const int n = 1000;
    float alpha = 2.0f;
    
    // Host data
    float *x_host = malloc(n * sizeof(float));
    float *y_host = malloc(n * sizeof(float));
    float *y_expected = malloc(n * sizeof(float));
    
    // Initialize
    for (int i = 0; i < n; i++) {
        x_host[i] = i;
        y_host[i] = i * 2;
        y_expected[i] = y_host[i] + alpha * x_host[i];
    }
    
    // GPU execution
    fb_gpu_ptr_t x_dev = fb_gpu_malloc(n * sizeof(float));
    fb_gpu_ptr_t y_dev = fb_gpu_malloc(n * sizeof(float));
    
    fb_gpu_memcpy_h2d(x_dev, x_host, n * sizeof(float));
    fb_gpu_memcpy_h2d(y_dev, y_host, n * sizeof(float));
    
    fb_gpu_saxpy(n, alpha, x_dev, 1, y_dev, 1);
    
    fb_gpu_memcpy_d2h(y_host, y_dev, n * sizeof(float));
    
    // Verify
    for (int i = 0; i < n; i++) {
        assert(fabs(y_host[i] - y_expected[i]) < 1e-5);
    }
    
    fb_gpu_free(x_dev);
    fb_gpu_free(y_dev);
    free(x_host);
    free(y_host);
    free(y_expected);
}
```

## Timeline Estimate

| Phase                | Operations      | Estimated Time  |
| -------------------- | --------------- | --------------- |
| Infrastructure       | ✅ Done          | -               |
| cuBLAS Level 1       | 54 ops          | 8-12 hours      |
| cuBLAS Level 2       | 70 ops          | 12-16 hours     |
| cuBLAS Level 3       | 28 ops          | 8-10 hours      |
| cuSOLVER LAPACK      | 60 ops          | 16-20 hours     |
| rocBLAS (copy-adapt) | 212 ops         | 8-12 hours      |
| Testing & Debug      | All             | 10-15 hours     |
| **Total**            | **424 GPU ops** | **60-85 hours** |

## Next Steps

1. **Decision Point**: Choose implementation approach (Option 1, 2, or 3)
2. **API Design**: Finalize GPU-specific API headers
3. **Implement cuBLAS Level 1**: Start with simplest operations
4. **Iterative Testing**: Test each level before moving to next
5. **rocBLAS Port**: Adapt cuBLAS to rocBLAS once pattern established

## Recommendation

Start with **Option 3 (Hybrid)** and implement **Phase 2 (cuBLAS Level 1)** first. This will:
- Establish patterns for device memory handling
- Provide immediate value (vector operations are common)
- Allow early testing and validation
- Inform decisions about more complex operations

Would you like me to proceed with implementing cuBLAS Level 1 operations following this strategy?
