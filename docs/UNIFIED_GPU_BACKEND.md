# Unified GPU Backend Architecture

## Concept

Create a **single GPU backend trait** that abstracts over cuBLAS, rocBLAS, and other GPU BLAS implementations, allowing transparent multi-GPU execution across vendors.

## Architecture

### Core Abstraction Layer

```c
// File: include/faster-blaster/gpu_backend.h

typedef enum {
    FB_GPU_BACKEND_CUBLAS,    // NVIDIA CUDA
    FB_GPU_BACKEND_ROCBLAS,   // AMD ROCm
    FB_GPU_BACKEND_ONEMKL,    // Intel oneAPI
    FB_GPU_BACKEND_OPENBLAS_GPU // OpenBLAS GPU (if available)
} fb_gpu_backend_type_t;

typedef struct fb_gpu_context fb_gpu_context_t;
typedef void* fb_gpu_ptr_t;
typedef void* fb_gpu_stream_t;

// Unified GPU operations (works with any backend)
typedef struct {
    // Level 1 BLAS
    void (*saxpy)(fb_gpu_context_t* ctx, fb_gpu_stream_t stream,
                  int n, float alpha, fb_gpu_ptr_t x, int incx, 
                  fb_gpu_ptr_t y, int incy);
    
    void (*daxpy)(fb_gpu_context_t* ctx, fb_gpu_stream_t stream,
                  int n, double alpha, fb_gpu_ptr_t x, int incx, 
                  fb_gpu_ptr_t y, int incy);
    
    // Level 2 BLAS
    void (*sgemv)(fb_gpu_context_t* ctx, fb_gpu_stream_t stream,
                  char trans, int m, int n, float alpha,
                  fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                  float beta, fb_gpu_ptr_t y, int incy);
    
    // Level 3 BLAS
    void (*sgemm)(fb_gpu_context_t* ctx, fb_gpu_stream_t stream,
                  char transa, char transb, int m, int n, int k,
                  float alpha, fb_gpu_ptr_t a, int lda,
                  fb_gpu_ptr_t b, int ldb, float beta,
                  fb_gpu_ptr_t c, int ldc);
    
    // ... all 212 operations with same signatures
} fb_gpu_blas_ops_t;

// GPU context (can represent NVIDIA, AMD, or Intel GPU)
struct fb_gpu_context {
    fb_gpu_backend_type_t backend_type;
    int device_id;
    void* backend_handle;  // cuBLAS handle, rocBLAS handle, etc.
    void* device_handle;   // CUDA device, HIP device, etc.
    fb_gpu_blas_ops_t* ops;
};
```

### Multi-GPU Manager

```c
// File: include/faster-blaster/gpu_manager.h

typedef struct {
    int num_devices;
    fb_gpu_context_t* contexts;  // One context per GPU
} fb_gpu_manager_t;

// Initialize multi-GPU manager (detects all GPUs)
fb_gpu_manager_t* fb_gpu_manager_init(void);

// Get context for specific GPU
fb_gpu_context_t* fb_gpu_get_context(fb_gpu_manager_t* mgr, int device_id);

// Execute operation on specific GPU
void fb_gpu_saxpy_on_device(fb_gpu_manager_t* mgr, int device_id,
                            int n, float alpha, fb_gpu_ptr_t x, int incx,
                            fb_gpu_ptr_t y, int incy);
```

## Your Use Case: AMD + NVIDIA GPUs

```c
// Example: Your PC with both AMD and NVIDIA GPUs

int main(void) {
    fb_gpu_manager_t* mgr = fb_gpu_manager_init();
    
    // Detect GPUs
    printf("Found %d GPUs:\n", mgr->num_devices);
    for (int i = 0; i < mgr->num_devices; i++) {
        fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, i);
        const char* backend_name = 
            (ctx->backend_type == FB_GPU_BACKEND_CUBLAS) ? "NVIDIA CUDA" :
            (ctx->backend_type == FB_GPU_BACKEND_ROCBLAS) ? "AMD ROCm" : "Unknown";
        printf("  GPU %d: %s\n", i, backend_name);
    }
    
    // GPU 0: NVIDIA (uses cuBLAS automatically)
    fb_gpu_context_t* nvidia_gpu = fb_gpu_get_context(mgr, 0);
    fb_gpu_ptr_t x_nvidia = fb_gpu_malloc(nvidia_gpu, 1000 * sizeof(float));
    fb_gpu_ptr_t y_nvidia = fb_gpu_malloc(nvidia_gpu, 1000 * sizeof(float));
    
    // GPU 1: AMD (uses rocBLAS automatically)
    fb_gpu_context_t* amd_gpu = fb_gpu_get_context(mgr, 1);
    fb_gpu_ptr_t x_amd = fb_gpu_malloc(amd_gpu, 1000 * sizeof(float));
    fb_gpu_ptr_t y_amd = fb_gpu_malloc(amd_gpu, 1000 * sizeof(float));
    
    // Same API, different GPUs - completely transparent!
    nvidia_gpu->ops->saxpy(nvidia_gpu, NULL, 1000, 2.0f, x_nvidia, 1, y_nvidia, 1);
    amd_gpu->ops->saxpy(amd_gpu, NULL, 1000, 2.0f, x_amd, 1, y_amd, 1);
    
    // Or use helper that picks GPU automatically
    fb_gpu_saxpy_on_device(mgr, 0, 1000, 2.0f, x_nvidia, 1, y_nvidia, 1);  // NVIDIA
    fb_gpu_saxpy_on_device(mgr, 1, 1000, 2.0f, x_amd, 1, y_amd, 1);       // AMD
    
    fb_gpu_manager_shutdown(mgr);
}
```

## Implementation Strategy

### Step 1: Define GPU Backend Trait

```c
// File: src/backends/gpu/gpu_backend_trait.c

typedef struct {
    // Backend identification
    const char* name;
    fb_gpu_backend_type_t type;
    
    // Initialization
    int (*init)(int device_id, void** backend_handle);
    void (*shutdown)(void* backend_handle);
    
    // Memory management
    int (*malloc)(void* backend_handle, fb_gpu_ptr_t* ptr, size_t size);
    void (*free)(void* backend_handle, fb_gpu_ptr_t ptr);
    int (*memcpy_h2d)(void* backend_handle, fb_gpu_ptr_t dst, const void* src, size_t size);
    int (*memcpy_d2h)(void* backend_handle, void* dst, fb_gpu_ptr_t src, size_t size);
    
    // Enum conversions (backend-specific)
    int (*convert_transpose)(char trans);
    int (*convert_uplo)(char uplo);
    int (*convert_diag)(char diag);
    int (*convert_side)(char side);
    
    // BLAS operations (all have same signature)
    void (*saxpy)(void* backend_handle, void* stream, int n, float alpha,
                  fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy);
    // ... all 212 operations
    
} fb_gpu_backend_trait_t;
```

### Step 2: Implement cuBLAS Backend

```c
// File: src/backends/gpu/cublas_trait_impl.c

#include <cublas_v2.h>

static int cublas_convert_transpose(char trans) {
    switch (trans) {
        case 'N': case 'n': return CUBLAS_OP_N;
        case 'T': case 't': return CUBLAS_OP_T;
        case 'C': case 'c': return CUBLAS_OP_C;
        default: return CUBLAS_OP_N;
    }
}

static void cublas_saxpy_impl(void* backend_handle, void* stream,
                               int n, float alpha, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    cublasHandle_t handle = (cublasHandle_t)backend_handle;
    
    if (stream) {
        cublasSetStream(handle, (cudaStream_t)stream);
    }
    
    cublasSaxpy(handle, n, &alpha, (const float*)x, incx, (float*)y, incy);
}

static void cublas_sgemm_impl(void* backend_handle, void* stream,
                               char transa, char transb, int m, int n, int k,
                               float alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb, float beta,
                               fb_gpu_ptr_t c, int ldc) {
    cublasHandle_t handle = (cublasHandle_t)backend_handle;
    cublasOperation_t opA = cublas_convert_transpose(transa);
    cublasOperation_t opB = cublas_convert_transpose(transb);
    
    if (stream) {
        cublasSetStream(handle, (cudaStream_t)stream);
    }
    
    cublasSgemm(handle, opA, opB, m, n, k, &alpha,
                (const float*)a, lda, (const float*)b, ldb,
                &beta, (float*)c, ldc);
}

// Export cuBLAS trait
const fb_gpu_backend_trait_t fb_cublas_trait = {
    .name = "NVIDIA cuBLAS",
    .type = FB_GPU_BACKEND_CUBLAS,
    .init = cublas_init_impl,
    .shutdown = cublas_shutdown_impl,
    .malloc = cublas_malloc_impl,
    .free = cublas_free_impl,
    .convert_transpose = cublas_convert_transpose,
    .saxpy = cublas_saxpy_impl,
    .sgemm = cublas_sgemm_impl,
    // ... all 212 operations
};
```

### Step 3: Implement rocBLAS Backend (Almost Identical!)

```c
// File: src/backends/gpu/rocblas_trait_impl.c

#include <rocblas/rocblas.h>

static int rocblas_convert_transpose(char trans) {
    switch (trans) {
        case 'N': case 'n': return rocblas_operation_none;
        case 'T': case 't': return rocblas_operation_transpose;
        case 'C': case 'c': return rocblas_operation_conjugate_transpose;
        default: return rocblas_operation_none;
    }
}

static void rocblas_saxpy_impl(void* backend_handle, void* stream,
                                int n, float alpha, fb_gpu_ptr_t x, int incx,
                                fb_gpu_ptr_t y, int incy) {
    rocblas_handle handle = (rocblas_handle)backend_handle;
    
    if (stream) {
        rocblas_set_stream(handle, (hipStream_t)stream);
    }
    
    rocblas_saxpy(handle, n, &alpha, (const float*)x, incx, (float*)y, incy);
}

static void rocblas_sgemm_impl(void* backend_handle, void* stream,
                                char transa, char transb, int m, int n, int k,
                                float alpha, fb_gpu_ptr_t a, int lda,
                                fb_gpu_ptr_t b, int ldb, float beta,
                                fb_gpu_ptr_t c, int ldc) {
    rocblas_handle handle = (rocblas_handle)backend_handle;
    rocblas_operation opA = rocblas_convert_transpose(transa);
    rocblas_operation opB = rocblas_convert_transpose(transb);
    
    if (stream) {
        rocblas_set_stream(handle, (hipStream_t)stream);
    }
    
    rocblas_sgemm(handle, opA, opB, m, n, k, &alpha,
                  (const float*)a, lda, (const float*)b, ldb,
                  &beta, (float*)c, ldc);
}

// Export rocBLAS trait
const fb_gpu_backend_trait_t fb_rocblas_trait = {
    .name = "AMD rocBLAS",
    .type = FB_GPU_BACKEND_ROCBLAS,
    .init = rocblas_init_impl,
    .shutdown = rocblas_shutdown_impl,
    .malloc = rocblas_malloc_impl,
    .free = rocblas_free_impl,
    .convert_transpose = rocblas_convert_transpose,
    .saxpy = rocblas_saxpy_impl,
    .sgemm = rocblas_sgemm_impl,
    // ... all 212 operations
};
```

### Step 4: Automatic Backend Detection

```c
// File: src/backends/gpu/gpu_backend_detector.c

static const fb_gpu_backend_trait_t* detect_gpu_backend(int device_id) {
    // Try CUDA first
    if (cuda_device_exists(device_id)) {
        return &fb_cublas_trait;
    }
    
    // Try ROCm/HIP
    if (hip_device_exists(device_id)) {
        return &fb_rocblas_trait;
    }
    
    // Try oneAPI
    if (sycl_device_exists(device_id)) {
        return &fb_onemkl_trait;
    }
    
    return NULL;
}

fb_gpu_manager_t* fb_gpu_manager_init(void) {
    fb_gpu_manager_t* mgr = calloc(1, sizeof(fb_gpu_manager_t));
    
    // Detect all GPUs (CUDA, ROCm, etc.)
    int cuda_devices = cuda_get_device_count();
    int rocm_devices = hip_get_device_count();
    
    mgr->num_devices = cuda_devices + rocm_devices;
    mgr->contexts = calloc(mgr->num_devices, sizeof(fb_gpu_context_t));
    
    int idx = 0;
    
    // Initialize CUDA devices
    for (int i = 0; i < cuda_devices; i++, idx++) {
        mgr->contexts[idx].backend_type = FB_GPU_BACKEND_CUBLAS;
        mgr->contexts[idx].device_id = i;
        mgr->contexts[idx].ops = &fb_cublas_trait;
        fb_cublas_trait.init(i, &mgr->contexts[idx].backend_handle);
    }
    
    // Initialize ROCm devices
    for (int i = 0; i < rocm_devices; i++, idx++) {
        mgr->contexts[idx].backend_type = FB_GPU_BACKEND_ROCBLAS;
        mgr->contexts[idx].device_id = i;
        mgr->contexts[idx].ops = &fb_rocblas_trait;
        fb_rocblas_trait.init(i, &mgr->contexts[idx].backend_handle);
    }
    
    return mgr;
}
```

## Advantages of This Approach

### 1. **Vendor Agnostic**
```c
// Same code works on NVIDIA, AMD, Intel GPUs
void compute_on_any_gpu(fb_gpu_context_t* ctx, fb_gpu_ptr_t x, fb_gpu_ptr_t y) {
    ctx->ops->saxpy(ctx, NULL, 1000, 2.0f, x, 1, y, 1);
    // Automatically uses cuBLAS, rocBLAS, or oneMKL depending on GPU
}
```

### 2. **Multi-GPU Workload Distribution**
```c
void parallel_compute(fb_gpu_manager_t* mgr, float* data, int n) {
    int chunk_size = n / mgr->num_devices;
    
    for (int i = 0; i < mgr->num_devices; i++) {
        fb_gpu_context_t* ctx = &mgr->contexts[i];
        
        // Allocate on this GPU
        fb_gpu_ptr_t gpu_data = ctx->ops->malloc(ctx->backend_handle, chunk_size * sizeof(float));
        
        // Copy chunk to this GPU
        ctx->ops->memcpy_h2d(ctx->backend_handle, gpu_data, 
                             &data[i * chunk_size], chunk_size * sizeof(float));
        
        // Compute on this GPU (cuBLAS OR rocBLAS OR oneMKL)
        ctx->ops->sscal(ctx, NULL, chunk_size, 2.0f, gpu_data, 1);
        
        // Copy result back
        ctx->ops->memcpy_d2h(ctx->backend_handle, &data[i * chunk_size],
                             gpu_data, chunk_size * sizeof(float));
        
        ctx->ops->free(ctx->backend_handle, gpu_data);
    }
}
```

### 3. **Easy to Extend**
Adding a new GPU backend (e.g., Apple Metal) only requires:
1. Implement the trait for Metal
2. Add detection logic
3. **Zero changes** to existing code

### 4. **Testing Made Easy**
```c
void test_all_gpu_backends(void) {
    const fb_gpu_backend_trait_t* backends[] = {
        &fb_cublas_trait,
        &fb_rocblas_trait,
        &fb_onemkl_trait
    };
    
    for (int i = 0; i < 3; i++) {
        test_saxpy_with_backend(backends[i]);
        test_sgemm_with_backend(backends[i]);
        // Same tests for all backends!
    }
}
```

## Implementation Effort

| Task                       | Effort          | Notes                                      |
| -------------------------- | --------------- | ------------------------------------------ |
| Define GPU trait interface | 2-4 hours       | Design vtable structure                    |
| Implement cuBLAS trait     | 40-50 hours     | All 212 operations                         |
| Implement rocBLAS trait    | 8-12 hours      | Copy cuBLAS, change API calls              |
| GPU manager/detector       | 4-6 hours       | Multi-GPU support                          |
| Testing framework          | 8-10 hours      | Test all backends                          |
| **Total**                  | **62-82 hours** | Similar to before, but better architecture |

## Recommendation

✅ **YES, implement unified GPU backend trait!**

Benefits:
- Transparent vendor support (your AMD + NVIDIA use case)
- Much easier testing
- Future-proof (easy to add new backends)
- Cleaner API for users
- rocBLAS becomes ~80% copy-paste from cuBLAS

This is the **professional, production-ready approach** that companies like PyTorch and TensorFlow use internally.

Would you like me to start implementing this unified GPU backend architecture?
