# Backend Plugin Architecture

## Overview

The faster-blaster backend plugin system provides a flexible, modular approach to integrating optimized BLAS/LAPACK implementations. This document describes the plugin architecture, interfaces, and guidelines for creating custom backends.

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│         Faster-Blaster Public API Layer            │
│  (Unified interface for all BLAS/LAPACK operations) │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│          Backend Loader & Dispatcher                │
│  (Runtime detection, loading, selection)            │
└──────────────────┬──────────────────────────────────┘
                   │
       ┌───────────┼───────────┬───────────┐
       │           │           │           │
┌──────▼───┐  ┌───▼────┐  ┌───▼────┐  ┌───▼────┐
│ Reference│  │  CPU   │  │  GPU   │  │ Custom │
│ Backend  │  │ Backends│  │Backends│  │ Plugin │
└──────────┘  └────┬───┘  └────┬───┘  └────────┘
                   │           │
       ┌───────────┼───────────┼───────────┐
       │           │           │           │
   ┌───▼────┐  ┌──▼───┐   ┌───▼─────┐ ┌──▼─────┐
   │OpenBLAS│  │ MKL  │   │ cuBLAS  │ │rocBLAS │
   │(bundled)│  │(opt.)│   │(NVIDIA) │ │ (AMD)  │
   └────────┘  └──────┘   └─────────┘ └────────┘
```

## Plugin Types

### 1. Bundled Backends
**Always available, shipped with faster-blaster**
- **Reference Backend**: Pure C implementation, serves as validation baseline
- **OpenBLAS**: BSD-licensed, optimized for general CPUs

### 2. Detected Backends
**Loaded at runtime if libraries are found**
- **Intel MKL**: Highest performance on Intel CPUs (proprietary)
- **Apple Accelerate**: macOS/iOS system framework
- **AMD AOCL**: Optimized for AMD Zen CPUs
- **BLIS**: Modern BLAS framework

### 3. GPU Backends
**Optional performance accelerators**
- **NVIDIA cuBLAS**: CUDA-based GPU acceleration
- **AMD rocBLAS**: ROCm-based GPU acceleration (AMD GPUs)
- **Intel oneMKL GPU**: GPU offload for Intel Arc/Data Center GPUs

### 4. Custom Backends
**User-provided implementations via plugin API**

## Backend Interface

All backends implement the `fb_backend_vtable_t` interface defined in `backend_interface.h`:

```c
typedef struct {
    /* Level 1 BLAS */
    fb_sasum_fn sasum;
    fb_dasum_fn dasum;
    // ... (52 Level 1 operations)
    
    /* Level 2 BLAS */
    fb_sgemv_fn sgemv;
    fb_dgemv_fn dgemv;
    // ... (70 Level 2 operations)
    
    /* Level 3 BLAS */
    fb_sgemm_fn sgemm;
    fb_dgemm_fn dgemm;
    // ... (30 Level 3 operations)
    
    /* LAPACK */
    fb_sgetrf_fn sgetrf;
    fb_dgetrf_fn dgetrf;
    // ... (60 LAPACK operations)
} fb_backend_vtable_t;
```

**Total: 212 operations (100% complete)**

## Plugin Development

### Step 1: Implement Backend Vtable

Create a source file implementing the vtable:

```c
/* my_custom_backend.c */
#include "backend_interface.h"

/* Implement operations */
static void my_sasum(int n, const float* x, int incx, float* result) {
    /* Custom implementation */
}

static void my_dgemm(char transa, char transb, int m, int n, int k,
                     double alpha, const double* a, int lda,
                     const double* b, int ldb,
                     double beta, double* c, int ldc) {
    /* Custom implementation */
}

/* Populate vtable */
const fb_backend_vtable_t my_custom_vtable = {
    .sasum = my_sasum,
    .dgemm = my_dgemm,
    /* ... other operations ... */
};
```

### Step 2: Register Backend

Use the backend loader to register your custom backend:

```c
#include "backend_loader.h"

int register_my_backend(void) {
    uint32_t capabilities = FB_CAP_CPU | FB_CAP_LEVEL1 | 
                           FB_CAP_LEVEL3 | FB_CAP_DOUBLE;
    
    return fb_backend_register_custom("MyBackend", 
                                      &my_custom_vtable, 
                                      capabilities);
}
```

### Step 3: Use Backend

Application code selects backend at runtime:

```c
/* Initialize backend system */
fb_backend_loader_init();

/* Option 1: Auto-select best available */
fb_backend_type_t backend = fb_backend_auto_select(false);
fb_backend_set_current(backend);

/* Option 2: Explicitly select */
fb_backend_set_current(FB_BACKEND_MKL);

/* Option 3: Use custom backend */
register_my_backend();
fb_backend_set_current(FB_BACKEND_CUSTOM);

/* Now all operations use selected backend */
double result = fb_ddot(n, x, incx, y, incy);
```

## GPU Backend Interface

GPU backends extend the base interface with memory management:

```c
typedef struct {
    /* Device management */
    int (*init)(int device_id, fb_gpu_context_t* ctx);
    void (*shutdown)(fb_gpu_context_t* ctx);
    
    /* Memory management */
    int (*malloc)(fb_gpu_ptr_t* ptr, size_t size);
    void (*free)(fb_gpu_ptr_t ptr);
    int (*memcpy_h2d)(fb_gpu_ptr_t dst, const void* src, size_t size);
    int (*memcpy_d2h)(void* dst, fb_gpu_ptr_t src, size_t size);
    
    /* BLAS operations vtable */
    const fb_backend_vtable_t* blas_vtable;
} fb_gpu_backend_t;
```

### GPU Backend Example

```c
/* Initialize GPU */
fb_gpu_context_t gpu_ctx;
fb_gpu_init(0, &gpu_ctx);  /* Device 0 */

/* Allocate device memory */
fb_gpu_ptr_t d_x, d_y;
fb_gpu_malloc(&d_x, n * sizeof(float));
fb_gpu_malloc(&d_y, n * sizeof(float));

/* Copy data to GPU */
fb_gpu_memcpy_h2d(d_x, h_x, n * sizeof(float), NULL);
fb_gpu_memcpy_h2d(d_y, h_y, n * sizeof(float), NULL);

/* Set GPU backend */
fb_backend_set_current(FB_BACKEND_CUBLAS);

/* Perform GPU operation (automatically uses device pointers) */
float result;
fb_sdot(n, d_x, 1, d_y, 1, &result);

/* Copy result back */
fb_gpu_memcpy_d2h(&result, &result, sizeof(float), NULL);

/* Cleanup */
fb_gpu_free(d_x);
fb_gpu_free(d_y);
fb_gpu_shutdown(&gpu_ctx);
```

## Backend Capabilities

Backends declare capabilities via bitflags:

```c
typedef enum {
    FB_CAP_CPU        = (1 << 0),  /* CPU operations */
    FB_CAP_GPU        = (1 << 1),  /* GPU operations */
    FB_CAP_LEVEL1     = (1 << 2),  /* BLAS Level 1 */
    FB_CAP_LEVEL2     = (1 << 3),  /* BLAS Level 2 */
    FB_CAP_LEVEL3     = (1 << 4),  /* BLAS Level 3 */
    FB_CAP_LAPACK     = (1 << 5),  /* LAPACK routines */
    FB_CAP_DOUBLE     = (1 << 6),  /* Double precision */
    FB_CAP_SINGLE     = (1 << 7),  /* Single precision */
    FB_CAP_COMPLEX    = (1 << 8),  /* Complex types */
    FB_CAP_THREADSAFE = (1 << 9),  /* Thread-safe */
} fb_backend_capability_t;
```

## Distribution Model

### Base Package (Always Included)
- Reference backend
- OpenBLAS (BSD license - redistributable)
- Backend loader infrastructure

### Optional Performance Packs (Separate Installers)
- **Intel MKL Pack**: Separate installer with Intel license agreement
- **GPU Acceleration Pack**: CUDA/ROCm installers with NVIDIA/AMD licenses
- **AMD Optimization Pack**: AOCL library for Zen CPUs

### Runtime Detection
- Apple Accelerate (macOS only, system framework)
- Already-installed MKL, CUDA, ROCm
- Custom user backends

## Backend Priority System

When auto-selecting, backends are prioritized:

| Priority | Backend          | Reason                     |
| -------- | ---------------- | -------------------------- |
| 200      | cuBLAS           | GPU acceleration (NVIDIA)  |
| 190      | rocBLAS          | GPU acceleration (AMD)     |
| 180      | oneMKL GPU       | GPU acceleration (Intel)   |
| 100      | Intel MKL        | Optimized for Intel CPUs   |
| 90       | Apple Accelerate | Native macOS optimization  |
| 85       | AMD AOCL         | Optimized for AMD CPUs     |
| 70       | BLIS             | Modern BLAS framework      |
| 60       | Custom           | User-provided              |
| 50       | OpenBLAS         | General-purpose, bundled   |
| 1        | Reference        | Fallback, always available |

## Thread Safety

All backends must be thread-safe when declared with `FB_CAP_THREADSAFE`. Backends handle internal locking/synchronization.

Users can also manage threading:
```c
/* Single-threaded mode */
fb_set_num_threads(1);

/* Multi-threaded (let backend decide) */
fb_set_num_threads(0);

/* Explicit thread count */
fb_set_num_threads(8);
```

## Error Handling

Backends should return error codes:
- `0`: Success
- `< 0`: Error (use `fb_backend_get_error()` for details)

```c
if (fb_backend_load(FB_BACKEND_MKL, &vtable) < 0) {
    fprintf(stderr, "Failed to load MKL: %s\n", fb_backend_get_error());
    /* Fall back to OpenBLAS or Reference */
}
```

## Performance Measurement

Built-in benchmarking support:

```c
/* Benchmark all available backends */
fb_backend_benchmark_t results[FB_BACKEND_COUNT];
int count = fb_benchmark_all_backends(results, FB_BACKEND_COUNT);

for (int i = 0; i < count; i++) {
    printf("%s: %.2f GFLOPS\n", results[i].name, results[i].gflops);
}
```

## License Compliance

| Backend          | License                    | Redistribution | Notes                     |
| ---------------- | -------------------------- | -------------- | ------------------------- |
| Reference        | MIT                        | ✅ Yes          | Part of faster-blaster    |
| OpenBLAS         | BSD-3-Clause               | ✅ Yes          | Bundled by default        |
| Intel MKL        | Intel Proprietary          | ⚠️ Separate     | Requires installer + EULA |
| Apple Accelerate | Apple System Framework     | ❌ No           | System-only, can link     |
| AMD AOCL         | BSD-3-Clause               | ✅ Yes          | Optional bundle           |
| BLIS             | BSD-3-Clause               | ✅ Yes          | Optional bundle           |
| cuBLAS           | NVIDIA CUDA EULA           | ⚠️ Separate     | Requires CUDA installer   |
| rocBLAS          | MIT                        | ✅ Yes          | Optional with ROCm        |
| oneMKL GPU       | Apache 2.0 + Intel Runtime | ⚠️ Separate     | Requires oneAPI installer |

## Future Extensions

### Planned Features
1. **Automatic benchmark-based selection**: First run benchmarks backends, caches best choice
2. **Operation-specific routing**: Use different backends for different operations
3. **Mixed-precision optimizations**: Automatic downcasting for performance
4. **Distributed backends**: MPI-aware PBLAS support
5. **WebAssembly backend**: Browser-based computation

### Plugin API Evolution
- Hot-reloading of backends
- Version negotiation
- ABI stability guarantees
- Plugin marketplace/registry

## References

- [Backend Interface Header](../backend_interface.h)
- [Backend Loader Implementation](../backend_loader.c)
- [GPU Backend Interface](../gpu/gpu_backend.h)
- [cuBLAS Backend](../gpu/cublas_backend.c)
- [rocBLAS Backend](../gpu/rocblas_backend.c)
