/**
 * @file gpu_manager.c
 * @brief Multi-GPU manager for vendor-agnostic GPU workload distribution
 * 
 * Automatically detects and manages NVIDIA, AMD, and Intel GPUs,
 * allowing transparent multi-GPU execution across vendors.
 */

#include "faster-blaster/gpu_backend_trait.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA detection
#ifdef __has_include
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define HAVE_CUDA 1
#endif
#endif

// ROCm/HIP detection
#ifdef __has_include
#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define HAVE_HIP 1
#endif
#endif

/* ============================================================================
 * Device Detection
 * ========================================================================== */

static int detect_cuda_devices(void) {
#ifdef HAVE_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
#else
    return 0;
#endif
}

static int detect_hip_devices(void) {
#ifdef HAVE_HIP
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        return 0;
    }
    return count;
#else
    return 0;
#endif
}

/* ============================================================================
 * GPU Manager Implementation
 * ========================================================================== */

fb_gpu_manager_t* fb_gpu_manager_init(void) {
    fb_gpu_manager_t* mgr = (fb_gpu_manager_t*)calloc(1, sizeof(fb_gpu_manager_t));
    if (!mgr) {
        fprintf(stderr, "GPU Manager: Failed to allocate manager\n");
        return NULL;
    }
    
    // Detect all GPU devices
    int cuda_count = detect_cuda_devices();
    int hip_count = detect_hip_devices();
    
    mgr->num_devices = cuda_count + hip_count;
    
    if (mgr->num_devices == 0) {
        fprintf(stderr, "GPU Manager: No GPU devices detected\n");
        free(mgr);
        return NULL;
    }
    
    // Allocate contexts
    mgr->contexts = (fb_gpu_context_t*)calloc(mgr->num_devices, sizeof(fb_gpu_context_t));
    if (!mgr->contexts) {
        fprintf(stderr, "GPU Manager: Failed to allocate contexts\n");
        free(mgr);
        return NULL;
    }
    
    int idx = 0;
    
    // Initialize CUDA devices (NVIDIA)
    for (int i = 0; i < cuda_count; i++, idx++) {
        mgr->contexts[idx].backend_type = FB_GPU_BACKEND_CUBLAS;
        mgr->contexts[idx].device_id = i;
        mgr->contexts[idx].trait = &fb_cublas_trait;
        
        if (fb_cublas_trait.init(i, NULL, &mgr->contexts[idx].backend_handle) != 0) {
            fprintf(stderr, "GPU Manager: Failed to initialize CUDA device %d\n", i);
            mgr->contexts[idx].trait = NULL;
            continue;
        }
        
        // Get device properties
        char name[256];
        size_t total_mem = 0;
        fb_cublas_trait.get_device_properties(mgr->contexts[idx].backend_handle, i,
                                              name, sizeof(name), &total_mem);
        
        printf("GPU Manager: Initialized GPU %d: %s (NVIDIA cuBLAS, %.2f GB)\n",
               idx, name, total_mem / (1024.0 * 1024.0 * 1024.0));
    }
    
    // Initialize HIP devices (AMD)
#ifdef HAVE_HIP
    for (int i = 0; i < hip_count; i++, idx++) {
        mgr->contexts[idx].backend_type = FB_GPU_BACKEND_ROCBLAS;
        mgr->contexts[idx].device_id = i;
        mgr->contexts[idx].trait = &fb_rocblas_trait;
        
        if (fb_rocblas_trait.init(i, NULL, &mgr->contexts[idx].backend_handle) != 0) {
            fprintf(stderr, "GPU Manager: Failed to initialize HIP device %d\n", i);
            mgr->contexts[idx].trait = NULL;
            continue;
        }
        
        // Get device properties
        char name[256];
        size_t total_mem = 0;
        fb_rocblas_trait.get_device_properties(mgr->contexts[idx].backend_handle, i,
                                               name, sizeof(name), &total_mem);
        
        printf("GPU Manager: Initialized GPU %d: %s (AMD rocBLAS, %.2f GB)\n",
               idx, name, total_mem / (1024.0 * 1024.0 * 1024.0));
    }
#endif
    
    return mgr;
}

void fb_gpu_manager_shutdown(fb_gpu_manager_t* mgr) {
    if (!mgr) {
        return;
    }
    
    if (mgr->contexts) {
        for (int i = 0; i < mgr->num_devices; i++) {
            if (mgr->contexts[i].trait && mgr->contexts[i].backend_handle) {
                mgr->contexts[i].trait->shutdown(mgr->contexts[i].backend_handle);
            }
        }
        free(mgr->contexts);
    }
    
    free(mgr);
}

fb_gpu_context_t* fb_gpu_get_context(fb_gpu_manager_t* mgr, int device_id) {
    if (!mgr || device_id < 0 || device_id >= mgr->num_devices) {
        return NULL;
    }
    
    return &mgr->contexts[device_id];
}

int fb_gpu_get_device_count(fb_gpu_manager_t* mgr) {
    if (!mgr) {
        return 0;
    }
    return mgr->num_devices;
}

/* ============================================================================
 * High-Level Helper Functions
 * ========================================================================== */

/**
 * @brief Execute SAXPY on specific GPU device
 */
void fb_gpu_saxpy_on_device(fb_gpu_manager_t* mgr, int device_id,
                            int n, float alpha, fb_gpu_ptr_t x, int incx,
                            fb_gpu_ptr_t y, int incy) {
    fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, device_id);
    if (!ctx || !ctx->trait || !ctx->trait->saxpy) {
        fprintf(stderr, "GPU Manager: Invalid device or SAXPY not available\n");
        return;
    }
    
    ctx->trait->saxpy(ctx->backend_handle, NULL, n, alpha, x, incx, y, incy);
}

/**
 * @brief Execute SGEMM on specific GPU device
 */
void fb_gpu_sgemm_on_device(fb_gpu_manager_t* mgr, int device_id,
                            char transa, char transb, int m, int n, int k,
                            float alpha, fb_gpu_ptr_t a, int lda,
                            fb_gpu_ptr_t b, int ldb, float beta,
                            fb_gpu_ptr_t c, int ldc) {
    fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, device_id);
    if (!ctx || !ctx->trait || !ctx->trait->sgemm) {
        fprintf(stderr, "GPU Manager: Invalid device or SGEMM not available\n");
        return;
    }
    
    ctx->trait->sgemm(ctx->backend_handle, NULL, transa, transb,
                      m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/**
 * @brief Allocate memory on specific GPU
 */
fb_gpu_ptr_t fb_gpu_malloc_on_device(fb_gpu_manager_t* mgr, int device_id, size_t size) {
    fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, device_id);
    if (!ctx || !ctx->trait || !ctx->trait->malloc) {
        return NULL;
    }
    
    fb_gpu_ptr_t ptr = NULL;
    if (ctx->trait->malloc(ctx->backend_handle, &ptr, size) != 0) {
        return NULL;
    }
    
    return ptr;
}

/**
 * @brief Free memory on specific GPU
 */
void fb_gpu_free_on_device(fb_gpu_manager_t* mgr, int device_id, fb_gpu_ptr_t ptr) {
    fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, device_id);
    if (!ctx || !ctx->trait || !ctx->trait->free) {
        return;
    }
    
    ctx->trait->free(ctx->backend_handle, ptr);
}

/**
 * @brief Copy host to device memory
 */
int fb_gpu_memcpy_h2d_on_device(fb_gpu_manager_t* mgr, int device_id,
                                 fb_gpu_ptr_t dst, const void* src, size_t size) {
    fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, device_id);
    if (!ctx || !ctx->trait || !ctx->trait->memcpy_h2d) {
        return -1;
    }
    
    return ctx->trait->memcpy_h2d(ctx->backend_handle, dst, src, size);
}

/**
 * @brief Copy device to host memory
 */
int fb_gpu_memcpy_d2h_on_device(fb_gpu_manager_t* mgr, int device_id,
                                 void* dst, fb_gpu_ptr_t src, size_t size) {
    fb_gpu_context_t* ctx = fb_gpu_get_context(mgr, device_id);
    if (!ctx || !ctx->trait || !ctx->trait->memcpy_d2h) {
        return -1;
    }
    
    return ctx->trait->memcpy_d2h(ctx->backend_handle, dst, src, size);
}
