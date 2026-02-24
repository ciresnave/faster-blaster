/**
 * @file backend_matcher.c
 * @brief Device-to-Backend Matching System
 * 
 * This module provides intelligent matching between detected devices
 * (from device_registry) and available backends (from backend_loader).
 * It integrates with the compute manager to enable seamless backend selection.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "backend_matcher.h"
#include "backend_loader.h"
#include "faster-blaster/device_registry.h"
#include "faster-blaster/compute_device.h"
#include "faster-blaster/cpu_backend_trait.h"
#include "faster-blaster/gpu_backend_trait.h"
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

/* Backend instance cache - complete definition of opaque type from header */
struct backend_instance {
    int device_id;
    fb_device_type_t device_type;
    fb_backend_type_t backend_type;
    bool initialized;
    void* backend_handle;  /* Backend-specific context */
    union {
        fb_cpu_backend_trait_t* cpu_backend;
        fb_gpu_backend_trait_t* gpu_backend;
    };
};

#define MAX_BACKEND_INSTANCES 32
static backend_instance_t g_instances[MAX_BACKEND_INSTANCES];
static int g_instance_count = 0;
static bool g_matcher_initialized = false;

/* Helper: convert char transpose to fb_transpose_t */
static fb_transpose_t char_to_transpose(char trans) {
    switch (trans) {
        case 'N': case 'n': return FB_NO_TRANS;
        case 'T': case 't': return FB_TRANS;
        case 'C': case 'c': return FB_CONJ_TRANS;
        default: return FB_NO_TRANS;
    }
}

/**
 * Initialize backend matcher system
 */
int fb_backend_matcher_init(void) {
    if (g_matcher_initialized) {
        return 0;
    }
    
    /* Initialize backend registry if not already done */
    if (fb_registry_get_device_count() == 0) {
        if (fb_registry_init(FB_DISCOVERY_ALL) < 0) {
            fprintf(stderr, "Failed to initialize device registry\n");
            return -1;
        }
    }
    
    /* Initialize backend loader */
    if (fb_backend_loader_init() != 0) {
        fprintf(stderr, "Failed to initialize backend loader\n");
        return -1;
    }
    
    /* Initialize device registry */
    if (fb_registry_init(FB_DISCOVERY_ALL) != 0) {
        fprintf(stderr, "Failed to initialize device registry\n");
        return -1;
    }
    
    memset(g_instances, 0, sizeof(g_instances));
    g_instance_count = 0;
    g_matcher_initialized = true;
    
    return 0;
}

/**
 * Shutdown backend matcher and cleanup all instances
 */
void fb_backend_matcher_shutdown(void) {
    if (!g_matcher_initialized) {
        return;
    }
    
    /* Cleanup all backend instances */
    for (int i = 0; i < g_instance_count; i++) {
        if (g_instances[i].initialized) {
            if (g_instances[i].device_type == FB_DEVICE_TYPE_GPU && 
                g_instances[i].gpu_backend && 
                g_instances[i].gpu_backend->shutdown) {
                g_instances[i].gpu_backend->shutdown(g_instances[i].backend_handle);
            }
            /* CPU backends typically don't need per-device cleanup */
        }
    }
    
    fb_backend_loader_shutdown();
    g_matcher_initialized = false;
}

/**
 * Select optimal backend for a CPU device
 */
static fb_backend_type_t select_cpu_backend(const fb_compute_device_t* device) {
    char vendor[64] = {0};
    uint64_t features = 0;
    fb_backend_detect_cpu(vendor, &features);
    
    /* Intel CPU -> prefer MKL */
    if (strstr(vendor, "Intel") || strstr(vendor, "GenuineIntel")) {
        fb_backend_metadata_t mkl_info;
        if (fb_backend_get_info(FB_BACKEND_MKL, &mkl_info) == 0 && mkl_info.available) {
            return FB_BACKEND_MKL;
        }
    }
    
    /* AMD CPU -> prefer AOCL or BLIS */
    if (strstr(vendor, "AMD") || strstr(vendor, "AuthenticAMD")) {
        fb_backend_metadata_t aocl_info;
        if (fb_backend_get_info(FB_BACKEND_AOCL, &aocl_info) == 0 && aocl_info.available) {
            return FB_BACKEND_AOCL;
        }
        
        fb_backend_metadata_t blis_info;
        if (fb_backend_get_info(FB_BACKEND_BLIS, &blis_info) == 0 && blis_info.available) {
            return FB_BACKEND_BLIS;
        }
    }
    
    /* Apple CPU -> prefer Accelerate */
#ifdef __APPLE__
    fb_backend_metadata_t accel_info;
    if (fb_backend_get_info(FB_BACKEND_ACCELERATE, &accel_info) == 0 && accel_info.available) {
        return FB_BACKEND_ACCELERATE;
    }
#endif
    
    /* Fallback: OpenBLAS (portable) */
    fb_backend_metadata_t openblas_info;
    if (fb_backend_get_info(FB_BACKEND_OPENBLAS, &openblas_info) == 0 && openblas_info.available) {
        return FB_BACKEND_OPENBLAS;
    }
    
    /* Last resort: reference backend */
    return FB_BACKEND_REFERENCE;
}

/**
 * Select optimal backend for a GPU device
 */
static fb_backend_type_t select_gpu_backend(const fb_compute_device_t* device) {
    /* Check vendor from device name */
    const char* name = device->properties.name;
    
    /* NVIDIA GPU -> cuBLAS */
    if (strstr(name, "NVIDIA") || strstr(name, "GeForce") || 
        strstr(name, "Quadro") || strstr(name, "Tesla") || 
        strstr(name, "RTX") || strstr(name, "GTX")) {
        fb_backend_metadata_t cublas_info;
        if (fb_backend_get_info(FB_BACKEND_CUBLAS, &cublas_info) == 0 && cublas_info.available) {
            return FB_BACKEND_CUBLAS;
        }
    }
    
    /* AMD GPU -> rocBLAS */
    if (strstr(name, "AMD") || strstr(name, "Radeon") || 
        strstr(name, "RDNA") || strstr(name, "CDNA")) {
        fb_backend_metadata_t rocblas_info;
        if (fb_backend_get_info(FB_BACKEND_ROCBLAS, &rocblas_info) == 0 && rocblas_info.available) {
            return FB_BACKEND_ROCBLAS;
        }
    }
    
    /* Intel GPU -> oneMKL */
    if (strstr(name, "Intel") || strstr(name, "Arc") || 
        strstr(name, "Iris") || strstr(name, "UHD")) {
        fb_backend_metadata_t onemkl_info;
        if (fb_backend_get_info(FB_BACKEND_ONEMKL_GPU, &onemkl_info) == 0 && onemkl_info.available) {
            return FB_BACKEND_ONEMKL_GPU;
        }
    }
    
    /* No suitable GPU backend found */
    fprintf(stderr, "Warning: No GPU backend available for device '%s'\n", name);
    return FB_BACKEND_REFERENCE;
}

/**
 * Get or create backend instance for a device
 */
fb_backend_instance_t* fb_backend_get_for_device(int device_id) {
    if (!g_matcher_initialized) {
        fprintf(stderr, "Backend matcher not initialized\n");
        return NULL;
    }
    
    /* Check if instance already exists */
    for (int i = 0; i < g_instance_count; i++) {
        if (g_instances[i].device_id == device_id && g_instances[i].initialized) {
            return &g_instances[i];
        }
    }
    
    /* Create new instance */
    if (g_instance_count >= MAX_BACKEND_INSTANCES) {
        fprintf(stderr, "Maximum backend instances reached\n");
        return NULL;
    }
    
    /* Get device info */
    const fb_compute_device_t* device = fb_registry_get_device(device_id);
    if (!device) {
        fprintf(stderr, "Device %d not found\n", device_id);
        return NULL;
    }
    
    /* Select appropriate backend */
    fb_backend_type_t backend_type;
    if (device->properties.type == FB_DEVICE_TYPE_CPU) {
        backend_type = select_cpu_backend(device);
    } else {
        backend_type = select_gpu_backend(device);
    }
    
    backend_instance_t* instance = &g_instances[g_instance_count];
    instance->device_id = device_id;
    instance->device_type = device->properties.type;
    instance->backend_type = backend_type;
    
    /* Initialize backend */
    if (device->properties.type == FB_DEVICE_TYPE_GPU) {
#ifdef FB_ENABLE_CUDA
        if (backend_type == FB_BACKEND_CUBLAS) {
            extern const fb_gpu_backend_trait_t* fb_cublas_get_backend(void);
            instance->gpu_backend = (fb_gpu_backend_trait_t*)fb_cublas_get_backend();
            
            if (instance->gpu_backend && instance->gpu_backend->init) {
                if (instance->gpu_backend->init(device_id, NULL, &instance->backend_handle) == 0) {
                    instance->initialized = true;
                    g_instance_count++;
                    
                    printf("✓ Initialized cuBLAS backend for device %d (%s)\n", 
                           device_id, device->properties.name);
                    return instance;
                }
            }
        }
#endif
        
#if defined(FB_ENABLE_HIP) && defined(FB_HAVE_ROCBLAS_BACKEND)
        if (backend_type == FB_BACKEND_ROCBLAS) {
            extern const fb_gpu_backend_trait_t* fb_rocblas_get_backend(void);
            instance->gpu_backend = (fb_gpu_backend_trait_t*)fb_rocblas_get_backend();
            
            if (instance->gpu_backend && instance->gpu_backend->init) {
                if (instance->gpu_backend->init(device_id, NULL, &instance->backend_handle) == 0) {
                    instance->initialized = true;
                    g_instance_count++;
                    
                    printf("✓ Initialized rocBLAS backend for device %d (%s)\n",
                           device_id, device->properties.name);
                    return instance;
                }
            }
        }
#endif
        
        fprintf(stderr, "Failed to initialize GPU backend for device %d\n", device_id);
        return NULL;
        
    } else {
        /* CPU backend */
        fb_backend_vtable_t vtable;
        if (fb_backend_load(backend_type, &vtable) == 0) {
            /* CPU backends typically use static global state */
            instance->cpu_backend = NULL;  /* Store vtable pointer if needed */
            instance->backend_handle = NULL;
            instance->initialized = true;
            g_instance_count++;
            
            fb_backend_metadata_t backend_info;
            fb_backend_get_info(backend_type, &backend_info);
            printf("✓ Loaded %s backend for device %d (%s)\n",
                   backend_info.name, device_id, device->properties.name);
            return instance;
        }
        
        fprintf(stderr, "Failed to load CPU backend for device %d\n", device_id);
        return NULL;
    }
}

/**
 * Get backend information for a device
 */
int fb_backend_get_device_backend_info(int device_id, fb_backend_metadata_t* info) {
    fb_backend_instance_t* instance = fb_backend_get_for_device(device_id);
    if (!instance) {
        return -1;
    }
    
    return fb_backend_get_info(instance->backend_type, info);
}

/**
 * Execute SGEMM on appropriate backend
 */
int fb_backend_execute_sgemm(int device_id, char transa, char transb,
                              int m, int n, int k, float alpha,
                              const float* A, int lda, const float* B, int ldb,
                              float beta, float* C, int ldc) {
    fb_backend_instance_t* instance = fb_backend_get_for_device(device_id);
    if (!instance) {
        return -1;
    }
    
    const fb_compute_device_t* device = fb_registry_get_device(device_id);
    
    if (device->properties.type == FB_DEVICE_TYPE_GPU && instance->gpu_backend) {
        if (instance->gpu_backend->sgemm) {
            instance->gpu_backend->sgemm(
                instance->backend_handle, NULL /* default stream */,
                transa, transb, m, n, k, alpha, (fb_gpu_ptr_t)A, lda, (fb_gpu_ptr_t)B, ldb, beta, (fb_gpu_ptr_t)C, ldc
            );
            return 0;
        }
        /* GPU backend doesn't support this operation */
        fprintf(stderr, "GPU backend for device %d doesn't support SGEMM\n", device_id);
        return -1;
    } else {
        /* CPU backend execution via vtable */
        fb_backend_vtable_t vtable;
        if (fb_backend_load(instance->backend_type, &vtable) == 0 && vtable.sgemm) {
            vtable.sgemm(FB_LAYOUT_COL_MAJOR, char_to_transpose(transa), char_to_transpose(transb),
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return 0;
        }
    }
    
    fprintf(stderr, "SGEMM not available for device %d\n", device_id);
    return -1;
}

/**
 * Execute DGEMM on appropriate backend
 */
int fb_backend_execute_dgemm(int device_id, char transa, char transb,
                              int m, int n, int k, double alpha,
                              const double* A, int lda, const double* B, int ldb,
                              double beta, double* C, int ldc) {
    fb_backend_instance_t* instance = fb_backend_get_for_device(device_id);
    if (!instance) {
        return -1;
    }
    
    const fb_compute_device_t* device = fb_registry_get_device(device_id);
    
    if (device->properties.type == FB_DEVICE_TYPE_GPU && instance->gpu_backend) {
        if (instance->gpu_backend->dgemm) {
            instance->gpu_backend->dgemm(
                instance->backend_handle, NULL /* default stream */,
                transa, transb, m, n, k, alpha, (fb_gpu_ptr_t)A, lda, (fb_gpu_ptr_t)B, ldb, beta, (fb_gpu_ptr_t)C, ldc
            );
            return 0;
        }
        /* GPU backend doesn't support this operation */
        fprintf(stderr, "GPU backend for device %d doesn't support DGEMM\n", device_id);
        return -1;
    } else {
        /* CPU backend execution via vtable */
        fb_backend_vtable_t vtable;
        if (fb_backend_load(instance->backend_type, &vtable) == 0 && vtable.dgemm) {
            vtable.dgemm(FB_LAYOUT_COL_MAJOR, char_to_transpose(transa), char_to_transpose(transb),
                        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            return 0;
        }
    }
    
    fprintf(stderr, "DGEMM not available for device %d\n", device_id);
    return -1;
}

/**
 * Print backend configuration for all devices
 */
void fb_backend_print_configuration(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  Backend Configuration\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    
    /* List available backends */
    printf("\nAvailable Backends:\n");
    fb_backend_metadata_t backends[FB_BACKEND_COUNT];
    int backend_count = fb_backend_list_available(backends, FB_BACKEND_COUNT);
    
    for (int i = 0; i < backend_count; i++) {
        printf("  %s\n", backends[i].name);
        printf("    Version: %s\n", backends[i].version);
        printf("    Vendor: %s\n", backends[i].vendor);
        printf("    License: %s\n", backends[i].license);
        printf("    Priority: %d\n", backends[i].priority);
        printf("    Type: %s\n", (backends[i].capabilities & FB_CAP_GPU) ? "GPU" : "CPU");
        printf("\n");
    }
    
    /* Device-to-backend mapping */
    printf("Device-to-Backend Mapping:\n");
    int total_devices, gpu_count;
    fb_registry_get_device_count(&total_devices, &gpu_count);
    
    for (int i = 0; i < total_devices; i++) {
        const fb_compute_device_t* device = fb_registry_get_device(i);
        if (!device) continue;
        
        fb_backend_metadata_t backend_info;
        fb_backend_instance_t* instance = fb_backend_get_for_device(i);
        
        if (instance && fb_backend_get_info(instance->backend_type, &backend_info) == 0) {
            printf("  Device %d (%s): %s\n", i, device->properties.name, backend_info.name);
        } else {
            printf("  Device %d (%s): No backend configured\n", i, device->properties.name);
        }
    }
    
    printf("\n");
}
 