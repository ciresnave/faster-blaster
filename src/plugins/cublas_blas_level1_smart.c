/**
 * @file cublas_blas_level1_smart.c
 * @brief Smart wrappers for BLAS Level 1 rotation operations
 * 
 * This file provides smart wrappers for rotation operations:
 * - srot, drot: Apply Givens rotation
 * - srotg, drotg: Generate Givens rotation
 * - srotm, drotm: Apply modified Givens rotation
 * - srotmg, drotmg: Generate modified Givens rotation
 * 
 * All operations use the device memory manager for intelligent caching.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/gpu_backend_trait.h"
#include "faster-blaster/device_memory_manager.h"
#include "faster_blaster.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* External cuBLAS trait and handle */
extern const fb_gpu_backend_trait_t fb_cublas_trait;
extern void* g_cublas_handle;  /* Shared handle */
static const int CUBLAS_DEVICE_ID = 0;

/* Helper function prototypes */
static fb_device_memory_manager_t* get_device_manager(void);
static size_t calculate_vector_size(int n, int inc);
static int get_device_buffer_input(const void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out);
static int get_device_buffer_inout(void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out);
static void mark_output_dirty(void* host_ptr);
static void release_device_buffer(const void* host_ptr);

/* ============================================================================
 * Helper Functions (Implementation)
 * ========================================================================== */

static fb_device_memory_manager_t* get_device_manager(void) {
    return fb_device_memory_get_manager(CUBLAS_DEVICE_ID);
}

static size_t calculate_vector_size(int n, int inc) {
    if (n <= 0 || inc == 0) return 0;
    return (size_t)(1 + (n - 1) * abs(inc));
}

static int get_device_buffer_input(const void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) return -1;
    
    return fb_device_memory_get_or_alloc(manager, &fb_cublas_trait, g_cublas_handle,
                                         host_ptr, size, FB_GPU_BACKEND_CUBLAS, device_ptr_out);
}

static int get_device_buffer_inout(void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) return -1;
    
    return fb_device_memory_get_or_alloc(manager, &fb_cublas_trait, g_cublas_handle,
                                         host_ptr, size, FB_GPU_BACKEND_CUBLAS, device_ptr_out);
}

static void mark_output_dirty(void* host_ptr) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (manager) {
        fb_device_memory_mark_dirty(manager, host_ptr, FB_GPU_BACKEND_CUBLAS);
    }
}

static void release_device_buffer(const void* host_ptr) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (manager) {
        fb_device_memory_release(manager, host_ptr);
    }
}

/* ============================================================================
 * BLAS Level 1: Rotation Operations (8 functions)
 * ========================================================================== */

/**
 * @brief Apply Givens rotation (single precision)
 * Computes:  x[i] = c*x[i] + s*y[i]
 *            y[i] = c*y[i] - s*x[i]
 */
void cublas_srot_smart_wrapper(int n, float* x, int incx, float* y, int incy,
                                float c, float s) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    
    /* Get device buffers (both are input/output) */
    fb_gpu_ptr_t d_x, d_y;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(x);
        return;
    }
    
    /* Execute GPU operation */
    if (fb_cublas_trait.srot) {
        fb_cublas_trait.srot(g_cublas_handle, NULL, n, d_x, incx, d_y, incy, c, s);
    }
    
    /* Mark outputs as dirty */
    mark_output_dirty(x);
    mark_output_dirty(y);
    
    /* Release references */
    release_device_buffer(x);
    release_device_buffer(y);
}

/**
 * @brief Apply Givens rotation (double precision)
 */
void cublas_drot_smart_wrapper(int n, double* x, int incx, double* y, int incy,
                                double c, double s) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    
    /* Get device buffers */
    fb_gpu_ptr_t d_x, d_y;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(x);
        return;
    }
    
    /* Execute GPU operation */
    if (fb_cublas_trait.drot) {
        fb_cublas_trait.drot(g_cublas_handle, NULL, n, d_x, incx, d_y, incy, c, s);
    }
    
    /* Mark outputs as dirty */
    mark_output_dirty(x);
    mark_output_dirty(y);
    
    /* Release references */
    release_device_buffer(x);
    release_device_buffer(y);
}

/**
 * @brief Generate Givens rotation (single precision)
 * Computes rotation parameters c and s such that:
 * [ c  s ] [ a ]   [ r ]
 * [-s  c ] [ b ] = [ 0 ]
 */
void cublas_srotg_smart_wrapper(float* a, float* b, float* c, float* s) {
    /* Allocate device memory for scalar parameters */
    fb_gpu_ptr_t d_a, d_b, d_c, d_s;
    
    if (get_device_buffer_inout(a, sizeof(float), &d_a) != 0) return;
    if (get_device_buffer_inout(b, sizeof(float), &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(c, sizeof(float), &d_c) != 0) {
        release_device_buffer(a);
        release_device_buffer(b);
        return;
    }
    if (get_device_buffer_inout(s, sizeof(float), &d_s) != 0) {
        release_device_buffer(a);
        release_device_buffer(b);
        release_device_buffer(c);
        return;
    }
    
    /* Execute GPU operation */
    if (fb_cublas_trait.srotg) {
        fb_cublas_trait.srotg(g_cublas_handle, NULL, d_a, d_b, d_c, d_s);
    }
    
    /* Mark all outputs as dirty */
    mark_output_dirty(a);
    mark_output_dirty(b);
    mark_output_dirty(c);
    mark_output_dirty(s);
    
    /* Release references */
    release_device_buffer(a);
    release_device_buffer(b);
    release_device_buffer(c);
    release_device_buffer(s);
}

/**
 * @brief Generate Givens rotation (double precision)
 */
void cublas_drotg_smart_wrapper(double* a, double* b, double* c, double* s) {
    /* Allocate device memory for scalar parameters */
    fb_gpu_ptr_t d_a, d_b, d_c, d_s;
    
    if (get_device_buffer_inout(a, sizeof(double), &d_a) != 0) return;
    if (get_device_buffer_inout(b, sizeof(double), &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(c, sizeof(double), &d_c) != 0) {
        release_device_buffer(a);
        release_device_buffer(b);
        return;
    }
    if (get_device_buffer_inout(s, sizeof(double), &d_s) != 0) {
        release_device_buffer(a);
        release_device_buffer(b);
        release_device_buffer(c);
        return;
    }
    
    /* Execute GPU operation */
    if (fb_cublas_trait.drotg) {
        fb_cublas_trait.drotg(g_cublas_handle, NULL, d_a, d_b, d_c, d_s);
    }
    
    /* Mark all outputs as dirty */
    mark_output_dirty(a);
    mark_output_dirty(b);
    mark_output_dirty(c);
    mark_output_dirty(s);
    
    /* Release references */
    release_device_buffer(a);
    release_device_buffer(b);
    release_device_buffer(c);
    release_device_buffer(s);
}

/**
 * @brief Apply modified Givens rotation (single precision)
 * Uses a 5-element param vector: [flag, h11, h21, h12, h22]
 */
void cublas_srotm_smart_wrapper(int n, float* x, int incx, float* y, int incy,
                                 const float* param) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    size_t size_param = 5 * sizeof(float);
    
    /* Get device buffers */
    fb_gpu_ptr_t d_x, d_y, d_param;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(x);
        return;
    }
    if (get_device_buffer_input(param, size_param, &d_param) != 0) {
        release_device_buffer(x);
        release_device_buffer(y);
        return;
    }
    
    /* Execute GPU operation */
    if (fb_cublas_trait.srotm) {
        fb_cublas_trait.srotm(g_cublas_handle, NULL, n, d_x, incx, d_y, incy, d_param);
    }
    
    /* Mark outputs as dirty */
    mark_output_dirty(x);
    mark_output_dirty(y);
    
    /* Release references */
    release_device_buffer(x);
    release_device_buffer(y);
    release_device_buffer(param);
}

/**
 * @brief Apply modified Givens rotation (double precision)
 */
void cublas_drotm_smart_wrapper(int n, double* x, int incx, double* y, int incy,
                                 const double* param) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    size_t size_param = 5 * sizeof(double);
    
    /* Get device buffers */
    fb_gpu_ptr_t d_x, d_y, d_param;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(x);
        return;
    }
    if (get_device_buffer_input(param, size_param, &d_param) != 0) {
        release_device_buffer(x);
        release_device_buffer(y);
        return;
    }
    
    /* Execute GPU operation */
    if (fb_cublas_trait.drotm) {
        fb_cublas_trait.drotm(g_cublas_handle, NULL, n, d_x, incx, d_y, incy, d_param);
    }
    
    /* Mark outputs as dirty */
    mark_output_dirty(x);
    mark_output_dirty(y);
    
    /* Release references */
    release_device_buffer(x);
    release_device_buffer(y);
    release_device_buffer(param);
}

/**
 * @brief Generate modified Givens rotation (single precision)
 * Constructs modified Givens rotation from d1, d2, x1, y1
 * Output: param[5] = [flag, h11, h21, h12, h22]
 */
void cublas_srotmg_smart_wrapper(float* d1, float* d2, float* x1, const float* y1,
                                  float* param) {
    /* Allocate device memory for scalar parameters */
    fb_gpu_ptr_t d_d1, d_d2, d_x1, d_y1, d_param;
    
    if (get_device_buffer_inout(d1, sizeof(float), &d_d1) != 0) return;
    if (get_device_buffer_inout(d2, sizeof(float), &d_d2) != 0) {
        release_device_buffer(d1);
        return;
    }
    if (get_device_buffer_inout(x1, sizeof(float), &d_x1) != 0) {
        release_device_buffer(d1);
        release_device_buffer(d2);
        return;
    }
    if (get_device_buffer_input(y1, sizeof(float), &d_y1) != 0) {
        release_device_buffer(d1);
        release_device_buffer(d2);
        release_device_buffer(x1);
        return;
    }
    if (get_device_buffer_inout(param, 5 * sizeof(float), &d_param) != 0) {
        release_device_buffer(d1);
        release_device_buffer(d2);
        release_device_buffer(x1);
        release_device_buffer(y1);
        return;
    }
    
    /* Execute GPU operation */
    if (fb_cublas_trait.srotmg) {
        fb_cublas_trait.srotmg(g_cublas_handle, NULL, d_d1, d_d2, d_x1, d_y1, d_param);
    }
    
    /* Mark outputs as dirty */
    mark_output_dirty(d1);
    mark_output_dirty(d2);
    mark_output_dirty(x1);
    mark_output_dirty(param);
    
    /* Release references */
    release_device_buffer(d1);
    release_device_buffer(d2);
    release_device_buffer(x1);
    release_device_buffer(y1);
    release_device_buffer(param);
}

/**
 * @brief Generate modified Givens rotation (double precision)
 */
void cublas_drotmg_smart_wrapper(double* d1, double* d2, double* x1, const double* y1,
                                  double* param) {
    /* Allocate device memory for scalar parameters */
    fb_gpu_ptr_t d_d1, d_d2, d_x1, d_y1, d_param;
    
    if (get_device_buffer_inout(d1, sizeof(double), &d_d1) != 0) return;
    if (get_device_buffer_inout(d2, sizeof(double), &d_d2) != 0) {
        release_device_buffer(d1);
        return;
    }
    if (get_device_buffer_inout(x1, sizeof(double), &d_x1) != 0) {
        release_device_buffer(d1);
        release_device_buffer(d2);
        return;
    }
    if (get_device_buffer_input(y1, sizeof(double), &d_y1) != 0) {
        release_device_buffer(d1);
        release_device_buffer(d2);
        release_device_buffer(x1);
        return;
    }
    if (get_device_buffer_inout(param, 5 * sizeof(double), &d_param) != 0) {
        release_device_buffer(d1);
        release_device_buffer(d2);
        release_device_buffer(x1);
        release_device_buffer(y1);
        return;
    }
    
    /* Execute GPU operation */
    if (fb_cublas_trait.drotmg) {
        fb_cublas_trait.drotmg(g_cublas_handle, NULL, d_d1, d_d2, d_x1, d_y1, d_param);
    }
    
    /* Mark outputs as dirty */
    mark_output_dirty(d1);
    mark_output_dirty(d2);
    mark_output_dirty(x1);
    mark_output_dirty(param);
    
    /* Release references */
    release_device_buffer(d1);
    release_device_buffer(d2);
    release_device_buffer(x1);
    release_device_buffer(y1);
    release_device_buffer(param);
}
