/**
 * @file operation_router.h
 * @brief Operation-level backend routing for optimal performance
 * 
 * Allows different BLAS operations to use different backends based on
 * operation type, data size, and performance characteristics.
 * 
 * Example: Small vector ops on CPU, large matrix ops on GPU
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_OPERATION_ROUTER_H
#define FASTER_BLASTER_OPERATION_ROUTER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief BLAS operation categories for routing decisions
 */
typedef enum {
    FB_OP_LEVEL1_DOT,        /**< Dot product operations (dot, nrm2, asum) */
    FB_OP_LEVEL1_AXPY,       /**< AXPY-style operations (axpy, scal, copy, swap) */
    FB_OP_LEVEL2_GEMV,       /**< Matrix-vector multiply */
    FB_OP_LEVEL2_GER,        /**< Rank-1 update */
    FB_OP_LEVEL2_SYMV,       /**< Symmetric matrix-vector */
    FB_OP_LEVEL3_GEMM,       /**< General matrix-matrix multiply */
    FB_OP_LEVEL3_SYMM,       /**< Symmetric matrix-matrix */
    FB_OP_LEVEL3_TRSM,       /**< Triangular solve */
    FB_OP_LAPACK_SOLVE,      /**< Linear system solvers */
    FB_OP_LAPACK_FACTOR,     /**< Matrix factorizations */
    FB_OP_LAPACK_EIGEN,      /**< Eigenvalue problems */
    FB_OP_CATEGORY_COUNT
} fb_operation_category_t;

/**
 * @brief Routing policy for backend selection
 */
typedef enum {
    FB_ROUTE_AUTO,           /**< Automatic selection based on operation and size */
    FB_ROUTE_BY_DEVICE,      /**< Use device's default backend */
    FB_ROUTE_BY_OPERATION,   /**< Route by operation category */
    FB_ROUTE_BY_SIZE,        /**< Route based on problem size thresholds */
    FB_ROUTE_CUSTOM          /**< User-provided routing function */
} fb_routing_policy_t;

/**
 * @brief Routing configuration for an operation category
 */
typedef struct {
    const char* preferred_backend;  /**< Backend name (e.g., "cublas", "mkl") */
    size_t size_threshold;          /**< Minimum size to use this backend */
    int device_id;                  /**< Target device ID (-1 = any) */
} fb_route_config_t;

/**
 * @brief Custom routing callback
 * @param operation Operation category
 * @param problem_size Estimated problem size (e.g., matrix dimensions)
 * @param user_data User-provided context
 * @return Device ID to use, or -1 for auto-select
 */
typedef int (*fb_routing_callback_t)(fb_operation_category_t operation,
                                      size_t problem_size,
                                      void* user_data);

/**
 * @brief Initialize operation routing system
 * @return 0 on success, negative on error
 */
int fb_routing_init(void);

/**
 * @brief Set routing policy
 * @param policy Routing policy to use
 * @return 0 on success, negative on error
 */
int fb_routing_set_policy(fb_routing_policy_t policy);

/**
 * @brief Configure routing for a specific operation category
 * @param category Operation category
 * @param config Routing configuration
 * @return 0 on success, negative on error
 */
int fb_routing_configure(fb_operation_category_t category, const fb_route_config_t* config);

/**
 * @brief Set custom routing callback
 * @param callback Routing function
 * @param user_data Context passed to callback
 * @return 0 on success, negative on error
 */
int fb_routing_set_callback(fb_routing_callback_t callback, void* user_data);

/**
 * @brief Get optimal device for an operation
 * @param category Operation category
 * @param problem_size Estimated problem size
 * @return Device ID, or -1 to use current device
 */
int fb_routing_get_device(fb_operation_category_t category, size_t problem_size);

/**
 * @brief Estimate problem size for routing decisions
 * Helper functions for common cases
 */
static inline size_t fb_routing_size_level1(int64_t n) {
    return (size_t)n;
}

static inline size_t fb_routing_size_level2(int64_t m, int64_t n) {
    return (size_t)(m * n);
}

static inline size_t fb_routing_size_level3(int64_t m, int64_t n, int64_t k) {
    return (size_t)(m * n + n * k + m * k);  /* Total data movement */
}

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_OPERATION_ROUTER_H */
