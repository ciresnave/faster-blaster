/**
 * @file dispatch_unified.h
 * @brief Unified dispatch API - Single entry point for BLAS/LAPACK operations
 * 
 * This is the highest-level API that brings together:
 * 1. Device selection (compute_manager)
 * 2. Backend loading (backend_instance)
 * 3. Operation execution (backend vtables)
 * 4. Fallback handling
 * 
 * Example usage:
 * ```c
 * // Initialize system (once at startup)
 * fb_init();
 * 
 * // Execute GEMM - system automatically:
 * //   - Selects best device
 * //   - Loads appropriate backend
 * //   - Executes operation
 * //   - Falls back if needed
 * fb_sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
 *          M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
 * 
 * // Cleanup (once at shutdown)
 * fb_shutdown();
 * ```
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_DISPATCH_UNIFIED_H
#define FASTER_BLASTER_DISPATCH_UNIFIED_H

#include "compute_manager.h"
#include "backend_instance.h"
#include "backends/backend_interface.h"
#include "backend_ids.h"   /* FB_BACKEND_ID_* and FB_BACKEND_ID_NONE */
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operation type for backend selection
 * This allows different backends to be chosen for different operation types
 */
typedef enum {
    FB_OP_LEVEL1,      /**< Level 1 BLAS (vector ops) */
    FB_OP_LEVEL2,      /**< Level 2 BLAS (matrix-vector) */
    FB_OP_LEVEL3,      /**< Level 3 BLAS (matrix-matrix) */
    FB_OP_LAPACK,      /**< LAPACK operations */
    FB_OP_SPARSE,      /**< Sparse operations */
    FB_OP_CUSTOM       /**< Custom/fused operations */
} fb_operation_type_t;

/**
 * Initialize the faster-blaster dispatch system
 * 
 * This initializes all subsystems:
 * - Device detection (CPU and GPU)
 * - Device registry
 * - Compute manager
 * - Plugin system
 * - Backend instance manager
 * 
 * @return 0 on success, negative on error
 */
int fb_init(void);

/**
 * Shutdown the faster-blaster system
 * Unloads all backends, frees resources
 */
void fb_shutdown(void);

/**
 * Set dispatch policy for automatic device selection
 * 
 * @param policy Scheduling policy to use
 */
void fb_set_policy(fb_scheduling_policy_t policy);

/**
 * Get current dispatch policy
 * 
 * @return Current scheduling policy
 */
fb_scheduling_policy_t fb_get_policy(void);

/**
 * Manually select device for next operation
 * 
 * @param device_id Device ID from fb_print_devices()
 * @return 0 on success, negative on error
 */
int fb_use_device(uint32_t device_id);

/**
 * Reset to automatic device selection
 */
void fb_use_auto(void);

/**
 * Get backend instance for a specific operation
 * 
 * This is the primary API for operation-level backend selection.
 * It performs:
 * 1. Device selection based on current policy
 * 2. Backend selection for the operation type
 * 3. Backend instance retrieval/loading
 * 
 * @param op_type Type of operation (Level1/2/3, LAPACK, etc.)
 * @param precision Operation precision
 * @param problem_size Problem size (e.g., M*N*K for GEMM)
 * @param data_ptrs Array of data pointers (for locality)
 * @param num_data_ptrs Number of data pointers
 * @return Backend instance ready to execute, or NULL on error
 */
fb_backend_instance_t* fb_get_backend_for_operation(
    fb_operation_type_t op_type,
    fb_precision_t precision,
    size_t problem_size,
    const void** data_ptrs,
    uint32_t num_data_ptrs);

/**
 * Get backend instance for current/selected device
 * 
 * Simplified version of fb_get_backend_for_operation() that assumes
 * Level 3 BLAS and uses default problem size.
 * 
 * @param precision Operation precision
 * @param data_ptrs Array of data pointers (for locality)
 * @param num_data_ptrs Number of data pointers
 * @return Backend instance, or NULL on error
 */
fb_backend_instance_t* fb_get_current_backend(fb_precision_t precision,
                                              const void** data_ptrs,
                                              uint32_t num_data_ptrs);

/**
 * Print system status - devices, backends, statistics
 */
void fb_print_status(void);

/**
 * Enable/disable verbose logging
 * 
 * @param enable true to enable verbose output
 */
void fb_set_verbose(bool enable);

/* =========================================================================
 * Judge Dispatch Accessors
 * ========================================================================= */

/**
 * Override the profile directory used by the judge dispatch system.
 * Must be called BEFORE fb_init().  Has no effect after initialization.
 *
 * Default: "judge_profiles" (relative to the working directory).
 */
void fb_judge_set_profile_dir(const char *dir);

/**
 * Return true if a judge-driven dispatch table was successfully loaded
 * from stored profiles during fb_init().  When false, the system falls
 * back to probe-score (hardware-affinity) backend selection.
 */
bool fb_judge_dispatch_is_loaded(void);

/**
 * Return the judge-selected backend ID for a single operation.
 *
 * @param op_id  Operation identifier (FB_OP_* from judge_op_ids.h).
 * @return       Winning backend ID, or FB_BACKEND_ID_NONE if the dispatch
 *               table is not loaded or op_id is out of range.
 *
 * Use FB_BACKEND_ID_NONE as the sentinel — the caller should fall back to
 * fb_get_backend_for_operation() when this returns FB_BACKEND_ID_NONE.
 */
uint32_t fb_judge_get_routed_backend_id(uint32_t op_id);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_DISPATCH_UNIFIED_H */
