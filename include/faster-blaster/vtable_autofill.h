/**
 * @file vtable_autofill.h
 * @brief Automatic vtable completion for missing backend operations
 *
 * Implements 4-strategy auto-fill system to ensure "correctness everywhere,
 * performance where available". When a plugin provides only a subset of
 * operations, faster-blaster generates missing vtable entries using:
 *
 * Strategy 1: Unified ↔ Specific (zero overhead)
 * Strategy 2: Batched → Single (zero overhead)
 * Strategy 3: Array → Strided (lightweight)
 * Strategy 4: Precision Promotion (last resort, 2× memory overhead)
 *
 * See VTABLE_AUTOFILL_DESIGN.md for complete specification.
 *
 * @copyright Copyright (c) 2025-2026
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_VTABLE_AUTOFILL_H
#define FASTER_BLASTER_VTABLE_AUTOFILL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Canonical complex types and fb_status_t */
#include "fb_types.h"

/* Opaque vtable handle — full struct defined in backend_interface.h */
typedef struct fb_backend_vtable fb_backend_vtable_t;

/* fb_generic_fn is defined in fb_types.h (included above) */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Core Auto-Fill Entry Point
 * ========================================================================= */

/**
 * @brief Finalize vtable after plugin registration
 *
 * Applies auto-fill strategies in order (zero overhead → last resort):
 * 1. Unified ↔ Specific wrappers
 * 2. Batched → Single wrappers
 * 3. Array → Strided wrappers
 * 4. Precision promotion wrappers (with overflow detection)
 *
 * Called automatically by fb_load_best_plugin() after plugin init.
 * Logs warnings when Strategy 4 (slow path) is activated.
 *
 * @param vtable Vtable to complete
 * @return FB_STATUS_SUCCESS on success, error code otherwise
 */
fb_status_t fb_finalize_plugin_vtable(fb_backend_vtable_t *vtable);

/* ============================================================================
 * Strategy 1: Unified ↔ Specific (Zero Overhead)
 * ========================================================================= */

/**
 * @brief Generate specific BLAS wrappers from unified operations
 *
 * If plugin provides gemm_unified but not sgemm/dgemm/etc., generate
 * specific wrappers that call unified with appropriate precision flags.
 *
 * Generated wrappers are inlined by compiler → zero overhead.
 *
 * @param vtable Vtable to fill
 * @return FB_STATUS_SUCCESS on success
 */
fb_status_t fb_autofill_unified_to_specific(fb_backend_vtable_t *vtable);

/**
 * @brief Generate unified dispatcher from specific operations
 *
 * If plugin provides sgemm/dgemm/etc. but not gemm_unified, generate
 * unified dispatcher that routes by precision enum to specific ops.
 *
 * @param vtable Vtable to fill
 * @return FB_STATUS_SUCCESS on success
 */
fb_status_t fb_autofill_specific_to_unified(fb_backend_vtable_t *vtable);

/* ============================================================================
 * Strategy 2: Batched → Single (Zero Overhead)
 * ========================================================================= */

/**
 * @brief Generate single-operation wrappers from batched operations
 *
 * If plugin provides sgemm_batch_strided but not sgemm, generate wrapper
 * that calls batched version with batch_count=1.
 *
 * Zero overhead - compiler optimizes away unnecessary batch handling.
 *
 * @param vtable Vtable to fill
 * @return FB_STATUS_SUCCESS on success
 */
fb_status_t fb_autofill_batched_to_single(fb_backend_vtable_t *vtable);

/* ============================================================================
 * Strategy 3: Array → Strided (Lightweight)
 * ========================================================================= */

/**
 * @brief Generate strided batched operations from array batched operations
 *
 * If plugin provides sgemm_batch (array-of-pointers) but not
 * sgemm_batch_strided, generate wrapper that constructs pointer array
 * {A, A+stride, A+2*stride, ...} and calls array version.
 *
 * Lightweight overhead - array construction is cheap for small batches.
 *
 * @param vtable Vtable to fill
 * @return FB_STATUS_SUCCESS on success
 */
fb_status_t fb_autofill_array_to_strided(fb_backend_vtable_t *vtable);

/* ============================================================================
 * Strategy 4: Precision Promotion (Last Resort)
 * ========================================================================= */

/**
 * @brief Generate lower precision operations from higher precision operations
 *
 * If plugin provides only dgemm (FP64) but not sgemm (FP32), generate wrapper
 * that promotes float→double, computes, then demotes with overflow checking.
 *
 * Trade-off: 2× memory overhead + allocation, but guarantees correctness.
 * Dispatch system naturally avoids these (slow), but they prevent crashes.
 *
 * Applies to:
 * - FP64 → FP32 (~87 BLAS ops)
 * - C128 → C64 (~87 BLAS ops)
 * - INT32 → INT8 (~15 quantization ops)
 *
 * Logs warning when used: "Slow precision promotion active for <op>"
 *
 * @param vtable Vtable to fill
 * @return FB_STATUS_SUCCESS on success
 */
fb_status_t fb_autofill_precision_promotion(fb_backend_vtable_t *vtable);

/* ============================================================================
 * Utility Functions: Array Promotion/Demotion
 * ========================================================================= */

/**
 * @brief Promote float array to double (lossless)
 *
 * Allocates double-precision buffer and copies float values.
 * User must call fb_free_temp_buffer() on result.
 *
 * @param input Source float array
 * @param count Number of elements
 * @return Promoted double array, or NULL on allocation failure
 */
double *fb_promote_array_f32_to_f64(const float *input, size_t count);

/**
 * @brief Demote double array to float with overflow checking
 *
 * Converts double values to float, clamping to ±FLT_MAX if overflow detected.
 * Returns FB_STATUS_OVERFLOW if any value exceeded float range.
 *
 * @param input Source double array
 * @param output Destination float array (pre-allocated)
 * @param count Number of elements
 * @return FB_STATUS_SUCCESS or FB_STATUS_OVERFLOW
 */
fb_status_t fb_demote_array_f64_to_f32_checked(const double *input,
                                               float *output, size_t count);

/**
 * @brief Promote single complex array to double complex (lossless)
 *
 * @param input Source complex float array
 * @param count Number of elements
 * @return Promoted complex double array, or NULL on failure
 */
fb_complex_double_t *
fb_promote_array_c64_to_c128(const fb_complex_float_t *input, size_t count);

/**
 * @brief Demote double complex array to single complex with overflow checking
 *
 * @param input Source complex double array
 * @param output Destination complex float array (pre-allocated)
 * @param count Number of elements
 * @return FB_STATUS_SUCCESS or FB_STATUS_OVERFLOW
 */
fb_status_t
fb_demote_array_c128_to_c64_checked(const fb_complex_double_t *input,
                                    fb_complex_float_t *output, size_t count);

/**
 * @brief Promote int8 array to int32 (lossless)
 *
 * @param input Source int8 array
 * @param count Number of elements
 * @return Promoted int32 array, or NULL on failure
 */
int32_t *fb_promote_array_i8_to_i32(const int8_t *input, size_t count);

/**
 * @brief Demote int32 array to int8 with overflow checking
 *
 * Clamps values to [-128, 127] range.
 *
 * @param input Source int32 array
 * @param output Destination int8 array (pre-allocated)
 * @param count Number of elements
 * @return FB_STATUS_SUCCESS or FB_STATUS_OVERFLOW
 */
fb_status_t fb_demote_array_i32_to_i8_checked(const int32_t *input,
                                              int8_t *output, size_t count);

/* ============================================================================
 * Utility Functions: Temporary Buffer Management
 * ========================================================================= */

/**
 * @brief Allocate temporary buffer for auto-fill wrappers
 *
 * Uses platform-specific allocation (aligned malloc on modern platforms).
 * Buffers must be freed with fb_free_temp_buffer().
 *
 * @param size Buffer size in bytes
 * @return Allocated buffer, or NULL on failure
 */
void *fb_allocate_temp_buffer(size_t size);

/**
 * @brief Free temporary buffer allocated by auto-fill system
 *
 * Safe to call with NULL pointer.
 *
 * @param buffer Buffer to free
 */
void fb_free_temp_buffer(void *buffer);

/* ============================================================================
 * Utility Functions: Active Vtable Access
 * ========================================================================= */

/**
 * @brief Get the currently active backend vtable
 *
 * Used by auto-generated wrappers to access backend operations.
 * Returns NULL if no plugin is loaded.
 *
 * @return Current vtable, or NULL
 */
const fb_backend_vtable_t *fb_get_active_vtable(void);

/**
 * @brief Set the active backend vtable
 *
 * Called by plugin system after loading backend.
 *
 * @param vtable Vtable to make active
 */
void fb_set_active_vtable(const fb_backend_vtable_t *vtable);

/* ============================================================================
 * Internal Helper Functions (Strategy Implementations)
 * ========================================================================= */

/**
 * @brief Fill missing GEMM entries from gemm_unified
 *
 * Internal helper called by fb_autofill_unified_to_specific().
 * Generates wrappers for sgemm, dgemm, cgemm, zgemm from gemm_unified.
 *
 * @param vtable Vtable to fill
 */
void fb_autofill_gemm_from_unified(fb_backend_vtable_t *vtable);

/**
 * @brief Fill missing normalization entries from normalize_unified
 *
 * Internal helper called by fb_autofill_unified_to_specific().
 * Generates wrappers for batch_norm, layer_norm, instance_norm, group_norm,
 * z-score, standardize, normalize, min_max_scale from normalize_unified.
 *
 * @param vtable Vtable to fill
 */
void fb_autofill_normalization_from_unified(fb_backend_vtable_t *vtable);

/**
 * @brief Fill missing reduction entries from reduce_unified
 *
 * Internal helper called by fb_autofill_unified_to_specific().
 * Generates wrappers for tensor reductions (sum, max, min, mean, norm),
 * parallel primitive reductions/scans, statistical reductions (sum, mean,
 * variance, std_dev), and collective communication reductions (AllReduce,
 * Reduce, ReduceScatter) from reduce_unified.
 *
 * @param vtable Vtable to fill
 */
void fb_autofill_reduction_from_unified(fb_backend_vtable_t *vtable);

/**
 * @brief Fill missing gemm_unified from specific operations
 *
 * Internal helper called by fb_autofill_specific_to_unified().
 * Generates unified GEMM dispatcher from sgemm/dgemm/cgemm/zgemm.
 *
 * @param vtable Vtable to fill
 */
void fb_autofill_gemm_unified_from_specific(fb_backend_vtable_t *vtable);

/**
 * @brief Fill missing single-operation entries from batched operations
 *
 * Internal helper called by fb_autofill_batched_to_single().
 * Generates wrappers for single-operation calls from batched variants.
 * Covers GEMM (sgemm_batch_strided→sgemm, etc.), BLAS L1 (saxpy_batch→saxpy),
 * and BLAS L2 (sgemv_batch_strided→sgemv).
 *
 * @param vtable Vtable to fill
 */
void fb_autofill_all_from_batched(fb_backend_vtable_t *vtable);

/**
 * @brief Fill missing strided batched entries from array-based batched
 * operations
 *
 * Internal helper called by fb_autofill_array_to_strided().
 * Generates wrappers that construct pointer arrays from strided parameters.
 * Covers GEMM (sgemm_batch→sgemm_batch_strided, etc.) and GEMV variants.
 *
 * @param vtable Vtable to fill
 */
void fb_autofill_all_strided_from_array(fb_backend_vtable_t *vtable);

/**
 * @brief Fill missing lower-precision entries using higher-precision promotion
 *
 * Internal helper called by fb_autofill_precision_promotion().
 * Generates last-resort wrappers that promote lower precision to higher
 * precision, execute with higher-precision backend, then demote with overflow
 * checking. Covers FP32←FP64 (SGEMM←DGEMM) and C64←C128 (CGEMM←ZGEMM).
 *
 * @param vtable Vtable to fill
 */
void fb_autofill_all_precision_promotion(fb_backend_vtable_t *vtable);

/* ============================================================================ 
 * Extended Op Dispatch (ext_ops[] sync + lookup)
 * ========================================================================= */

/**
 * @brief Mirror every populated named vtable field into vtable->ext_ops[op_id].
 *
 * Must be called once after the vtable is fully populated — typically at the
 * end of fb_finalize_plugin_vtable().  Does not overwrite non-NULL ext_ops
 * slots that backends have pre-populated with specialised variants.
 *
 * @param vtable Vtable to synchronise
 */
void fb_vtable_sync_ext_ops(fb_backend_vtable_t *vtable);

/**
 * @brief Reverse sync: propagate ext_ops[][FB_CONV_CBLAS] → named vtable
 * fields.
 *
 * Call this after fb_enumerate_and_populate() has filled ext_ops from a DLL
 * scan to make the named typed fields (vtable->saxpy, vtable->sgemm, …)
 * reachable by the judge and dispatch code.  Only fills fields that are NULL.
 *
 * @param vtable Vtable to update
 */
void fb_vtable_fill_named_from_ext_ops(fb_backend_vtable_t *vtable);

/**
 * @brief Return the function pointer for op_id, or NULL if unsupported.
 *
 * Requires a prior call to fb_vtable_sync_ext_ops().
 * Call is O(1) — direct array index into vtable->ext_ops[].
 *
 * @param vtable   Backend vtable to query
 * @param op_id    FB_OP_* constant from judge_op_ids.h
 * @return         Type-erased fn pointer, NULL if not supported
 */
fb_generic_fn fb_vtable_get_op(const fb_backend_vtable_t *vtable, uint32_t op_id);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_VTABLE_AUTOFILL_H */
