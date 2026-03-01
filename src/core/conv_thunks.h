/**
 * @file conv_thunks.h
 * @brief Cross-convention calling-convention thunks for BLAS/LAPACK operations.
 *
 * Provides static per-op thunk arrays that bridge the three calling conventions
 * (@ref fb_conv_t) supported by faster-blaster:
 *
 *   - FB_CONV_CBLAS   — scalars passed by value, C linkage
 *   - FB_CONV_FORTRAN — all arguments passed by pointer, trailing underscore
 *   - FB_CONV_REF     — C convention with _ref suffix (identical ABI to CBLAS)
 *
 * ## Architecture
 *
 * Since C has no closures, thunks use per-op global function-pointer slots
 * (`g_cblas_fn[]` / `g_fortran_fn[]`) to reach the "other side" implementation.
 * These globals are populated at vtable-finalisation time by
 * `fb_install_conv_thunks()`.
 *
 * **Single-backend limitation**: the global arrays hold one function pointer per
 * op, so only one active backend is fully supported.  Multi-backend operation
 * would require JIT-generated code trampolines — a future enhancement.
 *
 * ## Usage (in vtable_autofill.c — Strategy 5)
 *
 * ```c
 * #include "conv_thunks.h"
 *
 * // After Strategies 1-4, fill empty convention slots with thunks:
 * for (uint32_t op = 0; op < FB_JUDGE_MAX_OPERATIONS; op++) {
 *     fb_install_conv_thunks(vtable, op);
 * }
 * ```
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#pragma once

#include <stdint.h>
#include "../backends/backend_interface.h"   /* fb_backend_vtable_t, fb_conv_t */
#include "../judge/judge_op_ids.h"           /* FB_OP_* constants             */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Pre-built thunk: CBLAS → Fortran convention.
 *
 * `k_cblas_to_fortran_thunks[op]` is a fb_generic_fn that wraps the CBLAS
 * implementation of @p op and exposes it with Fortran linkage (all-pointer).
 * NULL when no thunk is available for the given op.
 */
extern fb_generic_fn const k_cblas_to_fortran_thunks[FB_JUDGE_MAX_OPERATIONS];

/**
 * @brief Pre-built thunk: Fortran → CBLAS convention.
 *
 * `k_fortran_to_cblas_thunks[op]` is a fb_generic_fn that wraps the Fortran
 * implementation of @p op and exposes it with CBLAS linkage (scalars by value).
 * NULL when no thunk is available for the given op.
 */
extern fb_generic_fn const k_fortran_to_cblas_thunks[FB_JUDGE_MAX_OPERATIONS];

/**
 * @brief Populate per-op global backing pointers and patch any empty convention
 *        slots in @p vtable for the given @p op_id using the appropriate thunk.
 *
 * Called by `fb_finalize_plugin_vtable()` (Strategy 5) for every op after
 * Strategies 1-4 have run.  Safe to call when both conv slots are already
 * filled — no-op in that case.
 *
 * @param vtable   The vtable being finalised.  Must not be NULL.
 * @param op_id    Operation index (0 … FB_JUDGE_MAX_OPERATIONS-1).
 */
void fb_install_conv_thunks(fb_backend_vtable_t *vtable, uint32_t op_id);

#ifdef __cplusplus
}
#endif
