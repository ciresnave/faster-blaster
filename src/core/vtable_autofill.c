/**
 * @file vtable_autofill.c
 * @brief Implementation of automatic vtable completion system
 *
 * @copyright Copyright (c) 2025-2026
 * @license MIT OR Apache-2.0
 */

#include "../../include/faster-blaster/vtable_autofill.h"
#include "../backends/backend_interface.h"
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global active vtable pointer */
static const fb_backend_vtable_t *g_active_vtable = NULL;

/* ============================================================================
 * Active Vtable Management
 * ========================================================================= */

const fb_backend_vtable_t *fb_get_active_vtable(void) {
  return g_active_vtable;
}

void fb_set_active_vtable(const fb_backend_vtable_t *vtable) {
  g_active_vtable = vtable;
}

/* ============================================================================
 * Core Auto-Fill Entry Point
 * ========================================================================= */

fb_status_t fb_finalize_plugin_vtable(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  fb_status_t status;

  /* Strategy 1: Unified ↔ Specific (zero overhead) */
  status = fb_autofill_unified_to_specific(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Unified→Specific autofill failed\n");
  }

  status = fb_autofill_specific_to_unified(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Specific→Unified autofill failed\n");
  }

  /* Strategy 2: Batched → Single (zero overhead) */
  status = fb_autofill_batched_to_single(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Batched→Single autofill failed\n");
  }

  /* Strategy 3: Array → Strided (lightweight) */
  status = fb_autofill_array_to_strided(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Array→Strided autofill failed\n");
  }

  /* Strategy 4: Precision promotion (last resort) */
  status = fb_autofill_precision_promotion(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Precision promotion autofill failed\n");
  }

  /* Final step: mirror named fields → ext_ops[] for uniform op-level dispatch */
  fb_vtable_sync_ext_ops(vtable);

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Strategy 1: Unified ↔ Specific
 * ========================================================================= */

/* External wrapper implementations */
extern void fb_autofill_gemm_from_unified(fb_backend_vtable_t *vtable);
#ifdef FB_EXTENDED_VTABLE_SUPPORT
extern void fb_autofill_normalization_from_unified(fb_backend_vtable_t *vtable);
extern void fb_autofill_reduction_from_unified(fb_backend_vtable_t *vtable);
#endif

fb_status_t fb_autofill_unified_to_specific(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  /* GEMM wrappers: sgemm, dgemm, cgemm, zgemm from gemm_unified */
  fb_autofill_gemm_from_unified(vtable);

#ifdef FB_EXTENDED_VTABLE_SUPPORT
  /* Normalization wrappers: batch_norm, layer_norm, etc. from normalize_unified
   */
  fb_autofill_normalization_from_unified(vtable);

  /* Reduction wrappers: tensor/parallel/stats/collective reductions from
   * reduce_unified */
  fb_autofill_reduction_from_unified(vtable);
#endif

  return FB_STATUS_SUCCESS;
}

/* External dispatcher implementations */
extern void fb_autofill_gemm_unified_from_specific(fb_backend_vtable_t *vtable);

fb_status_t fb_autofill_specific_to_unified(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  /* GEMM unified dispatcher: gemm_unified from sgemm/dgemm/cgemm/zgemm */
  fb_autofill_gemm_unified_from_specific(vtable);

  /* TODO: Phase 2.5 - Normalize unified dispatcher from specific normalization
   * ops */
  /* TODO: Phase 2.6 - Reduce unified dispatcher from specific reduction ops */

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Strategy 2: Batched → Single (TODO: Implement in Phase 3)
 * ========================================================================= */

#ifdef FB_EXTENDED_VTABLE_SUPPORT
/* External wrapper implementations */
extern void fb_autofill_all_from_batched(fb_backend_vtable_t *vtable);
#endif

fb_status_t fb_autofill_batched_to_single(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

#ifdef FB_EXTENDED_VTABLE_SUPPORT
  /* Install single wrappers from batched operations */
  fb_autofill_all_from_batched(vtable);
#endif

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Strategy 3: Array → Strided (TODO: Implement in Phase 3)
 * ========================================================================= */

#ifdef FB_EXTENDED_VTABLE_SUPPORT
/* External wrapper implementations */
extern void fb_autofill_all_strided_from_array(fb_backend_vtable_t *vtable);
#endif

fb_status_t fb_autofill_array_to_strided(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

#ifdef FB_EXTENDED_VTABLE_SUPPORT
  /* Install strided batched wrappers from array-based operations */
  fb_autofill_all_strided_from_array(vtable);
#endif

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Strategy 4: Precision Promotion (TODO: Implement in Phase 5)
 * ========================================================================= */

/* External wrapper implementations */
extern void fb_autofill_all_precision_promotion(fb_backend_vtable_t *vtable);

fb_status_t fb_autofill_precision_promotion(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  /* Install precision promotion fallback wrappers (last resort for correctness)
   */
  fb_autofill_all_precision_promotion(vtable);

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Utility Functions: Array Promotion/Demotion
 * ========================================================================= */

double *fb_promote_array_f32_to_f64(const float *input, size_t count) {
  if (!input || count == 0) {
    return NULL;
  }

  double *output = (double *)malloc(count * sizeof(double));
  if (!output) {
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    output[i] = (double)input[i];
  }

  return output;
}

fb_status_t fb_demote_array_f64_to_f32_checked(const double *input,
                                               float *output, size_t count) {
  if (!input || !output || count == 0) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  fb_status_t status = FB_STATUS_SUCCESS;

  for (size_t i = 0; i < count; i++) {
    double val = input[i];

    /* Check for overflow */
    if (val > (double)FLT_MAX) {
      output[i] = FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else if (val < -(double)FLT_MAX) {
      output[i] = -FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else {
      output[i] = (float)val;
    }
  }

  if (status == FB_STATUS_OVERFLOW) {
    printf("[faster-blaster] Warning: FP64→FP32 demotion overflow detected, "
           "values clamped to ±FLT_MAX\n");
  }

  return status;
}

fb_complex_double_t *
fb_promote_array_c64_to_c128(const fb_complex_float_t *input, size_t count) {
  if (!input || count == 0) {
    return NULL;
  }

  fb_complex_double_t *output =
      (fb_complex_double_t *)malloc(count * sizeof(fb_complex_double_t));
  if (!output) {
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    __real__(output[i]) = (double)__real__(input[i]);
    __imag__(output[i]) = (double)__imag__(input[i]);
  }

  return output;
}

fb_status_t
fb_demote_array_c128_to_c64_checked(const fb_complex_double_t *input,
                                    fb_complex_float_t *output, size_t count) {
  if (!input || !output || count == 0) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  fb_status_t status = FB_STATUS_SUCCESS;

  for (size_t i = 0; i < count; i++) {
    double real = __real__(input[i]);
    double imag = __imag__(input[i]);

    /* Check for overflow in real part */
    if (real > (double)FLT_MAX) {
      __real__(output[i]) = FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else if (real < -(double)FLT_MAX) {
      __real__(output[i]) = -FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else {
      __real__(output[i]) = (float)real;
    }

    /* Check for overflow in imaginary part */
    if (imag > (double)FLT_MAX) {
      __imag__(output[i]) = FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else if (imag < -(double)FLT_MAX) {
      __imag__(output[i]) = -FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else {
      __imag__(output[i]) = (float)imag;
    }
  }

  if (status == FB_STATUS_OVERFLOW) {
    printf("[faster-blaster] Warning: C128→C64 demotion overflow detected\n");
  }

  return status;
}

int32_t *fb_promote_array_i8_to_i32(const int8_t *input, size_t count) {
  if (!input || count == 0) {
    return NULL;
  }

  int32_t *output = (int32_t *)malloc(count * sizeof(int32_t));
  if (!output) {
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    output[i] = (int32_t)input[i];
  }

  return output;
}

fb_status_t fb_demote_array_i32_to_i8_checked(const int32_t *input,
                                              int8_t *output, size_t count) {
  if (!input || !output || count == 0) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  fb_status_t status = FB_STATUS_SUCCESS;

  for (size_t i = 0; i < count; i++) {
    int32_t val = input[i];

    /* Check for overflow */
    if (val > 127) {
      output[i] = 127;
      status = FB_STATUS_OVERFLOW;
    } else if (val < -128) {
      output[i] = -128;
      status = FB_STATUS_OVERFLOW;
    } else {
      output[i] = (int8_t)val;
    }
  }

  if (status == FB_STATUS_OVERFLOW) {
    printf("[faster-blaster] Warning: INT32→INT8 demotion overflow detected, "
           "values clamped to [-128, 127]\n");
  }

  return status;
}

/* ============================================================================
 * Utility Functions: Temporary Buffer Management
 * ========================================================================= */

void *fb_allocate_temp_buffer(size_t size) {
  if (size == 0) {
    return NULL;
  }

  /* Use standard malloc for now - could use aligned_alloc on modern platforms
   */
  return malloc(size);
}

void fb_free_temp_buffer(void *buffer) {
  if (buffer) {
    free(buffer);
  }
}
