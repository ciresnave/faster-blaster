/**
 * @file judge_metadata.c
 * @brief Metadata registry implementation.
 *
 * Provides per-type constant arrays (eps, element sizes, names) and
 * lookup helpers that patch unregistered entries with conservative
 * Phase 1 defaults before returning them to callers.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "judge_types.h"
#include "judge_metadata.h"
#include <math.h>

/* Forward declaration — defined in judge_metadata_table.c */
extern const fb_op_judge_meta_t fb_op_judge_table[FB_JUDGE_MAX_OPERATIONS];

/* =========================================================================
 * Per-dtype constant arrays
 * ========================================================================= */

/* Machine epsilon (half ULP of 1.0 in the given type) */
const double fb_dtype_eps[FB_DTYPE_COUNT] = {
    [FB_DTYPE_F32]  = 1.19209290e-07,  /* FLT_EPSILON  */
    [FB_DTYPE_F64]  = 2.22044605e-16,  /* DBL_EPSILON  */
    [FB_DTYPE_CF32] = 1.19209290e-07,  /* component eps */
    [FB_DTYPE_CF64] = 2.22044605e-16,
    [FB_DTYPE_F16]  = 9.76562500e-04,  /* half epsilon  */
    [FB_DTYPE_BF16] = 7.81250000e-03,  /* bfloat16 eps  */
    [FB_DTYPE_I8]   = 0.0,             /* integer: no eps */
    [FB_DTYPE_I32]  = 0.0,
};

/* Canonical name strings */
const char *const fb_dtype_name[FB_DTYPE_COUNT] = {
    [FB_DTYPE_F32]  = "float32",
    [FB_DTYPE_F64]  = "float64",
    [FB_DTYPE_CF32] = "complex64",
    [FB_DTYPE_CF64] = "complex128",
    [FB_DTYPE_F16]  = "float16",
    [FB_DTYPE_BF16] = "bfloat16",
    [FB_DTYPE_I8]   = "int8",
    [FB_DTYPE_I32]  = "int32",
};

/* Element size in bytes */
const size_t fb_dtype_element_size[FB_DTYPE_COUNT] = {
    [FB_DTYPE_F32]  = 4,
    [FB_DTYPE_F64]  = 8,
    [FB_DTYPE_CF32] = 8,
    [FB_DTYPE_CF64] = 16,
    [FB_DTYPE_F16]  = 2,
    [FB_DTYPE_BF16] = 2,
    [FB_DTYPE_I8]   = 1,
    [FB_DTYPE_I32]  = 4,
};

/* =========================================================================
 * Conservative defaults (PRECISION_GUARANTEES.md §4)
 * ========================================================================= */

static const fb_op_judge_meta_t k_default_meta = {
    .name                        = "unknown",
    .archetype                   = FB_JUDGE_DIRECT,
    .uniqueness                  = FB_OUTPUT_UNIQUE,
    .default_score_policy        = FB_SCORE_MIN_PRIMARY,
    .primary_metrics             = FB_JUDGE_LIMIT_DIRECT,
    .max_certifiable_f32         = 4,
    .max_certifiable_f64         = 10,
    .ortho_slack_f32             = 1e-5f,
    .ortho_slack_f64             = 1e-10,
    .cluster_threshold_multiplier = 4.0,
    .is_in_place                 = false,
};

/* Indicates a slot has not been explicitly registered */
static inline int entry_is_stub(const fb_op_judge_meta_t *e)
{
    return (e->max_certifiable_f64 == 0 && e->archetype == 0);
}

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * @brief Retrieve metadata for an operation, falling back to conservative
 *        defaults for unregistered slots.
 *
 * Returns a pointer to the static table entry (or the static default).
 * Never returns NULL.
 *
 * @param op_id   Operation ID (0 to FB_JUDGE_MAX_OPERATIONS-1).
 */
const fb_op_judge_meta_t *fb_judge_meta_get(uint32_t op_id)
{
    if (op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) return &k_default_meta;
    const fb_op_judge_meta_t *row = &fb_op_judge_table[op_id];
    if (entry_is_stub(row)) return &k_default_meta;
    return row;
}

/**
 * @brief Return the oracle-ceiling digit count for a given op and dtype.
 *
 * For complex types the component element type determines the ceiling.
 *
 * @param op_id   Operation ID.
 * @param dtype   Data type.
 * @return        Maximum certifiable digits (uint8_t).
 */
uint8_t fb_judge_meta_oracle_ceiling(uint32_t op_id, fb_dtype_t dtype)
{
    const fb_op_judge_meta_t *meta = fb_judge_meta_get(op_id);

    switch (dtype) {
    case FB_DTYPE_F32:
    case FB_DTYPE_CF32:
    case FB_DTYPE_F16:
    case FB_DTYPE_BF16:
        return meta->max_certifiable_f32;
    case FB_DTYPE_F64:
    case FB_DTYPE_CF64:
    case FB_DTYPE_I32:
        return meta->max_certifiable_f64;
    case FB_DTYPE_I8:
        return 0;   /* exact integer arithmetic */
    default:
        return meta->max_certifiable_f32;
    }
}

/**
 * @brief Return the cluster-detection threshold: cluster_threshold_multiplier × eps
 *        for the given dtype. Callers scale by their data magnitude.
 *
 * @param op_id      Operation ID.
 * @param dtype      Data type.
 * @return           Absolute cluster threshold (eps-multiples; scale by magnitude).
 */
double fb_judge_cluster_threshold(uint32_t op_id, fb_dtype_t dtype)
{
    const fb_op_judge_meta_t *meta = fb_judge_meta_get(op_id);

    double eps;
    switch (dtype) {
    case FB_DTYPE_F32:
    case FB_DTYPE_CF32:
        eps = fb_dtype_eps[FB_DTYPE_F32]; break;
    case FB_DTYPE_F64:
    case FB_DTYPE_CF64:
        eps = fb_dtype_eps[FB_DTYPE_F64]; break;
    case FB_DTYPE_F16:
        eps = fb_dtype_eps[FB_DTYPE_F16]; break;
    case FB_DTYPE_BF16:
        eps = fb_dtype_eps[FB_DTYPE_BF16]; break;
    default:
        eps = 1e-7;
    }

    return meta->cluster_threshold_multiplier * eps;
}

/**
 * @brief Return the orthogonality slack for Q^T Q ≈ I checks.
 *
 * @param op_id   Operation ID.
 * @param dtype   Data type (selects f32 vs f64 slack).
 * @return        Slack value as float.
 */
float fb_judge_ortho_slack(uint32_t op_id, fb_dtype_t dtype)
{
    const fb_op_judge_meta_t *meta = fb_judge_meta_get(op_id);
    switch (dtype) {
    case FB_DTYPE_F64:
    case FB_DTYPE_CF64:
        return (float)meta->ortho_slack_f64;
    default:
        return meta->ortho_slack_f32;
    }
}

/**
 * @brief Return true for operations that have Phase-1 archetype support
 *        (FB_JUDGE_DIRECT or FB_JUDGE_INDEX).
 *
 * @param op_id  Operation ID.
 * @return       Non-zero if Phase 1 judge support is available.
 */
bool fb_judge_meta_is_phase1(uint32_t op_id)
{
    if (op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) return false;
    const fb_op_judge_meta_t *row = &fb_op_judge_table[op_id];
    if (entry_is_stub(row)) return false;
    return (row->archetype == FB_JUDGE_DIRECT ||
            row->archetype == FB_JUDGE_INDEX);
}
