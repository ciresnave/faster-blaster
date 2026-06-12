/**
 * @file judge_metadata.h
 * @brief Operation metadata registry — lookup and query utilities.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_METADATA_H
#define FB_JUDGE_METADATA_H

#include "judge_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Look up the metadata for an operation.
 * Returns NULL if op_id >= FB_JUDGE_MAX_OPERATIONS.
 */
const fb_op_judge_meta_t* fb_judge_meta_get(uint32_t op_id);

/**
 * Return the oracle precision ceiling for op_id and dtype.
 * For non-FP32/FP64 types, returns the closest applicable ceiling.
 */
uint8_t fb_judge_meta_oracle_ceiling(uint32_t op_id, fb_dtype_t dtype);

/**
 * Return the cluster detection threshold (eps multiplier × machine epsilon)
 * for spectral ops with the given dtype.
 */
double fb_judge_cluster_threshold(uint32_t op_id, fb_dtype_t dtype);

/**
 * Return the ortho_slack for the given op and dtype.
 * This is the number of digits of orthogonality that can fall below the
 * value/reconstruction score before it is considered a failing metric.
 */
float fb_judge_ortho_slack(uint32_t op_id, fb_dtype_t dtype);

/**
 * Return true if op_id is in the range [begin, end) of a known
 * archetype group, for fast batch-initialization of the metadata table.
 */
bool fb_judge_meta_is_phase1(uint32_t op_id);  /* DIRECT archetype ops */

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_METADATA_H */
