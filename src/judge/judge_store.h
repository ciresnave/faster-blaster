/**
 * @file judge_store.h
 * @brief On-disk persistence for fb_precision_profile_t objects.
 *
 * Profiles are stored as binary blobs under:
 *   <profile_dir>/judge_profiles/<op_name>_be<B>_dev<D>_sc<S>_dt<T>.fbjp
 *
 * The file format includes a magic number, module+corpus version bytes,
 * and the raw profile struct.  Files from a different version are silently
 * discarded (treated as non-existent).
 *
 * Not part of the public API.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_JUDGE_STORE_H
#define FB_JUDGE_STORE_H

#include <stdbool.h>
#include <stdint.h>
#include "../../include/faster-blaster/judge.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Corpus generator version — increment whenever corpus algorithm changes. */
#define FB_CORPUS_VERSION  1

/**
 * Save a profile to disk.
 *
 * @param profile_dir  Root directory (created if absent).
 * @param profile      Profile to write.
 * @return FB_JUDGE_OK, FB_JUDGE_ERR_IO on write failure.
 */
fb_judge_status_t fb_judge_store_save(
    const char                   *profile_dir,
    const fb_precision_profile_t *profile
);

/**
 * Load a profile from disk.
 *
 * @return FB_JUDGE_OK on success.
 *         FB_JUDGE_ERR_NO_PROFILE if the file doesn't exist or version mismatch.
 *         FB_JUDGE_ERR_IO on read error.
 */
fb_judge_status_t fb_judge_store_load(
    const char             *profile_dir,
    const char             *op_name,
    uint32_t                backend_id,
    uint32_t                device_id,
    uint8_t                 size_class,
    uint8_t                 dtype,
    fb_precision_profile_t *profile_out
);

/**
 * Delete all stored profiles for the given (backend_id, device_id) pair.
 * Silently ignores missing files.
 */
void fb_judge_store_invalidate(
    const char *profile_dir,
    uint32_t    backend_id,
    uint32_t    device_id
);

/**
 * Return true if the profile for (backend_id, device_id) was written with the
 * current FB_JUDGE_MODULE_VERSION and FB_CORPUS_VERSION.
 */
bool fb_judge_store_profiles_are_current(
    const char *profile_dir,
    uint32_t    backend_id,
    uint32_t    device_id
);

/**
 * Enumerate all operation IDs that have at least one profile stored on disk
 * for the given (backend_id, device_id) pair.  Scans the profiles subdirectory
 * for .fbjp files and reverse-maps their op_name to an op_id via the metadata
 * table.
 *
 * @param profile_dir  Root directory passed to fb_judge_store_save().
 * @param backend_id   Backend filter.
 * @param device_id    Device filter.
 * @param op_ids_out   Caller-owned buffer to receive op IDs.  May be NULL to
 *                     count only (useful for a two-pass allocation pattern).
 * @param max_ops      Capacity of op_ids_out in elements (ignored when NULL).
 * @return             Number of distinct profiled ops found.  When greater than
 *                     max_ops, only the first max_ops entries are written.
 */
uint32_t fb_judge_store_list_profiled_ops(
    const char *profile_dir,
    uint32_t    backend_id,
    uint32_t    device_id,
    uint32_t   *op_ids_out,
    uint32_t    max_ops
);

/**
 * Convert a metadata table name (e.g. "FB_OP_SAXPY") to the canonical
 * lower-case short form used in profile file names (e.g. "saxpy").
 * Strips the "FB_OP_" prefix (if present) and lower-cases the remainder.
 * Safe with NULL inputs; always NUL-terminates buf.
 */
void fb_judge_meta_to_canonical_name(
    const char *meta_name,
    char       *buf,
    size_t      buf_size
);

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_STORE_H */
