/**
 * @file judge_store.h
 * @brief On-disk persistence for fb_precision_profile_t objects.
 *
 * Profiles are stored as binary blobs under:
 *   <profile_dir>/judge_profiles/op<N>_be<B>_dev<D>_sc<S>_dt<T>.fbjp
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
    uint32_t                op_id,
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

#ifdef __cplusplus
}
#endif

#endif /* FB_JUDGE_STORE_H */
