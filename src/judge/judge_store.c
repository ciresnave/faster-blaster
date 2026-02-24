/**
 * @file judge_store.c
 * @brief Binary profile store: save, load, invalidate, and version check.
 *
 * File format (.fbjp):
 *   [4]  Magic 0x46424A50 ("FBJP")
 *   [4]  FB_JUDGE_MODULE_VERSION (uint32_t)
 *   [4]  FB_CORPUS_VERSION       (uint32_t)
 *   [4]  op_id      (uint32_t, redundant — cross-check on load)
 *   [4]  backend_id (uint32_t)
 *   [4]  device_id  (uint32_t)
 *   [1]  size_class (uint8_t)
 *   [1]  dtype      (uint8_t)
 *   [2]  padding
 *   [sizeof(fb_precision_profile_t)] The profile struct
 *
 * Invalidation file (.inv):
 *   <profile_dir>/judge_profiles/inv_be<backend_id>_dev<device_id>
 *   Presence of this file makes all profiles for (be, dev) invalid until
 *   any profile is successfully saved with current versions.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "judge_store.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Platform-specific headers for directory creation. */
#ifdef _WIN32
#  include <direct.h>   /* _mkdir */
#  define FB_MKDIR(path)  _mkdir(path)
#else
#  include <sys/stat.h>
#  define FB_MKDIR(path)  mkdir((path), 0755)
#endif

/* =========================================================================
 * Constants
 * ========================================================================= */

#define FB_STORE_MAGIC         UINT32_C(0x46424A50)  /* "FBJP" */
#define FB_STORE_SUBDIR_NAME   "judge_profiles"
#define FB_STORE_MAX_PATH      512

/* =========================================================================
 * File header (binary; packed to avoid alignment surprises)
 * ========================================================================= */

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t module_version;
    uint32_t corpus_version;
    uint32_t op_id;
    uint32_t backend_id;
    uint32_t device_id;
    uint8_t  size_class;
    uint8_t  dtype;
    uint8_t  _pad[2];
} fb_store_header_t;
#pragma pack(pop)

/* =========================================================================
 * Private helpers
 * ========================================================================= */

/** Build path to the judge_profiles subdirectory. */
static void build_subdir_path(
    const char *profile_dir, char *out, size_t out_size)
{
    snprintf(out, out_size, "%s/%s", profile_dir, FB_STORE_SUBDIR_NAME);
}

/** Build path to a .fbjp profile file. */
static void build_profile_path(
    const char *profile_dir,
    uint32_t op_id, uint32_t backend_id, uint32_t device_id,
    uint8_t size_class, uint8_t dtype,
    char *out, size_t out_size)
{
    snprintf(out, out_size,
             "%s/%s/op%u_be%u_dev%u_sc%u_dt%u.fbjp",
             profile_dir, FB_STORE_SUBDIR_NAME,
             op_id, backend_id, device_id, (unsigned)size_class, (unsigned)dtype);
}

/** Build path to an invalidation marker file. */
static void build_inv_path(
    const char *profile_dir,
    uint32_t backend_id, uint32_t device_id,
    char *out, size_t out_size)
{
    snprintf(out, out_size,
             "%s/%s/inv_be%u_dev%u.inv",
             profile_dir, FB_STORE_SUBDIR_NAME, backend_id, device_id);
}

/**
 * Attempt to create the full directory chain up to and including 'path'.
 * Ignores EEXIST. Returns 0 on success, -1 on error.
 */
static int ensure_dir(const char *path)
{
    /* Make a mutable copy to walk and create each component. */
    char tmp[FB_STORE_MAX_PATH];
    snprintf(tmp, sizeof(tmp), "%s", path);

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char saved = *p;
            *p = '\0';
            int rc = FB_MKDIR(tmp);
            if (rc != 0 && errno != EEXIST)
                return -1;
            *p = saved;
        }
    }
    /* Create the leaf directory. */
    int rc = FB_MKDIR(tmp);
    return (rc == 0 || errno == EEXIST) ? 0 : -1;
}

/** Return true if the invalidation marker exists for (backend_id, device_id). */
static bool inv_marker_exists(
    const char *profile_dir, uint32_t backend_id, uint32_t device_id)
{
    char path[FB_STORE_MAX_PATH];
    build_inv_path(profile_dir, backend_id, device_id, path, sizeof(path));
    FILE *f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

/** Remove the invalidation marker file (silently ignore errors). */
static void remove_inv_marker(
    const char *profile_dir, uint32_t backend_id, uint32_t device_id)
{
    char path[FB_STORE_MAX_PATH];
    build_inv_path(profile_dir, backend_id, device_id, path, sizeof(path));
    remove(path);
}

/* =========================================================================
 * fb_judge_store_save
 * ========================================================================= */

fb_judge_status_t fb_judge_store_save(
    const char                   *profile_dir,
    const fb_precision_profile_t *profile)
{
    if (!profile_dir || !profile)
        return FB_JUDGE_ERR_IO;

    /* Ensure directory exists. */
    char subdir[FB_STORE_MAX_PATH];
    build_subdir_path(profile_dir, subdir, sizeof(subdir));
    if (ensure_dir(subdir) < 0)
        return FB_JUDGE_ERR_IO;

    /* Build file path. */
    char path[FB_STORE_MAX_PATH];
    build_profile_path(profile_dir,
                       profile->op_id, profile->backend_id, profile->device_id,
                       profile->size_class, profile->dtype,
                       path, sizeof(path));

    FILE *f = fopen(path, "wb");
    if (!f)
        return FB_JUDGE_ERR_IO;

    /* Write header. */
    fb_store_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic          = FB_STORE_MAGIC;
    hdr.module_version = (uint32_t)FB_JUDGE_MODULE_VERSION;
    hdr.corpus_version = (uint32_t)FB_CORPUS_VERSION;
    hdr.op_id          = profile->op_id;
    hdr.backend_id     = profile->backend_id;
    hdr.device_id      = profile->device_id;
    hdr.size_class     = profile->size_class;
    hdr.dtype          = profile->dtype;

    int ok = (fwrite(&hdr, sizeof(hdr), 1, f) == 1) &&
             (fwrite(profile, sizeof(*profile), 1, f) == 1);

    int err = ferror(f);
    fclose(f);

    if (!ok || err) {
        remove(path);  /* don't leave a partial file */
        return FB_JUDGE_ERR_IO;
    }

    /* Successful save — remove any stale invalidation marker. */
    remove_inv_marker(profile_dir, profile->backend_id, profile->device_id);
    return FB_JUDGE_OK;
}

/* =========================================================================
 * fb_judge_store_load
 * ========================================================================= */

fb_judge_status_t fb_judge_store_load(
    const char             *profile_dir,
    uint32_t                op_id,
    uint32_t                backend_id,
    uint32_t                device_id,
    uint8_t                 size_class,
    uint8_t                 dtype,
    fb_precision_profile_t *profile_out)
{
    if (!profile_dir || !profile_out)
        return FB_JUDGE_ERR_IO;

    /* If invalidation marker is present, report missing. */
    if (inv_marker_exists(profile_dir, backend_id, device_id))
        return FB_JUDGE_ERR_NO_PROFILE;

    char path[FB_STORE_MAX_PATH];
    build_profile_path(profile_dir, op_id, backend_id, device_id,
                       size_class, dtype, path, sizeof(path));

    FILE *f = fopen(path, "rb");
    if (!f)
        return FB_JUDGE_ERR_NO_PROFILE;

    fb_store_header_t hdr;
    size_t read_ok = fread(&hdr, sizeof(hdr), 1, f);
    if (read_ok != 1) {
        fclose(f);
        return FB_JUDGE_ERR_IO;
    }

    /* Verify magic and version. */
    if (hdr.magic          != FB_STORE_MAGIC          ||
        hdr.module_version != (uint32_t)FB_JUDGE_MODULE_VERSION ||
        hdr.corpus_version != (uint32_t)FB_CORPUS_VERSION       ||
        hdr.op_id          != op_id       ||
        hdr.backend_id     != backend_id  ||
        hdr.device_id      != device_id   ||
        hdr.size_class     != size_class  ||
        hdr.dtype          != dtype) {
        fclose(f);
        return FB_JUDGE_ERR_NO_PROFILE;  /* stale or mismatched file */
    }

    read_ok = fread(profile_out, sizeof(*profile_out), 1, f);
    int err = ferror(f);
    fclose(f);

    if (read_ok != 1 || err)
        return FB_JUDGE_ERR_IO;

    return FB_JUDGE_OK;
}

/* =========================================================================
 * fb_judge_store_invalidate  (also implements public fb_judge_invalidate)
 * ========================================================================= */

void fb_judge_store_invalidate(
    const char *profile_dir,
    uint32_t    backend_id,
    uint32_t    device_id)
{
    if (!profile_dir)
        return;

    /* Ensure directory exists before writing the marker. */
    char subdir[FB_STORE_MAX_PATH];
    build_subdir_path(profile_dir, subdir, sizeof(subdir));
    (void)ensure_dir(subdir);  /* ignore failure if dir doesn't exist yet */

    char path[FB_STORE_MAX_PATH];
    build_inv_path(profile_dir, backend_id, device_id, path, sizeof(path));

    FILE *f = fopen(path, "wb");
    if (f) {
        /* Write a one-byte marker so the file has non-zero size. */
        fputc(1, f);
        fclose(f);
    }
}

/* =========================================================================
 * fb_judge_store_profiles_are_current
 * (also implements public fb_judge_profiles_are_current)
 * ========================================================================= */

bool fb_judge_store_profiles_are_current(
    const char *profile_dir,
    uint32_t    backend_id,
    uint32_t    device_id)
{
    if (!profile_dir)
        return false;

    /* Fast path: if an invalidation marker exists, they're not current. */
    if (inv_marker_exists(profile_dir, backend_id, device_id))
        return false;

    return true;
}

/* =========================================================================
 * NOTE: fb_judge_profiles_are_current() and fb_judge_invalidate() are
 * implemented in judge.c (the module singleton) which holds the profile_dir.
 * ========================================================================= */
