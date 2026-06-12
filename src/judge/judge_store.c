/**
 * @file judge_store.c
 * @brief Binary profile store: save, load, invalidate, and version check.
 *
 * File format (.fbjp):
 *   [4]  Magic 0x46424A50 ("FBJP")
 *   [4]  FB_JUDGE_MODULE_VERSION (uint32_t)
 *   [4]  FB_CORPUS_VERSION       (uint32_t)
 *   [32] op_name    (char[32], null-terminated — serialization key, cross-check on load)
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
#include <ctype.h>

/* Platform-specific headers for directory creation and scanning. */
#ifdef _WIN32
#  include <direct.h>   /* _mkdir */
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>  /* FindFirstFile, FindNextFile */
#  define FB_MKDIR(path)  _mkdir(path)
#else
#  include <sys/stat.h>
#  include <dirent.h>   /* opendir, readdir, closedir */
#  define FB_MKDIR(path)  mkdir((path), 0755)
#endif

#include "judge_types.h"     /* fb_op_judge_table, FB_JUDGE_MAX_OPERATIONS */

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
    char     op_name[32];    /* null-terminated operation name */
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
    const char *op_name, uint32_t backend_id, uint32_t device_id,
    uint8_t size_class, uint8_t dtype,
    char *out, size_t out_size)
{
    snprintf(out, out_size,
             "%s/%s/%s_be%u_dev%u_sc%u_dt%u.fbjp",
             profile_dir, FB_STORE_SUBDIR_NAME,
             op_name, backend_id, device_id, (unsigned)size_class, (unsigned)dtype);
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
                       profile->op_name, profile->backend_id, profile->device_id,
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
    strncpy(hdr.op_name, profile->op_name, sizeof(hdr.op_name) - 1);
    hdr.op_name[sizeof(hdr.op_name) - 1] = '\0';
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
    const char             *op_name,
    uint32_t                backend_id,
    uint32_t                device_id,
    uint8_t                 size_class,
    uint8_t                 dtype,
    fb_precision_profile_t *profile_out)
{
    if (!profile_dir || !op_name || !profile_out)
        return FB_JUDGE_ERR_IO;

    /* If invalidation marker is present, report missing. */
    if (inv_marker_exists(profile_dir, backend_id, device_id))
        return FB_JUDGE_ERR_NO_PROFILE;

    char path[FB_STORE_MAX_PATH];
    build_profile_path(profile_dir, op_name, backend_id, device_id,
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
        strncmp(hdr.op_name, op_name, sizeof(hdr.op_name)) != 0 ||
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

/* =========================================================================
 * fb_judge_store_list_profiled_ops
 * ========================================================================= */

/** Reverse-lookup op_id from op_name by scanning the metadata table.
 *
 * The table stores names as stringified enum tokens ("FB_OP_SAXPY") while
 * profile files record the lower-case short name ("saxpy").  This function
 * handles both forms:
 *  1. Direct match: name == op_name  (future-proof if table ever stores short names)
 *  2. Prefix-stripped, case-insensitive match: strip "FB_OP_" from the table
 *     name, then compare case-insensitively with op_name.
 */
static uint32_t meta_id_from_name(const char *op_name)
{
    if (!op_name) return UINT32_MAX;

    for (uint32_t id = 0; id < FB_JUDGE_MAX_OPERATIONS; id++) {
        const fb_op_judge_meta_t *m = &fb_op_judge_table[id];
        if (!m->name) continue;

        /* 1. Direct match (e.g. table already stores "saxpy"). */
        if (strcmp(m->name, op_name) == 0)
            return id;

        /* 2. Strip "FB_OP_" prefix (6 chars) and compare case-insensitively.
         *    Table stores "FB_OP_SAXPY"; file stores "saxpy". */
        if (strncmp(m->name, "FB_OP_", 6) == 0) {
            const char *short_name = m->name + 6;  /* e.g. "SAXPY" */
            size_t sn_len = strlen(short_name);
            size_t op_len = strlen(op_name);
            if (sn_len == op_len) {
                bool match = true;
                for (size_t i = 0; i < sn_len; i++) {
                    if (tolower((unsigned char)short_name[i]) !=
                        tolower((unsigned char)op_name[i])) {
                        match = false;
                        break;
                    }
                }
                if (match) return id;
            }
        }
    }
    return UINT32_MAX;  /* not found */
}

void fb_judge_meta_to_canonical_name(
    const char *meta_name,
    char       *buf,
    size_t      buf_size)
{
    if (!buf || buf_size == 0) return;
    buf[0] = '\0';
    if (!meta_name) return;
    const char *src = meta_name;
    if (strncmp(src, "FB_OP_", 6) == 0) src += 6;
    size_t i = 0;
    while (*src && i < buf_size - 1)
        buf[i++] = (char)tolower((unsigned char)*src++);
    buf[i] = '\0';
}

uint32_t fb_judge_store_list_profiled_ops(
    const char *profile_dir,
    uint32_t    backend_id,
    uint32_t    device_id,
    uint32_t   *op_ids_out,
    uint32_t    max_ops)
{
    if (!profile_dir) return 0;

    char subdir[FB_STORE_MAX_PATH];
    build_subdir_path(profile_dir, subdir, sizeof(subdir));

    /* Build the backend/device infix we filter on. */
    char filter[64];
    snprintf(filter, sizeof(filter), "_be%u_dev%u_", backend_id, device_id);

    uint32_t count = 0;

#ifdef _WIN32
    char pattern[FB_STORE_MAX_PATH];
    snprintf(pattern, sizeof(pattern), "%s\\*.fbjp", subdir);

    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;

    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        if (!strstr(fd.cFileName, filter)) continue;

        char full[FB_STORE_MAX_PATH];
        snprintf(full, sizeof(full), "%s\\%s", subdir, fd.cFileName);
        FILE *f = fopen(full, "rb");
        if (!f) continue;

        fb_store_header_t hdr;
        size_t nr = fread(&hdr, sizeof(hdr), 1, f);
        fclose(f);
        if (nr != 1 || hdr.magic != FB_STORE_MAGIC) continue;
        if (hdr.backend_id != backend_id || hdr.device_id != device_id) continue;

        hdr.op_name[sizeof(hdr.op_name) - 1] = '\0';
        uint32_t id = meta_id_from_name(hdr.op_name);
        if (id == UINT32_MAX) continue;

        /* Deduplicate: each op can have multiple profiles (size_class, dtype). */
        bool already = false;
        if (op_ids_out) {
            for (uint32_t i = 0; i < count && i < max_ops; i++) {
                if (op_ids_out[i] == id) { already = true; break; }
            }
        }
        if (!already) {
            if (op_ids_out && count < max_ops)
                op_ids_out[count] = id;
            count++;
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);

#else /* POSIX */
    DIR *d = opendir(subdir);
    if (!d) return 0;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *name = ent->d_name;
        size_t nlen = strlen(name);
        if (nlen < 5 || strcmp(name + nlen - 5, ".fbjp") != 0) continue;
        if (!strstr(name, filter)) continue;

        char full[FB_STORE_MAX_PATH];
        snprintf(full, sizeof(full), "%s/%s", subdir, name);
        FILE *f = fopen(full, "rb");
        if (!f) continue;

        fb_store_header_t hdr;
        size_t nr = fread(&hdr, sizeof(hdr), 1, f);
        fclose(f);
        if (nr != 1 || hdr.magic != FB_STORE_MAGIC) continue;
        if (hdr.backend_id != backend_id || hdr.device_id != device_id) continue;

        hdr.op_name[sizeof(hdr.op_name) - 1] = '\0';
        uint32_t id = meta_id_from_name(hdr.op_name);
        if (id == UINT32_MAX) continue;

        bool already = false;
        if (op_ids_out) {
            for (uint32_t i = 0; i < count && i < max_ops; i++) {
                if (op_ids_out[i] == id) { already = true; break; }
            }
        }
        if (!already) {
            if (op_ids_out && count < max_ops)
                op_ids_out[count] = id;
            count++;
        }
    }
    closedir(d);
#endif /* _WIN32 */

    return count;
}
