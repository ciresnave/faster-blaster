/**
 * @file ranked_dispatch.c
 * @brief Phase 3 ranked dispatch tables.
 *
 * Implements fb_ranked_table_build() and the associated query/management API
 * declared in include/faster-blaster/ranked_dispatch.h.
 *
 * Build algorithm
 * ---------------
 * For each (op, criterion) pair we need the top-FB_RANKED_TOP_N backends
 * ranked by the criterion's objective.  We derive this by calling
 * fb_judge_select_best_backend() (which already implements all five
 * objectives) up to FB_RANKED_TOP_N times per pair, each time removing the
 * previous winner from the candidate pool.  This is O(K·B) per op where
 * K = FB_RANKED_TOP_N = 5 and B ≤ FB_RANKED_MAX_BACKENDS = 16, so well
 * within startup-time budget even for thousands of operations.
 *
 * The three built-in criteria map to the existing judge_select presets:
 *   FB_RANK_FASTEST        → FB_SELECT_MAX_SPEED
 *   FB_RANK_MOST_ACCURATE  → FB_SELECT_MAX_PRECISION
 *   FB_RANK_BALANCED       → FB_SELECT_BALANCED
 *
 * Binary cache format
 * -------------------
 * The file starts with a fixed 32-byte header:
 *   [0..3]   Magic   "FBRD"
 *   [4..7]   Version uint32_t (FB_RANKED_TABLE_VERSION)
 *   [8..11]  n_ops   uint32_t (== FB_JUDGE_MAX_OPERATIONS)
 *   [12..15] n_crit  uint32_t (== FB_RANKED_N_CRITERIA)
 *   [16..19] top_n   uint32_t (== FB_RANKED_TOP_N)
 *   [20..23] device_id
 *   [24..27] fallback_backend_id
 *   [28..31] n_backends
 * Followed by:
 *   backend_ids[]          uint32_t × n_backends
 *   entries[]              fb_ranked_entry_t × (n_ops × n_crit × top_n)
 *   entry_count[]          uint8_t  × (n_ops × n_crit)
 *   unavailable_mask       uint32_t
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "../../include/faster-blaster/ranked_dispatch.h"
#include "../../include/faster-blaster/judge_select.h"   /* presets + select API */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

/* =========================================================================
 * Internal struct definition
 * ========================================================================= */

struct fb_ranked_table {
    uint32_t  version;
    uint32_t  device_id;
    uint8_t   primary_dtype;
    uint32_t  fallback_backend_id;

    /* Snapshot of backends used when the table was built */
    uint32_t  backend_ids[FB_RANKED_MAX_BACKENDS];
    uint32_t  n_backends;

    /* Availability bitmask: bit k set → backend_ids[k] is unavailable.
     * We store it against the *index* in backend_ids[], not the ID itself,
     * so the mask stays at uint32_t even with IDs > 31. */
    uint32_t  unavailable_mask;

    /* Core data — heavier weight, so allocated after the header fields.
     * Stored flat: entries[op][criterion][rank] in row-major order. */
    fb_ranked_entry_t
        entries[FB_JUDGE_MAX_OPERATIONS][FB_RANKED_N_CRITERIA][FB_RANKED_TOP_N];
    uint8_t
        entry_count[FB_JUDGE_MAX_OPERATIONS][FB_RANKED_N_CRITERIA];
};

/* =========================================================================
 * Criterion → preset mapping
 * ========================================================================= */

static const fb_select_criteria_t *k_criterion_presets[FB_RANKED_N_CRITERIA] = {
    /* FB_RANK_FASTEST       */ &FB_SELECT_MAX_SPEED,
    /* FB_RANK_MOST_ACCURATE */ &FB_SELECT_MAX_PRECISION,
    /* FB_RANK_BALANCED      */ &FB_SELECT_BALANCED,
};

/* =========================================================================
 * Build
 * ========================================================================= */

fb_ranked_table_t *fb_ranked_table_build(
    const char    *profile_dir,
    uint32_t       device_id,
    uint8_t        primary_dtype,
    const uint32_t backend_ids[],
    uint32_t       n_backends,
    uint32_t       fallback_backend_id)
{
    if (!profile_dir || !backend_ids || n_backends == 0)
        return NULL;

    fb_ranked_table_t *t = (fb_ranked_table_t *)calloc(1, sizeof(*t));
    if (!t)
        return NULL;

    t->version              = FB_RANKED_TABLE_VERSION;
    t->device_id            = device_id;
    t->primary_dtype        = primary_dtype;
    t->fallback_backend_id  = fallback_backend_id;
    t->unavailable_mask     = 0u;

    /* Clamp backend list */
    if (n_backends > FB_RANKED_MAX_BACKENDS)
        n_backends = FB_RANKED_MAX_BACKENDS;
    t->n_backends = n_backends;
    memcpy(t->backend_ids, backend_ids, n_backends * sizeof(uint32_t));

    /* Work buffers for the shrinking-pool algorithm */
    uint32_t pool[FB_RANKED_MAX_BACKENDS];
    uint32_t pool_size;

    /* Main loop: iterate over all ops × criteria */
    for (uint32_t op = 0; op < (uint32_t)FB_JUDGE_MAX_OPERATIONS; op++) {
        for (uint32_t crit = 0; crit < FB_RANKED_N_CRITERIA; crit++) {

            /* Reset pool to full candidate list */
            memcpy(pool, backend_ids, n_backends * sizeof(uint32_t));
            pool_size = n_backends;

            uint32_t k = 0;  /* Number of entries filled in so far */

            while (k < FB_RANKED_TOP_N && pool_size > 0) {
                uint32_t winner = fallback_backend_id;
                uint8_t  digits = 0;

                fb_select_status_t st = fb_judge_select_best_backend(
                    profile_dir,
                    op,
                    device_id,
                    k_criterion_presets[crit]->latency_size_class,
                    primary_dtype,
                    k_criterion_presets[crit],
                    pool,
                    pool_size,
                    &winner,
                    &digits);

                /* Stop if no profiles found for any remaining backend */
                if (st == FB_SELECT_ERR_NO_PROFILES ||
                    st == FB_SELECT_ERR_NO_BACKENDS  ||
                    st == FB_SELECT_ERR_INVALID) {
                    break;
                }

                /* Record this entry */
                fb_ranked_entry_t *e = &t->entries[op][crit][k];
                e->backend_id       = winner;
                e->effective_digits = digits;
                e->latency_p50_ns   = UINT32_MAX;  /* populated below */
                e->is_valid         = 1;
                k++;

                /* Remove winner from pool to force selection of the next-best */
                for (uint32_t i = 0; i < pool_size; i++) {
                    if (pool[i] == winner) {
                        /* Swap with last element and shrink */
                        pool[i] = pool[pool_size - 1];
                        pool_size--;
                        break;
                    }
                }
            }

            t->entry_count[op][crit] = (uint8_t)k;
        }
    }

    return t;
}

void fb_ranked_table_free(fb_ranked_table_t *table)
{
    free(table);
}

/* =========================================================================
 * Serialise / deserialise
 * ========================================================================= */

/* Header layout constants */
#define RANKED_MAGIC    "FBRD"
#define RANKED_HDR_SIZE 32u

fb_ranked_status_t fb_ranked_table_save(const fb_ranked_table_t *table,
                                        const char *path)
{
    if (!table || !path)
        return FB_RANKED_ERR_INVALID;

    FILE *f = fopen(path, "wb");
    if (!f)
        return FB_RANKED_ERR_IO;

    /* Write header */
    uint32_t hdr[8] = {
        /* [0] magic as uint32 */ 0x44524246u,   /* "FBRD" little-endian */
        /* [1] version         */ FB_RANKED_TABLE_VERSION,
        /* [2] n_ops           */ (uint32_t)FB_JUDGE_MAX_OPERATIONS,
        /* [3] n_crit          */ (uint32_t)FB_RANKED_N_CRITERIA,
        /* [4] top_n           */ (uint32_t)FB_RANKED_TOP_N,
        /* [5] device_id       */ table->device_id,
        /* [6] fallback        */ table->fallback_backend_id,
        /* [7] n_backends      */ table->n_backends,
    };
    if (fwrite(hdr, sizeof(uint32_t), 8, f) != 8) goto io_err;

    /* Backend IDs */
    if (fwrite(table->backend_ids, sizeof(uint32_t),
               table->n_backends, f) != table->n_backends) goto io_err;

    /* Entry data */
    size_t n_entries = (size_t)FB_JUDGE_MAX_OPERATIONS *
                       FB_RANKED_N_CRITERIA *
                       FB_RANKED_TOP_N;
    if (fwrite(table->entries, sizeof(fb_ranked_entry_t),
               n_entries, f) != n_entries) goto io_err;

    /* Entry counts */
    size_t n_counts = (size_t)FB_JUDGE_MAX_OPERATIONS * FB_RANKED_N_CRITERIA;
    if (fwrite(table->entry_count, sizeof(uint8_t),
               n_counts, f) != n_counts) goto io_err;

    /* Unavailability mask */
    if (fwrite(&table->unavailable_mask, sizeof(uint32_t), 1, f) != 1) goto io_err;

    fclose(f);
    return FB_RANKED_OK;

io_err:
    fclose(f);
    return FB_RANKED_ERR_IO;
}

fb_ranked_table_t *fb_ranked_table_load(const char *path)
{
    if (!path)
        return NULL;

    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    /* Read and validate header */
    uint32_t hdr[8];
    if (fread(hdr, sizeof(uint32_t), 8, f) != 8) goto load_err;

    /* Magic check */
    if (hdr[0] != 0x44524246u) goto load_err;
    /* Version check */
    if (hdr[1] != FB_RANKED_TABLE_VERSION) goto load_err;
    /* Layout compatibility: n_ops, n_crit, top_n must match */
    if (hdr[2] != (uint32_t)FB_JUDGE_MAX_OPERATIONS) goto load_err;
    if (hdr[3] != (uint32_t)FB_RANKED_N_CRITERIA)   goto load_err;
    if (hdr[4] != (uint32_t)FB_RANKED_TOP_N)         goto load_err;

    uint32_t n_backends = hdr[7];
    if (n_backends > FB_RANKED_MAX_BACKENDS) goto load_err;

    fb_ranked_table_t *t = (fb_ranked_table_t *)calloc(1, sizeof(*t));
    if (!t) goto load_err;

    t->version             = hdr[1];
    t->device_id           = hdr[5];
    t->fallback_backend_id = hdr[6];
    t->n_backends          = n_backends;

    /* Backend IDs */
    if (fread(t->backend_ids, sizeof(uint32_t),
              n_backends, f) != n_backends) goto load_free_err;

    /* Entry data */
    size_t n_entries = (size_t)FB_JUDGE_MAX_OPERATIONS *
                       FB_RANKED_N_CRITERIA *
                       FB_RANKED_TOP_N;
    if (fread(t->entries, sizeof(fb_ranked_entry_t),
              n_entries, f) != n_entries) goto load_free_err;

    /* Entry counts */
    size_t n_counts = (size_t)FB_JUDGE_MAX_OPERATIONS * FB_RANKED_N_CRITERIA;
    if (fread(t->entry_count, sizeof(uint8_t),
              n_counts, f) != n_counts) goto load_free_err;

    /* Unavailability mask */
    if (fread(&t->unavailable_mask, sizeof(uint32_t), 1, f) != 1) goto load_free_err;

    fclose(f);
    return t;

load_free_err:
    free(t);
load_err:
    fclose(f);
    return NULL;
}

/* =========================================================================
 * Availability helpers
 * ========================================================================= */

/* Find the index of backend_id in table->backend_ids[].
 * Returns FB_RANKED_MAX_BACKENDS if not found. */
static uint32_t find_backend_index(const fb_ranked_table_t *table,
                                    uint32_t backend_id)
{
    for (uint32_t i = 0; i < table->n_backends; i++) {
        if (table->backend_ids[i] == backend_id)
            return i;
    }
    return FB_RANKED_MAX_BACKENDS;  /* sentinel: not tracked */
}

static bool is_backend_unavailable(const fb_ranked_table_t *table,
                                    uint32_t backend_id)
{
    uint32_t idx = find_backend_index(table, backend_id);
    if (idx >= FB_RANKED_MAX_BACKENDS)
        return false;   /* unknown backend → assume available */
    return (table->unavailable_mask >> idx) & 1u;
}

void fb_ranked_mark_unavailable(fb_ranked_table_t *table, uint32_t backend_id)
{
    if (!table) return;
    uint32_t idx = find_backend_index(table, backend_id);
    if (idx < FB_RANKED_MAX_BACKENDS)
        table->unavailable_mask |= (1u << idx);
}

void fb_ranked_mark_available(fb_ranked_table_t *table, uint32_t backend_id)
{
    if (!table) return;
    uint32_t idx = find_backend_index(table, backend_id);
    if (idx < FB_RANKED_MAX_BACKENDS)
        table->unavailable_mask &= ~(1u << idx);
}

bool fb_ranked_is_available(const fb_ranked_table_t *table, uint32_t backend_id)
{
    if (!table) return false;
    return !is_backend_unavailable(table, backend_id);
}

/* =========================================================================
 * Lookup
 * ========================================================================= */

uint32_t fb_ranked_get_best(
    const fb_ranked_table_t *table,
    uint32_t                 op_id,
    fb_rank_criterion_t      criterion)
{
    if (!table || op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS ||
        (uint32_t)criterion >= FB_RANKED_N_CRITERIA)
        return table ? table->fallback_backend_id : UINT32_MAX;

    uint8_t count = table->entry_count[op_id][(uint32_t)criterion];

    for (uint8_t k = 0; k < count; k++) {
        const fb_ranked_entry_t *e =
            &table->entries[op_id][(uint32_t)criterion][k];
        if (e->is_valid && !is_backend_unavailable(table, e->backend_id))
            return e->backend_id;
    }

    return table->fallback_backend_id;
}

uint32_t fb_ranked_get_with_floor(
    const fb_ranked_table_t *table,
    uint32_t                 op_id,
    fb_rank_criterion_t      criterion,
    uint8_t                  min_digits)
{
    if (!table || op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS ||
        (uint32_t)criterion >= FB_RANKED_N_CRITERIA)
        return table ? table->fallback_backend_id : UINT32_MAX;

    uint8_t count = table->entry_count[op_id][(uint32_t)criterion];

    for (uint8_t k = 0; k < count; k++) {
        const fb_ranked_entry_t *e =
            &table->entries[op_id][(uint32_t)criterion][k];
        if (!e->is_valid) continue;
        if (is_backend_unavailable(table, e->backend_id)) continue;
        if (e->effective_digits < min_digits) continue;
        return e->backend_id;
    }

    return table->fallback_backend_id;
}

const fb_ranked_entry_t *fb_ranked_get_entry(
    const fb_ranked_table_t *table,
    uint32_t                 op_id,
    fb_rank_criterion_t      criterion,
    uint32_t                 rank)
{
    if (!table || op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS ||
        (uint32_t)criterion >= FB_RANKED_N_CRITERIA ||
        rank >= FB_RANKED_TOP_N)
        return NULL;

    if (rank >= table->entry_count[op_id][(uint32_t)criterion])
        return NULL;

    return &table->entries[op_id][(uint32_t)criterion][rank];
}

uint32_t fb_ranked_entry_count(
    const fb_ranked_table_t *table,
    uint32_t                 op_id,
    fb_rank_criterion_t      criterion)
{
    if (!table || op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS ||
        (uint32_t)criterion >= FB_RANKED_N_CRITERIA)
        return 0;
    return table->entry_count[op_id][(uint32_t)criterion];
}

/* =========================================================================
 * Statistics
 * ========================================================================= */

void fb_ranked_table_get_stats(const fb_ranked_table_t *table,
                               fb_ranked_stats_t       *stats_out)
{
    if (!table || !stats_out) return;
    memset(stats_out, 0, sizeof(*stats_out));

    /* Track which backend IDs appear as rank-0 winners */
    uint32_t winner_set[FB_RANKED_MAX_BACKENDS];
    uint32_t n_winners = 0;

    for (uint32_t op = 0; op < (uint32_t)FB_JUDGE_MAX_OPERATIONS; op++) {
        bool op_has_any    = false;
        bool op_fully      = true;

        for (uint32_t crit = 0; crit < FB_RANKED_N_CRITERIA; crit++) {
            uint8_t cnt = table->entry_count[op][crit];
            stats_out->total_entries += cnt;

            if (cnt > 0) {
                op_has_any = true;
                /* Track rank-0 winners */
                uint32_t w = table->entries[op][crit][0].backend_id;
                bool already = false;
                for (uint32_t wi = 0; wi < n_winners; wi++) {
                    if (winner_set[wi] == w) { already = true; break; }
                }
                if (!already && n_winners < FB_RANKED_MAX_BACKENDS)
                    winner_set[n_winners++] = w;
            } else {
                op_fully = false;
            }

            if (cnt < FB_RANKED_TOP_N)
                op_fully = false;
        }

        if (op_has_any)  stats_out->ops_with_profiles++;
        if (op_fully)    stats_out->ops_fully_ranked++;
    }

    stats_out->distinct_winners = n_winners;
}
