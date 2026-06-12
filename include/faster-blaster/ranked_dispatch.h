/**
 * @file ranked_dispatch.h
 * @brief Ranked dispatch tables — top-N per criterion per operation.
 *
 * Phase 3 dispatch architecture.
 *
 * Problem with a single-winner dispatch table
 * -------------------------------------------
 * fb_judge_build_dispatch_table() elects ONE backend per operation.  That
 * makes it impossible to:
 *   (a) switch the dispatch policy at runtime (FASTEST vs BALANCED vs
 *       MOST_ACCURATE) without rebuilding the table from disk,
 *   (b) satisfy an ad-hoc constraint ("fastest backend with ≥12 digits"),
 *   (c) fall back gracefully if the first-choice backend becomes unavailable.
 *
 * Ranked dispatch tables fix this by storing the top-N candidates per
 * (operation, criterion) so the router can walk the list at call time.
 *
 * Stored orderings
 * ----------------
 *   FB_RANK_FASTEST       — sorted ascending by p50_ns at FB_SIZE_MEDIUM;
 *                           effective_digits breaks ties (descending).
 *   FB_RANK_MOST_ACCURATE — sorted descending by effective_digits;
 *                           p50_ns breaks ties (ascending).
 *   FB_RANK_BALANCED      — sorted by weighted score (50/50 precision+speed,
 *                           FB_SELECT_BALANCED preset); ties broken by latency.
 *
 * Each slot carries the effective digit score at 50 % corpus pass rate
 * so that a runtime precision floor check requires only a comparison, not
 * a disk read.
 *
 * Typical usage
 * -------------
 *   // At startup — build once from judge profiles:
 *   uint32_t backends[] = { FB_BACKEND_ID_AOCL_BLIS, FB_BACKEND_ID_MKL,
 *                            FB_BACKEND_ID_OPENBLAS,  FB_BACKEND_ID_REFERENCE };
 *   fb_ranked_table_t *tbl = fb_ranked_table_build(
 *       "~/.cache/faster-blaster", 0,
 *       FB_DTYPE_FLOAT32, backends, 4, FB_BACKEND_ID_REFERENCE);
 *
 *   // During dispatch — O(1) policy lookup:
 *   uint32_t bid = fb_ranked_get_best(tbl, FB_OP_SAXPY, FB_RANK_FASTEST);
 *   // OR with a precision floor:
 *   uint32_t bid = fb_ranked_get_with_floor(tbl, FB_OP_DGEMM,
 *                                           FB_RANK_FASTEST, 12);
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_RANKED_DISPATCH_H
#define FASTER_BLASTER_RANKED_DISPATCH_H

#include <stdint.h>
#include <stdbool.h>
#include "judge.h"          /* FB_JUDGE_MAX_OPERATIONS, fb_dtype_t */
#include "judge_select.h"   /* fb_select_criteria_t, fb_select_objective_t */
#include "backend_ids.h"    /* FB_BACKEND_ID_* constants */

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Constants
 * ========================================================================= */

/** Number of top candidates stored per (operation, criterion). */
#define FB_RANKED_TOP_N          5u

/** Number of built-in criteria axes stored in a ranked table. */
#define FB_RANKED_N_CRITERIA     3u

/** Maximum number of backends a ranked table can describe. */
#define FB_RANKED_MAX_BACKENDS   16u

/** On-disk format version — increment when the binary layout changes. */
#define FB_RANKED_TABLE_VERSION  1u

/* =========================================================================
 * Criterion axis
 * ========================================================================= */

/**
 * Which ordering axis to consult when querying the ranked table.
 *
 * Values are stable integers used as array indices — do not renumber.
 */
typedef enum {
    /** Ascending p50 latency (speed first, digits tie-break). */
    FB_RANK_FASTEST       = 0,
    /** Descending effective digits (precision first, latency tie-break). */
    FB_RANK_MOST_ACCURATE = 1,
    /** 50/50 weighted precision+speed score. */
    FB_RANK_BALANCED      = 2,
} fb_rank_criterion_t;

/* =========================================================================
 * Entry type
 * ========================================================================= */

/**
 * One candidate backend for a given (operation, criterion, rank) triple.
 *
 * Populated from profiled data in .fbjp files.  All values are snapshots
 * taken at table-build time; they do not update after the table is built.
 */
typedef struct {
    /** Backend numeric ID (FB_BACKEND_ID_*). */
    uint32_t backend_id;

    /** p50 wall-clock time in nanoseconds at FB_SIZE_MEDIUM.
     *  UINT32_MAX means no timing data was available. */
    uint32_t latency_p50_ns;

    /** Effective digit score at 50 % corpus pass rate across all primary
     *  metrics.  0 indicates no precision data was profiled. */
    uint8_t  effective_digits;

    /** 1 = this slot holds valid profiled data; 0 = slot unused. */
    uint8_t  is_valid;

    uint8_t  _pad[2];            /* align to 12 bytes */
} fb_ranked_entry_t;             /* sizeof == 12 */

/* =========================================================================
 * Opaque table type
 * ========================================================================= */

/**
 * Ranked dispatch table.  Always heap-allocated; never embed by value.
 * Access via the fb_ranked_* API below.
 */
typedef struct fb_ranked_table fb_ranked_table_t;

/* =========================================================================
 * Status codes
 * ========================================================================= */

typedef enum {
    FB_RANKED_OK                    =  0,
    /** No profiles found for any backend on any operation.
     *  Table is valid but empty; all lookups return fallback. */
    FB_RANKED_WARN_EMPTY            =  1,
    /** I/O error reading or writing the table cache file. */
    FB_RANKED_ERR_IO                = -1,
    /** A required argument was NULL or an index was out of range. */
    FB_RANKED_ERR_INVALID           = -2,
    /** table_load() found an incompatible format version. */
    FB_RANKED_ERR_VERSION_MISMATCH  = -3,
    /** malloc() returned NULL. */
    FB_RANKED_ERR_OOM               = -4,
} fb_ranked_status_t;

/* =========================================================================
 * Build
 * ========================================================================= */

/**
 * Build a ranked dispatch table from on-disk judge profiles.
 *
 * For every (op, criterion) pair, loads profiles for each backend in
 * @p backend_ids, scores them according to the criterion's objective, and
 * stores the top-FB_RANKED_TOP_N results in ascending rank order (rank 0 =
 * best).
 *
 * Building the table is a startup-time operation; it is NOT called during
 * normal dispatch.  Profile loading is from disk, so the first call may be
 * slow on a cold filesystem cache.  Subsequent calls reading the same files
 * will be fast due to OS page caching.
 *
 * @param profile_dir         Directory containing .fbjp files, passed to
 *                            fb_judge_store_load().
 * @param device_id           Device ordinal (0 for primary, matches profiles).
 * @param primary_dtype       Data type to drive precision + latency scoring.
 *                            Use FB_DTYPE_FLOAT32 or FB_DTYPE_FLOAT64.
 * @param backend_ids         Candidate backend IDs to consider.
 * @param n_backends          Length of backend_ids[]; capped at
 *                            FB_RANKED_MAX_BACKENDS.
 * @param fallback_backend_id Returned by lookup functions when no profiled
 *                            candidate exists.
 * @return Heap-allocated table (free with fb_ranked_table_free()), or NULL
 *         on allocation failure.  An empty table (no profiles found) is
 *         returned with status FB_RANKED_WARN_EMPTY recorded internally.
 */
fb_ranked_table_t *fb_ranked_table_build(
    const char    *profile_dir,
    uint32_t       device_id,
    uint8_t        primary_dtype,
    const uint32_t backend_ids[],
    uint32_t       n_backends,
    uint32_t       fallback_backend_id
);

/**
 * Free a ranked dispatch table allocated by fb_ranked_table_build() or
 * fb_ranked_table_load().
 */
void fb_ranked_table_free(fb_ranked_table_t *table);

/* =========================================================================
 * Serialise / deserialise
 * ========================================================================= */

/**
 * Save the ranked table to a binary file for fast restart-time loading.
 *
 * The saved file starts with a 16-byte header that includes the format
 * version so that fb_ranked_table_load() can detect stale files.
 *
 * @return FB_RANKED_OK, or FB_RANKED_ERR_IO / FB_RANKED_ERR_INVALID.
 */
fb_ranked_status_t fb_ranked_table_save(const fb_ranked_table_t *table,
                                        const char *path);

/**
 * Load a ranked table from a file written by fb_ranked_table_save().
 *
 * Returns NULL on I/O error or version mismatch (caller should rebuild).
 */
fb_ranked_table_t *fb_ranked_table_load(const char *path);

/* =========================================================================
 * Lookup — O(1) or O(FB_RANKED_TOP_N) per call
 * ========================================================================= */

/**
 * Get the best backend for an operation under a given criterion.
 *
 * Returns the rank-0 entry that is currently marked available (not disabled
 * via fb_ranked_mark_unavailable()).  If all ranked backends are unavailable,
 * or no profile exists, returns the table's fallback_backend_id.
 *
 * @param table      Built or loaded ranked table.
 * @param op_id      Operation ID (FB_OP_* from judge_op_ids.h).
 * @param criterion  Which ranking axis to consult.
 * @return Backend ID to use.
 */
uint32_t fb_ranked_get_best(
    const fb_ranked_table_t *table,
    uint32_t                 op_id,
    fb_rank_criterion_t      criterion
);

/**
 * Get the fastest available backend whose effective_digits ≥ @p min_digits.
 *
 * Walks the criterion's ranked list from best to worst, skipping entries
 * whose digit score is below the floor or whose backend is unavailable.
 * Falls back to the table's fallback_backend_id when no entry qualifies.
 *
 * Typical use: "fastest backend that still gives me at least 12 digits."
 *
 * @param table       Built or loaded ranked table.
 * @param op_id       Operation ID.
 * @param criterion   Which ranked list to walk (usually FB_RANK_FASTEST).
 * @param min_digits  Minimum acceptable digit score (0 = no constraint).
 * @return Backend ID satisfying the floor, or fallback.
 */
uint32_t fb_ranked_get_with_floor(
    const fb_ranked_table_t *table,
    uint32_t                 op_id,
    fb_rank_criterion_t      criterion,
    uint8_t                  min_digits
);

/**
 * Select the best available backend for @p op_id according to a full
 * fb_select_criteria_t specification.
 *
 * Maps the criteria's objective to the most appropriate ranked-list axis
 * and applies any hard constraint (min_digits or max_latency_ns):
 *
 *   MAXIMIZE_PRECISION         → fb_ranked_get_best(MOST_ACCURATE)
 *   MAXIMIZE_SPEED             → fb_ranked_get_best(FASTEST)
 *   PRECISION_FLOOR_THEN_SPEED → fb_ranked_get_with_floor(FASTEST, min_digits)
 *   SPEED_FLOOR_THEN_PRECISION → walk MOST_ACCURATE list, skip over-latency entries
 *   WEIGHTED / default         → fb_ranked_get_best(BALANCED)
 *
 * When allow_degraded is false and SPEED_FLOOR_THEN_PRECISION finds no
 * backend within the latency ceiling, the table's fallback is returned.
 * When allow_degraded is true, the precision-best available entry is used.
 *
 * @param table     Built or loaded ranked table (may not be NULL).
 * @param op_id     Operation ID (FB_OP_* from judge_op_ids.h).
 * @param criteria  Selection policy — must not be NULL.
 * @return Backend ID to use, or fallback_backend_id when nothing qualifies.
 */
uint32_t fb_ranked_select_with_criteria(
    const fb_ranked_table_t    *table,
    uint32_t                    op_id,
    const fb_select_criteria_t *criteria
);

/**
 * Get the k-th ranked entry (0 = best) for an (op, criterion) pair.
 *
 * @param table     Ranked table.
 * @param op_id     Operation ID.
 * @param criterion Criterion axis.
 * @param rank      0-based rank (must be < fb_ranked_entry_count()).
 * @return Pointer to the slot, or NULL if rank is out of range.
 *         The pointer is valid for the lifetime of @p table.
 */
const fb_ranked_entry_t *fb_ranked_get_entry(
    const fb_ranked_table_t *table,
    uint32_t                 op_id,
    fb_rank_criterion_t      criterion,
    uint32_t                 rank
);

/**
 * Number of valid ranked entries for (op, criterion).
 * Returns 0..FB_RANKED_TOP_N.
 */
uint32_t fb_ranked_entry_count(
    const fb_ranked_table_t *table,
    uint32_t                 op_id,
    fb_rank_criterion_t      criterion
);

/* =========================================================================
 * Availability management
 * ========================================================================= */

/**
 * Mark a backend as unavailable in this table.
 *
 * After this call, fb_ranked_get_best() and fb_ranked_get_with_floor() will
 * skip all entries with this backend_id, as if they don't exist.
 *
 * Intended use: call when a backend fails to initialise at runtime so that
 * the next-best candidate is used automatically, without rebuilding the table.
 *
 * @param table      Table to update (mutating operation).
 * @param backend_id The backend to disable.
 */
void fb_ranked_mark_unavailable(fb_ranked_table_t *table,
                                uint32_t           backend_id);

/**
 * Re-enable a previously disabled backend.
 * Counterpart to fb_ranked_mark_unavailable().
 */
void fb_ranked_mark_available(fb_ranked_table_t *table,
                              uint32_t           backend_id);

/**
 * Returns true if @p backend_id is currently marked unavailable.
 */
bool fb_ranked_is_available(const fb_ranked_table_t *table,
                            uint32_t                  backend_id);

/* =========================================================================
 * Statistics
 * ========================================================================= */

/**
 * Summary statistics for a ranked table.
 */
typedef struct {
    /** Number of operations where at least one backend has a profile. */
    uint32_t ops_with_profiles;
    /** Number of operations with all FB_RANKED_N_CRITERIA fully populated
     *  (each criterion has FB_RANKED_TOP_N entries). */
    uint32_t ops_fully_ranked;
    /** Sum of valid entry counts across all ops and criteria. */
    uint32_t total_entries;
    /** Number of distinct backend_ids that appear at least once (rank 0). */
    uint32_t distinct_winners;
} fb_ranked_stats_t;

/**
 * Compute summary statistics for a ranked table.
 *
 * @param table     Table to inspect.
 * @param stats_out Caller-allocated result struct.
 */
void fb_ranked_table_get_stats(const fb_ranked_table_t *table,
                               fb_ranked_stats_t       *stats_out);

/* =========================================================================
 * Operation-level dispatch helpers
 * =========================================================================
 *
 * These functions enable application code and the dispatch layer to select
 * the best backend for a specific operation ID without holding an explicit
 * ranked-table reference.  A single active table is stored globally and
 * queried by name.
 *
 * Typical startup sequence:
 *
 *   fb_ranked_table_t *tbl = fb_ranked_table_build(...);
 *   fb_op_dispatch_set_table(tbl);          // register globally
 *
 *   // In a hot dispatch path:
 *   uint32_t bid = fb_select_backend_for_op(FB_OP_SGEMM, FB_RANK_FASTEST);
 * ========================================================================= */

/**
 * Set (or replace) the globally active ranked table used by
 * fb_select_backend_for_op().  Does not take ownership; the caller must
 * keep the table alive.
 *
 * @param table  Ranked table to register; NULL clears the global.
 */
void fb_op_dispatch_set_table(fb_ranked_table_t *table);

/**
 * Return the currently active ranked table, or NULL if none has been set.
 */
const fb_ranked_table_t *fb_op_dispatch_get_table(void);

/**
 * Select the best available backend for @p op_id using the globally active
 * ranked table.  Falls back to @c FB_BACKEND_ID_REFERENCE when no table is
 * set or the op is unranked.
 *
 * This is the primary entry point for operation-level dispatch across the
 * full 2266-operation superset.
 *
 * @param op_id     FB_OP_* constant from judge_op_ids.h (0 – 2265).
 * @param criterion Ranking axis: FB_RANK_FASTEST / MOST_ACCURATE / BALANCED.
 * @return          Backend ID to use.
 */
uint32_t fb_select_backend_for_op(uint32_t            op_id,
                                  fb_rank_criterion_t criterion);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_RANKED_DISPATCH_H */
