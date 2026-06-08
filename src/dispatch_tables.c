/**
 * @file dispatch_tables.c
 * @brief Ranked dispatch table generation and constraint-based operation selection
 *
 * This module implements the critical dispatch table architecture that enables
 * intelligent operation selection based on multiple optimization criteria (speed,
 * accuracy, precision, power efficiency, balanced).
 *
 * KEY ARCHITECTURAL INSIGHT:
 * Don't generate single vtables from benchmarks - instead use benchmarks to RANK
 * operations and store multiple candidates per criterion. This allows runtime
 * selection based on constraints (min accuracy, max time, preferred criterion).
 */

#include "../include/dispatch_tables.h"
#include "../include/faster-blaster/ranked_dispatch.h"
/* ranked_dispatch.h transitively provides judge_select.h and backend_ids.h */
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

/**
 * Helper: Compare two ranked operations by time (ascending for FASTEST)
 */
[[maybe_unused]] static int compare_by_time(const void *a, const void *b) {
    const fb_ranked_operation_t *op_a = (const fb_ranked_operation_t *)a;
    const fb_ranked_operation_t *op_b = (const fb_ranked_operation_t *)b;
    
    if (op_a->metrics.time_mean_ns < op_b->metrics.time_mean_ns) return -1;
    if (op_a->metrics.time_mean_ns > op_b->metrics.time_mean_ns) return 1;
    return 0;
}

/**
 * Helper: Compare by accuracy (descending - higher is better)
 */
[[maybe_unused]] static int compare_by_accuracy(const void *a, const void *b) {
    const fb_ranked_operation_t *op_a = (const fb_ranked_operation_t *)a;
    const fb_ranked_operation_t *op_b = (const fb_ranked_operation_t *)b;
    
    if (op_a->metrics.accuracy_mean > op_b->metrics.accuracy_mean) return -1;
    if (op_a->metrics.accuracy_mean < op_b->metrics.accuracy_mean) return 1;
    return 0;
}

/**
 * Helper: Compare by precision (descending - higher is better)
 */
[[maybe_unused]] static int compare_by_precision(const void *a, const void *b) {
    const fb_ranked_operation_t *op_a = (const fb_ranked_operation_t *)a;
    const fb_ranked_operation_t *op_b = (const fb_ranked_operation_t *)b;
    
    if (op_a->metrics.precision_mean > op_b->metrics.precision_mean) return -1;
    if (op_a->metrics.precision_mean < op_b->metrics.precision_mean) return 1;
    return 0;
}

/**
 * Helper: Compare by power efficiency (ascending - lower power is better)
 * Estimate: Power ~ FLOPs / time, so power efficiency ~ time / (relative perf)
 * For now: approximate as inverse of time_mean_ns (higher is better)
 */
[[maybe_unused]] static int compare_by_power(const void *a, const void *b) {
    const fb_ranked_operation_t *op_a = (const fb_ranked_operation_t *)a;
    const fb_ranked_operation_t *op_b = (const fb_ranked_operation_t *)b;
    
    // Lower time = higher power efficiency (more work in same power budget)
    // So same ordering as FASTEST
    if (op_a->metrics.time_mean_ns < op_b->metrics.time_mean_ns) return -1;
    if (op_a->metrics.time_mean_ns > op_b->metrics.time_mean_ns) return 1;
    return 0;
}

/**
 * Helper: Balanced score = weighted combination of criteria
 * Weight: 40% speed, 30% accuracy, 20% precision, 10% power
 */
[[maybe_unused]] static double compute_balanced_score(const fb_benchmark_stats_t *metrics) {
    double speed_score = 1.0 / (1.0 + (metrics->time_mean_ns / 1e9));  // Normalize to [0,1]
    double accuracy_score = metrics->accuracy_mean;  // Already [0,1]
    double precision_score = metrics->precision_mean;  // Already [0,1]
    double power_score = 1.0 / (1.0 + (metrics->time_mean_ns / 1e9));  // Same as speed
    
    return 0.4 * speed_score + 0.3 * accuracy_score + 0.2 * precision_score + 0.1 * power_score;
}

/**
 * Helper: Compare by balanced score (descending)
 */
[[maybe_unused]] static int compare_by_balanced(const void *a, const void *b) {
    const fb_ranked_operation_t *op_a = (const fb_ranked_operation_t *)a;
    const fb_ranked_operation_t *op_b = (const fb_ranked_operation_t *)b;
    
    double score_a = compute_balanced_score(&op_a->metrics);
    double score_b = compute_balanced_score(&op_b->metrics);
    
    if (score_a > score_b) return -1;
    if (score_a < score_b) return 1;
    return 0;
}

/**
 * Generate all five dispatch tables from cached benchmark results
 *
 * Algorithm:
 * 1. Collect all benchmark results for each operation across backends/devices/sizes/shapes
 * 2. For each criterion (FASTEST, MOST_ACCURATE, MOST_PRECISE, POWER_EFFICIENT, BALANCED):
 *    a. Sort collected results by criterion
 *    b. Extract top 5 operations
 *    c. Store in ranked table with full metrics
 * 3. Save/load dispatch tables as binary file
 *
 * PLACEHOLDER IMPLEMENTATION: Full implementation requires integration with benchmark cache.
 * For now, this creates empty dispatch tables that can be populated by later phases.
 *
 * @param cache Benchmark cache with all results (from fb_benchmark_cache_load)
 * @param tables Output: generated dispatch tables
 * @return 0 on success, negative on error
 */
int fb_generate_dispatch_tables(void *cache, 
                                 fb_dispatch_tables_t *tables) {
    if (!cache || !tables) {
        return -1;  // Invalid input
    }
    
    // Initialize all tables
    memset(tables, 0, sizeof(fb_dispatch_tables_t));
    
    // PLACEHOLDER: Full implementation TODO
    // In Phase 3.1, this will:
    // 1. Iterate through benchmark cache
    // 2. For each operation (0-1247):
    //    - Collect all implementations from cache
    //    - Sort by each criterion (FASTEST, MOST_ACCURATE, etc.)
    //    - Extract top 5 for each criterion
    //    - Store with full metrics (time, accuracy, precision)
    //
    // For now: just return initialized (empty) tables for testing
    
    return 0;
}

/**
 * Select the best operation for a given operation ID
 *
 * PLACEHOLDER: This is simplified for Phase 3.0
 * Full implementation will support constraint-based selection
 *
 * @param tables Ranked dispatch tables
 * @param op_id Operation ID (0-1247)
 * @param size_class Size classification (0-4)
 * @param shape_class Shape classification (0-4)
 * @return Ranked operation pointer, or NULL if not available
 */
const fb_ranked_operation_t* fb_select_operation(
    const fb_dispatch_tables_t *tables,
    uint32_t op_id,
    const fb_dispatch_constraints_t* constraints,
    int size_class,
    int shape_class) {
    (void)constraints;
    if (!tables || op_id >= (uint32_t)FB_MAX_OPERATIONS) {
        return NULL;
    }
    
    if (size_class < 0 || size_class >= 5 || shape_class < 0 || shape_class >= 5) {
        return NULL;
    }
    
    // Placeholder: Return fastest available operation for this op_id
    if (tables->fastest[op_id].top_n[0].func_ptr != NULL) {
        return &tables->fastest[op_id].top_n[0];
    }
    
    return NULL;
}

/**
 * Load dispatch tables from disk cache
 *
 * @param cache_path Path to dispatch table file
 * @param tables Output: loaded dispatch tables
 * @return 0 on success, negative on error
 */
int fb_dispatch_tables_load(const char *cache_path, fb_dispatch_tables_t *tables) {
    if (!cache_path || !tables) {
        return -1;
    }
    
    // PLACEHOLDER: Implement binary format loading
    // Format: Magic + version + tables data
    // Size: ~12MB (FB_MAX_OPERATIONS ops × 5 criteria × 5 entries × sizeof(entry))
    
    memset(tables, 0, sizeof(fb_dispatch_tables_t));
    return 0;  // TODO: Actual implementation
}

/**
 * Save dispatch tables to disk cache
 *
 * @param cache_path Path to dispatch table file
 * @param tables Dispatch tables to save
 * @return 0 on success, negative on error
 */
int fb_dispatch_tables_save(const char *cache_path, const fb_dispatch_tables_t *tables) {
    if (!cache_path || !tables) {
        return -1;
    }
    
    // PLACEHOLDER: Implement binary format saving
    // Format: Magic + version + tables data
    // Size: ~5MB
    
    return 0;  // TODO: Actual implementation
}

/**
 * Check if dispatch tables are valid (all operations have at least one candidate)
 *
 * @param tables Dispatch tables to validate
 * @return Number of operations with at least one candidate
 */
int fb_dispatch_tables_validate(const fb_dispatch_tables_t *tables) {
    if (!tables) {
        return 0;
    }
    
    // Placeholder: For now, return 0 (tables are empty during development)
    // Full implementation will count operations with valid candidates
    return 0;
}

/**
 * Get dispatch statistics for diagnostics
 *
 * @param tables Dispatch tables
 * @param stats Output: statistics structure
 * @return 0 on success
 */
int fb_dispatch_tables_get_stats(const fb_dispatch_tables_t *tables,
                                  void *stats) {
    if (!tables || !stats) {
        return -1;
    }
    
    // Placeholder: Will be implemented when stats structure is defined
    return 0;
}

/* =========================================================================
 * Constraint-based backend selection
 * ========================================================================= */

uint32_t fb_select_backend_with_constraints(
    uint32_t                          op_id,
    const fb_dispatch_constraints_t  *constraints)
{
    const fb_ranked_table_t *tbl = fb_op_dispatch_get_table();

    /* No table registered yet — reference backend is always correct. */
    if (!tbl)
        return FB_BACKEND_ID_REFERENCE;

    /* NULL constraints — balanced default. */
    if (!constraints)
        return fb_ranked_get_best(tbl, op_id, FB_RANK_BALANCED);

    /* Scale user accuracy (0-255) to digit score (0-15).
     * Use the stricter of min_accuracy / min_precision. */
    uint8_t acc = (constraints->min_accuracy > constraints->min_precision)
                      ? constraints->min_accuracy
                      : constraints->min_precision;
    uint8_t min_dig = (uint8_t)((acc * 15u + 127u) / 255u);

    bool has_acc = (min_dig > 0);
    bool has_lat = (constraints->max_time_us > 0);

    if (!has_acc && !has_lat) {
        /* No hard constraints — honour preferred criterion. */
        switch (constraints->prefer_criterion) {
        case FB_OPTIMIZE_FASTEST:
        case FB_OPTIMIZE_POWER_EFFICIENT:
            return fb_ranked_get_best(tbl, op_id, FB_RANK_FASTEST);
        case FB_OPTIMIZE_ACCURACY:
        case FB_OPTIMIZE_PRECISION:
            return fb_ranked_get_best(tbl, op_id, FB_RANK_MOST_ACCURATE);
        case FB_OPTIMIZE_BALANCED:
        default:
            return fb_ranked_get_best(tbl, op_id, FB_RANK_BALANCED);
        }
    }

    /* At least one hard constraint: walk MOST_ACCURATE list so that when
     * multiple backends pass all constraints we prefer the most precise one.
     * Skip entries that violate the latency ceiling or the accuracy floor. */
    uint32_t max_lat_ns = constraints->max_time_us * 1000u;
    uint32_t count = fb_ranked_entry_count(tbl, op_id, FB_RANK_MOST_ACCURATE);

    for (uint32_t r = 0; r < count; r++) {
        const fb_ranked_entry_t *e =
            fb_ranked_get_entry(tbl, op_id, FB_RANK_MOST_ACCURATE, r);
        if (!e || !e->is_valid) break;
        if (!fb_ranked_is_available(tbl, e->backend_id)) continue;
        if (has_acc && e->effective_digits < min_dig) continue;
        /* latency_p50_ns == UINT32_MAX means "no profile" : treat as unknown
         * and let the entry pass the latency check (conservative fallback). */
        if (has_lat && e->latency_p50_ns != UINT32_MAX
                    && e->latency_p50_ns > max_lat_ns)
            continue;
        return e->backend_id;
    }

    /* Nothing met all hard constraints. */
    if (constraints->require_exact_precision)
        return FB_BACKEND_ID_REFERENCE;  /* hard failure: no degradation */

    /* Degraded mode: ignore constraints, return best for preferred criterion. */
    switch (constraints->prefer_criterion) {
    case FB_OPTIMIZE_FASTEST:
    case FB_OPTIMIZE_POWER_EFFICIENT:
        return fb_ranked_get_best(tbl, op_id, FB_RANK_FASTEST);
    case FB_OPTIMIZE_ACCURACY:
    case FB_OPTIMIZE_PRECISION:
        return fb_ranked_get_best(tbl, op_id, FB_RANK_MOST_ACCURATE);
    case FB_OPTIMIZE_BALANCED:
    default:
        return fb_ranked_get_best(tbl, op_id, FB_RANK_BALANCED);
    }
}
