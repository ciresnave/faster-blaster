#ifndef FASTER_BLASTER_DISPATCH_TABLES_H
#define FASTER_BLASTER_DISPATCH_TABLES_H

#include "benchmark_types.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Ranked Dispatch Tables - Phase 3.0 Architecture
 * 
 * CRITICAL DESIGN PRINCIPLE:
 * We DON'T generate vtables from benchmarks - we use benchmarks to RANK operations!
 * 
 * Each optimization criterion (FASTEST, MOST_ACCURATE, LOWEST_POWER) has its own
 * ranked list of top N operations. The dispatcher selects from this list based on:
 * - User constraints (min_accuracy, max_time_us, priority level)
 * - Runtime conditions (thermal throttling, battery status, system load)
 * - Operation availability (some backends may crash, timeout, or be disabled)
 */

#define FB_MAX_OPERATIONS 1248
#define FB_TOP_N_RANKED 5   // Keep top 5 candidates per optimization criterion

/**
 * Single ranked operation entry
 * Contains function pointer + full benchmark metrics
 */
typedef struct {
    void* func_ptr;                     // Function pointer (NULL if not available)
    uint32_t backend_device_id;         // Which backend+device combo
    fb_benchmark_stats_t metrics;       // Full benchmark statistics
    bool available;                     // Currently available (not crashed/disabled)
} fb_ranked_operation_t;

/**
 * Vtable entry: Top N operations ranked by specific criterion
 */
typedef struct {
    fb_ranked_operation_t top_n[FB_TOP_N_RANKED];
    uint8_t count;                      // Actual number of valid entries (≤ TOP_N)
} fb_vtable_entry_t;

/**
 * Optimization criteria for dispatch
 */
typedef enum {
    FB_OPTIMIZE_FASTEST = 0,            // Minimize time_mean_ns
    FB_OPTIMIZE_ACCURACY,               // Maximize accuracy_mean
    FB_OPTIMIZE_PRECISION,              // Maximize precision_mean
    FB_OPTIMIZE_POWER_EFFICIENT,        // Minimize power consumption
    FB_OPTIMIZE_BALANCED,               // Weighted combination
    FB_OPTIMIZE_CRITERION_COUNT         // Total: 5 criteria
} fb_optimization_criterion_t;

/**
 * Complete dispatch table set
 * One ranked table per optimization criterion
 */
typedef struct {
    // Ranked tables by optimization criterion
    fb_vtable_entry_t fastest[FB_MAX_OPERATIONS];
    fb_vtable_entry_t most_accurate[FB_MAX_OPERATIONS];
    fb_vtable_entry_t most_precise[FB_MAX_OPERATIONS];
    fb_vtable_entry_t most_power_efficient[FB_MAX_OPERATIONS];
    fb_vtable_entry_t balanced[FB_MAX_OPERATIONS];
    
    // Metadata
    fb_hardware_fingerprint_t hw_fingerprint;
    uint64_t timestamp_generated;
    uint32_t total_backends;
} fb_dispatch_tables_t;

/**
 * User constraints for operation dispatch
 */
typedef struct {
    // Hard constraints (fail if unmet)
    uint8_t min_accuracy;               // 0-255 (0 = no constraint)
    uint8_t min_precision;              // 0-255 (0 = no constraint)
    uint32_t max_time_us;               // Maximum time in microseconds (0 = no constraint)
    
    // Soft constraints (best effort)
    fb_optimization_criterion_t prefer_criterion;  // Preferred optimization
    uint8_t priority;                   // 0-255 (0=background, 255=critical)
    
    // Behavioral flags
    bool allow_dtype_conversion;        // Allow D→S→D if faster+accurate enough
    bool allow_transpose;               // Allow transpose optimization
    bool require_exact_precision;       // Don't allow any precision loss
} fb_dispatch_constraints_t;

// ============================================================================
// Dispatch Table Generation
// ============================================================================

/**
 * Generate dispatch tables from benchmark cache
 * 
 * This is the key function that transforms benchmark data into ranked dispatch tables.
 * For each operation and optimization criterion:
 * 1. Collect all benchmark results across all backends/devices/sizes/shapes
 * 2. Sort by the criterion (fastest, most accurate, etc.)
 * 3. Store top N in ranked table with full metrics
 * 
 * @param cache Benchmark cache (already loaded)
 * @param tables Output: Generated dispatch tables
 * @return 0 on success, negative on error
 */
int fb_generate_dispatch_tables(void* cache, fb_dispatch_tables_t* tables);

/**
 * Update dispatch tables when hardware/backends change
 * 
 * Re-ranks operations based on new benchmark data
 * 
 * @param cache Updated benchmark cache
 * @param tables Existing dispatch tables to update
 * @return 0 on success, negative on error
 */
int fb_update_dispatch_tables(void* cache, fb_dispatch_tables_t* tables);

/**
 * Save dispatch tables to disk
 * 
 * @param tables Dispatch tables
 * @param path File path (NULL = default ~/.cache/faster-blaster/dispatch_tables.bin)
 * @return 0 on success, negative on error
 */
int fb_save_dispatch_tables(const fb_dispatch_tables_t* tables, const char* path);

/**
 * Load dispatch tables from disk
 * 
 * Returns NULL if:
 * - File doesn't exist
 * - Hardware fingerprint mismatch
 * - Version incompatible
 * 
 * @param path File path (NULL = default)
 * @return Loaded dispatch tables, or NULL on failure
 */
fb_dispatch_tables_t* fb_load_dispatch_tables(const char* path);

/**
 * Free dispatch tables
 */
void fb_free_dispatch_tables(fb_dispatch_tables_t* tables);

// ============================================================================
// Operation Selection (Runtime Dispatch)
// ============================================================================

/**
 * Select best operation from ranked table given constraints
 * 
 * This is called at runtime for each operation dispatch.
 * Iterates through top N ranked operations and selects first one that:
 * 1. Meets all hard constraints (min_accuracy, max_time)
 * 2. Is currently available (not crashed/disabled)
 * 3. Best matches soft constraints (preferred criterion, priority)
 * 
 * @param tables Complete dispatch table set
 * @param operation_id Operation ID (0-1247)
 * @param constraints User constraints
 * @param size_class Current problem size class
 * @param shape_class Current problem shape class
 * @return Selected operation entry (NULL if none match constraints)
 */
const fb_ranked_operation_t* fb_select_operation(
    const fb_dispatch_tables_t* tables,
    uint32_t operation_id,
    const fb_dispatch_constraints_t* constraints,
    fb_size_class_t size_class,
    fb_shape_class_t shape_class
);

/**
 * Mark operation as temporarily unavailable (crashed/timeout)
 * 
 * Updates the 'available' flag so dispatcher skips this entry.
 * This implements the implicit blocklist.
 * 
 * @param tables Dispatch tables
 * @param operation_id Operation ID
 * @param backend_device_id Backend+device combo
 * @return 0 on success, negative on error
 */
int fb_mark_operation_unavailable(
    fb_dispatch_tables_t* tables,
    uint32_t operation_id,
    uint32_t backend_device_id
);

/**
 * Re-enable previously disabled operation
 * 
 * @param tables Dispatch tables
 * @param operation_id Operation ID
 * @param backend_device_id Backend+device combo
 * @return 0 on success, negative on error
 */
int fb_mark_operation_available(
    fb_dispatch_tables_t* tables,
    uint32_t operation_id,
    uint32_t backend_device_id
);

// ============================================================================
// Query Functions
// ============================================================================

/**
 * Get all ranked operations for an operation ID and criterion
 * 
 * @param tables Dispatch tables
 * @param operation_id Operation ID (0-1247)
 * @param criterion Optimization criterion
 * @return Vtable entry with top N ranked operations
 */
const fb_vtable_entry_t* fb_get_ranked_operations(
    const fb_dispatch_tables_t* tables,
    uint32_t operation_id,
    fb_optimization_criterion_t criterion
);

/**
 * Get best operation for given criterion (ignoring constraints)
 * 
 * @param tables Dispatch tables
 * @param operation_id Operation ID
 * @param criterion Optimization criterion
 * @return Best ranked operation (may be NULL if none available)
 */
const fb_ranked_operation_t* fb_get_best_operation(
    const fb_dispatch_tables_t* tables,
    uint32_t operation_id,
    fb_optimization_criterion_t criterion
);

/**
 * Print dispatch tables summary
 * 
 * @param tables Dispatch tables
 */
void fb_print_dispatch_tables_summary(const fb_dispatch_tables_t* tables);

#ifdef __cplusplus
}
#endif

#endif // FASTER_BLASTER_DISPATCH_TABLES_H
