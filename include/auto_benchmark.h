#ifndef FASTER_BLASTER_AUTO_BENCHMARK_H
#define FASTER_BLASTER_AUTO_BENCHMARK_H

#include "benchmark_types.h"
#include "dispatch_tables.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Automatic Benchmark System
 * 
 * Detects hardware/backend changes on initialization and automatically
 * benchmarks if needed. No CLI tool required - everything is automatic!
 * 
 * Workflow:
 * 1. fb_init() calls fb_auto_benchmark_check()
 * 2. System generates hardware fingerprint
 * 3. Loads existing cache/dispatch tables
 * 4. Compares fingerprints
 * 5. If mismatch → auto-benchmark in background
 * 6. If match → use cached data
 */

/**
 * Change detection reasons (when to re-benchmark)
 */
typedef enum {
    FB_CHANGE_NONE = 0,                 // No changes detected
    FB_CHANGE_FIRST_RUN,                // First run - no cache exists
    FB_CHANGE_CPU,                      // CPU changed
    FB_CHANGE_GPU_ADDED,                // New GPU detected
    FB_CHANGE_GPU_REMOVED,              // GPU removed
    FB_CHANGE_MEMORY,                   // Memory size changed significantly
    FB_CHANGE_BACKEND_VERSION,          // Backend library updated (e.g., cuBLAS 13.0→13.1)
    FB_CHANGE_DRIVER_VERSION,           // GPU driver updated
    FB_CHANGE_SCHEMA_VERSION,           // Benchmark data format changed
    FB_CHANGE_HIGH_SYSTEM_LOAD,         // Initial benchmark done under high load (>80% CPU)
    FB_CHANGE_THERMAL_THROTTLING,       // Thermal throttling detected during benchmark
    FB_CHANGE_SUSPICIOUS_RESULTS,       // Results inconsistent (high variance, outliers)
    FB_CHANGE_BIOS_UPDATE,              // BIOS/firmware version changed
    FB_CHANGE_MANUAL_REQUEST            // User forced re-benchmark
} fb_change_reason_t;

/**
 * Auto-benchmark status
 */
typedef enum {
    FB_AUTO_BENCH_IDLE = 0,             // Not running
    FB_AUTO_BENCH_DETECTING,            // Checking for changes
    FB_AUTO_BENCH_BENCHMARKING,         // Running benchmarks
    FB_AUTO_BENCH_GENERATING_TABLES,    // Generating dispatch tables
    FB_AUTO_BENCH_COMPLETE,             // Finished successfully
    FB_AUTO_BENCH_ERROR                 // Error occurred
} fb_auto_benchmark_status_t;

/**
 * Auto-benchmark configuration
 */
typedef struct {
    bool enabled;                       // Enable automatic benchmarking
    bool quick_mode_first_run;          // Use quick benchmark on first run
    bool background;                    // Run in background thread
    uint32_t max_benchmark_time_sec;    // Max time for benchmarking (0=unlimited)
    void (*progress_callback)(float);   // Progress callback (0.0-1.0)
    void (*status_callback)(fb_auto_benchmark_status_t, const char*); // Status updates
} fb_auto_benchmark_config_t;

/**
 * Auto-benchmark result
 */
typedef struct {
    fb_change_reason_t change_reason;   // Why benchmarking was needed
    bool benchmarking_performed;        // Did we actually benchmark?
    uint32_t operations_benchmarked;    // Number of operations tested
    uint32_t operations_passed;         // Number that passed
    uint32_t operations_failed;         // Number that failed
    float elapsed_seconds;              // Time taken
    fb_dispatch_tables_t* dispatch_tables; // Generated dispatch tables
} fb_auto_benchmark_result_t;

// ============================================================================
// Automatic Benchmark Detection & Execution
// ============================================================================

/**
 * Check if benchmarking is needed and auto-benchmark if necessary
 * 
 * Called automatically by fb_init(). This is the main entry point.
 * 
 * @param config Auto-benchmark configuration (NULL = defaults)
 * @param result_out Output: Benchmark result (NULL if not interested)
 * @return 0 on success, negative on error
 */
int fb_auto_benchmark_check(
    const fb_auto_benchmark_config_t* config,
    fb_auto_benchmark_result_t* result_out
);

/**
 * Detect if hardware or backends have changed
 * 
 * Compares current hardware fingerprint with cached fingerprint.
 * 
 * @param change_reason_out Output: Why re-benchmark is needed
 * @return true if changes detected, false otherwise
 */
bool fb_detect_hardware_changes(fb_change_reason_t* change_reason_out);

/**
 * Force re-benchmark on next initialization
 * 
 * Sets a flag file that triggers re-benchmark even if hardware unchanged.
 * Useful for debugging or after manually updating backends.
 * 
 * @return 0 on success, negative on error
 */
int fb_force_rebenchmark(void);

/**
 * Clear force-rebenchmark flag
 * 
 * @return 0 on success, negative on error
 */
int fb_clear_rebenchmark_flag(void);

/**
 * Check if force-rebenchmark flag is set
 * 
 * @return true if flag set, false otherwise
 */
bool fb_is_rebenchmark_requested(void);

// ============================================================================
// Background Benchmarking
// ============================================================================

/**
 * Start benchmarking in background thread
 * 
 * Non-blocking. Use fb_auto_benchmark_wait() or callbacks to track progress.
 * 
 * @param config Auto-benchmark configuration
 * @return 0 on success, negative on error
 */
int fb_auto_benchmark_start_async(const fb_auto_benchmark_config_t* config);

/**
 * Get current auto-benchmark status
 * 
 * @return Current status
 */
fb_auto_benchmark_status_t fb_auto_benchmark_get_status(void);

/**
 * Get auto-benchmark progress
 * 
 * @return Progress (0.0-1.0)
 */
float fb_auto_benchmark_get_progress(void);

/**
 * Wait for background benchmarking to complete
 * 
 * Blocks until benchmarking finishes.
 * 
 * @param result_out Output: Benchmark result
 * @return 0 on success, negative on error
 */
int fb_auto_benchmark_wait(fb_auto_benchmark_result_t* result_out);

/**
 * Cancel ongoing background benchmarking
 * 
 * @return 0 on success, negative on error
 */
int fb_auto_benchmark_cancel(void);

// ============================================================================
// Query & Diagnostics
// ============================================================================

/**
 * Get string description of change reason
 * 
 * @param reason Change reason enum
 * @return Human-readable string
 */
const char* fb_change_reason_string(fb_change_reason_t reason);

/**
 * Get string description of auto-benchmark status
 * 
 * @param status Status enum
 * @return Human-readable string
 */
const char* fb_auto_benchmark_status_string(fb_auto_benchmark_status_t status);

/**
 * Get default auto-benchmark configuration
 * 
 * @param config_out Output: Default configuration
 */
void fb_get_default_auto_benchmark_config(fb_auto_benchmark_config_t* config_out);

/**
 * Print auto-benchmark result summary
 * 
 * @param result Benchmark result
 */
void fb_print_auto_benchmark_result(const fb_auto_benchmark_result_t* result);

#ifdef __cplusplus
}
#endif

#endif // FASTER_BLASTER_AUTO_BENCHMARK_H
