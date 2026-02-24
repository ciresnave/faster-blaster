#ifndef FASTER_BLASTER_BENCHMARK_SYSTEM_H
#define FASTER_BLASTER_BENCHMARK_SYSTEM_H

#include "benchmark_types.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Benchmark System - Phase 3.0
 * 
 * Safe, comprehensive benchmarking of all 1248 BLAS/LAPACK operations
 * across all available backend+device combinations.
 * 
 * Key features:
 * - Protected execution (crash/timeout detection)
 * - Size and shape stratification (5×5 = 25 configs per op)
 * - Statistical sampling (warmup + N iterations)
 * - Hardware-aware caching (~20MB per system)
 * - Implicit blocklist (failed ops → vtable[op] = NULL)
 */

// ============================================================================
// Size and Shape Classification
// ============================================================================

/**
 * Classify problem size into one of 5 size classes
 * 
 * @param m First dimension
 * @param n Second dimension
 * @param k Third dimension (0 if not applicable)
 * @return Size class enum
 */
fb_size_class_t fb_classify_size(size_t m, size_t n, size_t k);

/**
 * Classify matrix shape into one of 5 shape classes
 * 
 * @param m Number of rows
 * @param n Number of columns
 * @return Shape class enum
 */
fb_shape_class_t fb_classify_shape(size_t m, size_t n);

/**
 * Get representative problem sizes for a given size class
 * 
 * @param size_class The size class
 * @param m_out Output: M dimension
 * @param n_out Output: N dimension
 * @param k_out Output: K dimension (may be 0)
 */
void fb_get_representative_size(fb_size_class_t size_class, 
                                size_t* m_out, 
                                size_t* n_out, 
                                size_t* k_out);

/**
 * Get representative shape dimensions for a given shape class
 * 
 * @param shape_class The shape class
 * @param base_size Base size for scaling (e.g., 1024)
 * @param m_out Output: M dimension
 * @param n_out Output: N dimension
 */
void fb_get_representative_shape(fb_shape_class_t shape_class,
                                 size_t base_size,
                                 size_t* m_out,
                                 size_t* n_out);

// ============================================================================
// Hardware Fingerprinting
// ============================================================================

/**
 * Generate hardware fingerprint for current system
 * 
 * Includes: CPU vendor/model, GPU device IDs, RAM size, backend versions
 * 
 * @param fingerprint Output: Hardware fingerprint structure
 * @return 0 on success, negative on error
 */
int fb_generate_hardware_fingerprint(fb_hardware_fingerprint_t* fingerprint);

/**
 * Compare two hardware fingerprints for equality
 * 
 * @param a First fingerprint
 * @param b Second fingerprint
 * @return true if fingerprints match, false otherwise
 */
bool fb_fingerprints_match(const fb_hardware_fingerprint_t* a,
                           const fb_hardware_fingerprint_t* b);

// ============================================================================
// Benchmark Cache Management
// ============================================================================

/**
 * Load benchmark cache from disk
 * 
 * Loads cached benchmark results. Returns NULL if:
 * - Cache file doesn't exist
 * - Hardware fingerprint mismatch
 * - Backend versions changed
 * - Schema version incompatible
 * - Checksum validation failed
 * 
 * @param cache_path Path to cache file (NULL = default ~/.cache/faster-blaster/benchmarks.bin)
 * @return Benchmark cache handle, or NULL on failure
 */
void* fb_benchmark_cache_load(const char* cache_path);

/**
 * Save benchmark cache to disk
 * 
 * @param cache Benchmark cache handle
 * @param cache_path Path to cache file (NULL = default)
 * @return 0 on success, negative on error
 */
int fb_benchmark_cache_save(void* cache, const char* cache_path);

/**
 * Free benchmark cache
 * 
 * @param cache Benchmark cache handle
 */
void fb_benchmark_cache_free(void* cache);

/**
 * Query benchmark statistics from cache
 * 
 * @param cache Benchmark cache handle
 * @param operation_id Operation ID (0-1247)
 * @param backend_device_id Backend+device combo ID
 * @param size_class Size class (0-4)
 * @param shape_class Shape class (0-4)
 * @param stats_out Output: Benchmark statistics (if found)
 * @return true if found, false otherwise
 */
bool fb_benchmark_cache_query(void* cache,
                              uint32_t operation_id,
                              uint32_t backend_device_id,
                              fb_size_class_t size_class,
                              fb_shape_class_t shape_class,
                              fb_benchmark_stats_t* stats_out);

/**
 * Store benchmark statistics in cache
 * 
 * @param cache Benchmark cache handle
 * @param operation_id Operation ID (0-1247)
 * @param backend_device_id Backend+device combo ID
 * @param size_class Size class (0-4)
 * @param shape_class Shape class (0-4)
 * @param stats Benchmark statistics to store
 * @return 0 on success, negative on error
 */
int fb_benchmark_cache_store(void* cache,
                             uint32_t operation_id,
                             uint32_t backend_device_id,
                             fb_size_class_t size_class,
                             fb_shape_class_t shape_class,
                             const fb_benchmark_stats_t* stats);

// ============================================================================
// Safe Benchmark Execution
// ============================================================================

/**
 * Run a single benchmark with protected execution
 * 
 * Protects against:
 * - Crashes (segfault, access violations)
 * - Timeouts (operation hangs)
 * - NaN/Inf outputs
 * - Memory corruption (guard pages)
 * 
 * On failure, sets appropriate error flags in result.
 * 
 * @param operation_id Operation ID (0-1247)
 * @param backend_device_id Backend+device combo ID
 * @param config Benchmark configuration
 * @param result_out Output: Benchmark result
 * @return 0 on success (doesn't mean op succeeded, check result.success)
 */
int fb_benchmark_run_safe(uint32_t operation_id,
                          uint32_t backend_device_id,
                          const fb_benchmark_config_t* config,
                          fb_benchmark_result_t* result_out);

/**
 * Run full benchmark suite for a backend+device
 * 
 * Benchmarks all 1248 operations across all size/shape classes.
 * Failed operations are marked with error flags.
 * 
 * @param backend_device_id Backend+device combo ID
 * @param cache Benchmark cache to populate
 * @param progress_callback Optional callback for progress updates (0.0-1.0)
 * @return Number of operations successfully benchmarked
 */
int fb_benchmark_run_full(uint32_t backend_device_id,
                          void* cache,
                          void (*progress_callback)(float progress));

/**
 * Run quick benchmark suite (samples only)
 * 
 * Tests representative sizes/shapes only (~100 configs instead of 6240).
 * Useful for initial calibration.
 * 
 * @param backend_device_id Backend+device combo ID
 * @param cache Benchmark cache to populate
 * @param progress_callback Optional callback for progress updates
 * @return Number of operations successfully benchmarked
 */
int fb_benchmark_run_quick(uint32_t backend_device_id,
                           void* cache,
                           void (*progress_callback)(float progress));

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if operation passed benchmark
 * 
 * @param stats Benchmark statistics
 * @return true if operation is usable, false if should be blocked
 */
bool fb_benchmark_operation_usable(const fb_benchmark_stats_t* stats);

/**
 * Get default cache path for current platform
 * 
 * Windows: %LOCALAPPDATA%\faster-blaster\benchmarks.bin
 * Linux/macOS: ~/.cache/faster-blaster/benchmarks.bin
 * 
 * @param buffer Output buffer
 * @param buffer_size Size of buffer
 * @return 0 on success, negative on error
 */
int fb_get_default_cache_path(char* buffer, size_t buffer_size);

/**
 * Print benchmark statistics summary
 * 
 * @param stats Benchmark statistics
 */
void fb_print_benchmark_stats(const fb_benchmark_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif // FASTER_BLASTER_BENCHMARK_SYSTEM_H
