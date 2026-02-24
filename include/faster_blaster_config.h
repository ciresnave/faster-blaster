/**
 * @file faster_blaster_config.h
 * @brief Runtime configuration and control for faster-blaster
 * 
 * This header provides functions to:
 * - Configure which backends are enabled/disabled
 * - Control calibration behavior
 * - Query performance statistics
 * - Export/import hardware profiles
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_CONFIG_H
#define FASTER_BLASTER_CONFIG_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Backend Management
 * ========================================================================= */

/**
 * @brief Backend identifier
 */
typedef enum {
    FB_BACKEND_REFERENCE  = 0,  /**< Built-in reference implementation */
    FB_BACKEND_OPENBLAS   = 1,  /**< OpenBLAS */
    FB_BACKEND_BLIS       = 2,  /**< BLIS */
    FB_BACKEND_MKL        = 3,  /**< Intel MKL */
    FB_BACKEND_CUBLAS     = 4,  /**< NVIDIA cuBLAS */
    FB_BACKEND_ROCBLAS    = 5,  /**< AMD rocBLAS */
    FB_BACKEND_ACCELERATE = 6,  /**< Apple Accelerate */
    FB_BACKEND_AOCL       = 7,  /**< AMD Optimizing CPU Libraries */
    FB_BACKEND_COUNT              /**< Total number of backends */
} FB_BACKEND_ID;

/**
 * @brief Backend information structure
 */
typedef struct {
    FB_BACKEND_ID id;           /**< Backend identifier */
    const char *name;           /**< Human-readable name */
    const char *version;        /**< Version string */
    bool available;             /**< Whether backend is available on this system */
    bool enabled;               /**< Whether backend is currently enabled */
    bool supports_cpu;          /**< Can run on CPU */
    bool supports_gpu;          /**< Can run on GPU */
    uint32_t capabilities;      /**< Capability bitmask */
} fb_backend_info_t;

/**
 * @brief Get list of all available backends
 * 
 * @param info_array Output array for backend information
 * @param array_size Size of info_array
 * @return Number of backends written to array
 */
int fb_get_backend_list(fb_backend_info_t *info_array, int array_size);

/**
 * @brief Enable a specific backend
 * 
 * When a backend is enabled, its functions become candidates for dispatch.
 * The system will automatically re-optimize the dispatch table based on
 * previously stored calibration data.
 * 
 * @param backend Backend to enable
 * @return 0 on success, negative error code on failure
 */
int fb_enable_backend(FB_BACKEND_ID backend);

/**
 * @brief Disable a specific backend
 * 
 * Disabled backends will not be used for any operations. The dispatch table
 * will be updated to use the next-best backend for each operation.
 * 
 * @param backend Backend to disable
 * @return 0 on success, negative error code on failure
 */
int fb_disable_backend(FB_BACKEND_ID backend);

/**
 * @brief Force use of a specific backend for all operations
 * 
 * Overrides the auto-dispatch mechanism. Useful for debugging or when
 * the user knows a specific backend is always best for their workload.
 * 
 * @param backend Backend to force, or -1 to restore auto-dispatch
 * @return 0 on success, negative error code on failure
 */
int fb_force_backend(FB_BACKEND_ID backend);

/**
 * @brief Check which backend is currently selected for an operation
 * 
 * @param operation_name Name of the operation (e.g., "sgemm", "daxpy")
 * @param M Matrix/vector dimension M (0 if not applicable)
 * @param N Matrix/vector dimension N (0 if not applicable)
 * @param K Matrix/vector dimension K (0 if not applicable)
 * @return Backend ID that would be used for this operation
 */
FB_BACKEND_ID fb_query_dispatch(const char *operation_name, int M, int N, int K);

/* ============================================================================
 * Calibration Control
 * ========================================================================= */

/**
 * @brief Calibration progress callback
 * 
 * Called periodically during calibration to report progress.
 * 
 * @param operation Current operation being calibrated
 * @param backend Current backend being tested
 * @param progress Progress percentage (0-100)
 * @param user_data User-provided data pointer
 */
typedef void (*fb_calibration_callback_t)(
    const char *operation,
    FB_BACKEND_ID backend,
    int progress,
    void *user_data
);

/**
 * @brief Calibration configuration
 */
typedef struct {
    bool skip_if_cached;            /**< Skip calibration if hardware profile exists */
    bool verbose;                   /**< Print detailed calibration info */
    fb_calibration_callback_t callback;  /**< Progress callback (optional) */
    void *callback_user_data;       /**< User data for callback */
    int warmup_iterations;          /**< Number of warmup iterations per test (default: 3) */
    int timing_iterations;          /**< Number of timing iterations per test (default: 10) */
    bool test_correctness;          /**< Verify correctness against reference (default: true) */
    bool test_precision;            /**< Measure numerical precision (default: true) */
    bool test_memory;               /**< Measure memory usage (default: false) */
    double timeout_seconds;         /**< Maximum time per operation test (default: 5.0) */
} fb_calibration_config_t;

/**
 * @brief Get default calibration configuration
 * 
 * @param config Output parameter for default configuration
 */
void fb_calibration_config_default(fb_calibration_config_t *config);

/**
 * @brief Run calibration for the current hardware
 * 
 * This will benchmark all available backends across a range of problem sizes
 * and data types, measuring performance, correctness, and precision. Results
 * are automatically saved to the user's config directory.
 * 
 * @param config Calibration configuration (NULL for defaults)
 * @return 0 on success, negative error code on failure
 */
int fb_run_calibration(const fb_calibration_config_t *config);

/**
 * @brief Check if calibration data exists for this hardware
 * 
 * @return true if calibration data is available, false otherwise
 */
bool fb_has_calibration_data(void);

/**
 * @brief Export calibration data to file
 * 
 * Creates a JSON file containing the hardware fingerprint and all
 * calibration results. This file can be:
 * - Shared with the faster-blaster project to improve the database
 * - Transferred to another machine with identical hardware
 * - Used for analysis and debugging
 * 
 * @param output_path Path to output JSON file
 * @return 0 on success, negative error code on failure
 */
int fb_export_calibration(const char *output_path);

/**
 * @brief Import calibration data from file
 * 
 * Loads calibration data from a JSON file. If the hardware fingerprint
 * matches the current system, the data will be used for dispatch.
 * 
 * @param input_path Path to input JSON file
 * @return 0 on success, negative error code on failure
 */
int fb_import_calibration(const char *input_path);

/**
 * @brief Clear all cached calibration data
 * 
 * Forces a re-calibration on next fb_init(). Useful for testing or
 * if calibration data becomes corrupted.
 * 
 * @return 0 on success, negative error code on failure
 */
int fb_clear_calibration_cache(void);

/* ============================================================================
 * Hardware Detection
 * ========================================================================= */

/**
 * @brief Hardware information structure
 */
typedef struct {
    char cpu_vendor[64];        /**< CPU vendor (Intel, AMD, ARM, etc.) */
    char cpu_model[128];        /**< CPU model name */
    int cpu_cores;              /**< Number of physical CPU cores */
    int cpu_threads;            /**< Number of logical threads */
    uint64_t cpu_l1_cache;      /**< L1 cache size in bytes */
    uint64_t cpu_l2_cache;      /**< L2 cache size in bytes */
    uint64_t cpu_l3_cache;      /**< L3 cache size in bytes */
    uint64_t ram_bytes;         /**< Total system RAM in bytes */
    
    bool has_gpu;               /**< Whether GPU is available */
    char gpu_vendor[64];        /**< GPU vendor (NVIDIA, AMD, Intel, Apple) */
    char gpu_model[128];        /**< GPU model name */
    int gpu_compute_units;      /**< Number of compute units / SMs */
    uint64_t gpu_memory_bytes;  /**< GPU memory in bytes */
    
    char fingerprint[64];       /**< Unique hardware fingerprint hash */
} fb_hardware_info_t;

/**
 * @brief Get information about the current hardware
 * 
 * @param info Output parameter for hardware information
 * @return 0 on success, negative error code on failure
 */
int fb_get_hardware_info(fb_hardware_info_t *info);

/* ============================================================================
 * Performance Statistics
 * ========================================================================= */

/**
 * @brief Operation statistics
 */
typedef struct {
    uint64_t call_count;        /**< Number of times operation was called */
    double total_time_ns;       /**< Total time spent in this operation */
    double min_time_ns;         /**< Fastest execution time */
    double max_time_ns;         /**< Slowest execution time */
    double avg_time_ns;         /**< Average execution time */
    uint64_t total_flops;       /**< Total floating-point operations */
    FB_BACKEND_ID last_backend; /**< Backend used for last call */
} fb_operation_stats_t;

/**
 * @brief Enable performance statistics collection
 * 
 * When enabled, the library tracks timing and usage statistics for all
 * operations. This adds minimal overhead (<1%) but provides valuable
 * profiling information.
 * 
 * @param enable true to enable, false to disable
 */
void fb_enable_statistics(bool enable);

/**
 * @brief Get statistics for a specific operation
 * 
 * @param operation_name Name of the operation (e.g., "sgemm")
 * @param stats Output parameter for statistics
 * @return 0 on success, negative error code if operation not found
 */
int fb_get_operation_stats(const char *operation_name, fb_operation_stats_t *stats);

/**
 * @brief Print statistics summary to stdout
 * 
 * Displays a formatted table of all operations with their performance metrics.
 */
void fb_print_statistics(void);

/**
 * @brief Reset all statistics counters
 */
void fb_reset_statistics(void);

/* ============================================================================
 * Debugging and Diagnostics
 * ========================================================================= */

/**
 * @brief Logging level
 */
typedef enum {
    FB_LOG_NONE    = 0,  /**< No logging */
    FB_LOG_ERROR   = 1,  /**< Errors only */
    FB_LOG_WARNING = 2,  /**< Warnings and errors */
    FB_LOG_INFO    = 3,  /**< General information */
    FB_LOG_DEBUG   = 4,  /**< Detailed debug information */
    FB_LOG_TRACE   = 5   /**< Extremely verbose trace logging */
} FB_LOG_LEVEL;

/**
 * @brief Set logging level
 * 
 * @param level Desired logging level
 */
void fb_set_log_level(FB_LOG_LEVEL level);

/**
 * @brief Custom logging callback
 * 
 * @param level Message log level
 * @param message Log message
 * @param user_data User-provided data pointer
 */
typedef void (*fb_log_callback_t)(FB_LOG_LEVEL level, const char *message, void *user_data);

/**
 * @brief Set custom logging callback
 * 
 * By default, logs go to stderr. This allows redirecting to a custom handler.
 * 
 * @param callback Logging callback function (NULL to restore default)
 * @param user_data User data passed to callback
 */
void fb_set_log_callback(fb_log_callback_t callback, void *user_data);

/**
 * @brief Run self-test suite
 * 
 * Performs correctness tests on all enabled backends across various
 * problem sizes and data types. Useful for verifying installation.
 * 
 * @return 0 if all tests pass, negative error code on failure
 */
int fb_run_self_test(void);

/**
 * @brief Verify an operation against reference implementation
 * 
 * Performs the operation using both the current backend and the reference
 * implementation, then compares results. Useful for debugging precision issues.
 * 
 * @param operation_name Name of operation to verify
 * @param ... Operation-specific parameters
 * @return 0 if results match within tolerance, negative error code if mismatch
 */
int fb_verify_operation(const char *operation_name, ...);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_CONFIG_H */
