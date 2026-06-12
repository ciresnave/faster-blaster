/**
 * @file calibration.h
 * @brief Calibration system for benchmarking and optimizing backend selection
 * 
 * This module implements the calibration framework that:
 * 1. Benchmarks all available backends across problem sizes
 * 2. Verifies correctness against reference implementation
 * 3. Measures numerical precision and memory usage
 * 4. Generates optimal dispatch tables
 * 5. Manages calibration data persistence
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_CALIBRATION_H
#define FB_CALIBRATION_H

#include "../backends/backend_interface.h"
#include "hardware_detect.h"
#include "dispatch.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Calibration Configuration
 * ========================================================================= */

/* Forward declaration for callback */
struct fb_calibration_result_t;

/**
 * @brief Callback function type for calibration progress updates
 */
typedef void (*fb_calibration_callback_t)(
    const struct fb_calibration_result_t *result,
    void *user_data
);

/**
 * @brief Problem size descriptor for calibration (operation-agnostic)
 */
typedef struct {
    size_t m, n, k;              /**< Matrix/vector dimensions */
    fb_layout_t layout;          /**< Memory layout (ROW/COL_MAJOR) */
    fb_transpose_t trans_a;      /**< Transpose for A (if applicable) */
    fb_transpose_t trans_b;      /**< Transpose for B (if applicable) */
    fb_uplo_t uplo;              /**< Upper/lower triangle (if applicable) */
    fb_diag_t diag;              /**< Diagonal type (if applicable) */
    fb_side_t side;              /**< Side specification (if applicable) */
} fb_problem_size_t;

/**
 * @brief What to measure during calibration
 */
typedef struct {
    bool measure_time;           /**< Measure execution time */
    bool verify_correctness;     /**< Check results against reference */
    bool measure_precision;      /**< Measure numerical error */
    bool measure_memory;         /**< Track memory usage */
    bool measure_bandwidth;      /**< Calculate memory bandwidth */
    bool test_stability;         /**< Test with ill-conditioned matrices */
    
    int warmup_iterations;       /**< Number of warmup runs */
    int timing_iterations;       /**< Number of timing runs */
    double timeout_seconds;      /**< Max time per test */
    double error_tolerance;      /**< Acceptable numerical error */
} fb_calibration_metrics_config_t;

/**
 * @brief Results from a single calibration test
 */
typedef struct fb_calibration_result_t {
    /* Performance metrics */
    double execution_time_ns;           /**< Median execution time */
    double execution_time_min_ns;       /**< Minimum execution time */
    double execution_time_max_ns;       /**< Maximum execution time */
    double execution_time_stddev_ns;    /**< Standard deviation */
    double first_run_time_ns;           /**< First run time (includes warmup) */
    double gflops;                      /**< Achieved GFLOPS */
    double memory_bandwidth_gbps;       /**< Memory bandwidth in GB/s */
    
    /* Memory metrics */
    size_t memory_peak_bytes;           /**< Peak memory allocation */
    size_t memory_allocated_bytes;      /**< Total memory allocated */
    size_t memory_freed_bytes;          /**< Total memory freed */
    
    /* Correctness metrics */
    bool correctness_passed;            /**< Did the result match reference? */
    double max_absolute_error;          /**< Maximum absolute element error */
    double max_relative_error;          /**< Maximum relative element error */
    double avg_absolute_error;          /**< Average absolute error */
    double rms_error;                   /**< Root mean square error */
    
    /* Numerical stability metrics */
    double condition_number;            /**< Condition number (if applicable) */
    double forward_error;               /**< Forward error estimate */
    double backward_error;              /**< Backward error estimate */
    int ill_conditioned_behavior;       /**< 0=good, 1=warning, 2=bad */
    
    /* Metadata */
    fb_backend_info_t backend_info;     /**< Which backend was tested */
    fb_problem_size_t problem_size;     /**< Test parameters */
    fb_operation_id_t operation_id;     /**< Which operation was tested */
    uint64_t timestamp;                 /**< Unix timestamp of test */
    bool timed_out;                     /**< Whether test exceeded timeout */
    bool crashed;                       /**< Whether backend crashed */
    char error_message[256];            /**< Error details if failed */
} fb_calibration_result_t;

/**
 * @brief Collection of calibration results for one operation
 */
typedef struct {
    const char *operation_name;         /**< BLAS operation name */
    fb_calibration_result_t *results;   /**< Array of results */
    size_t result_count;                /**< Number of results */
    size_t capacity;                    /**< Allocated capacity */
    
    /* Winner for each problem size/backend combination */
    int *best_backend_id;               /**< Best backend per problem size */
    double *speedup_vs_reference;       /**< Speedup compared to reference */
} fb_operation_calibration_t;

/**
 * @brief Complete calibration database for a hardware configuration
 */
typedef struct {
    fb_hardware_info_t hardware;        /**< Hardware this data is for */
    fb_operation_calibration_t *operations;  /**< Per-operation results */
    size_t operation_count;             /**< Number of operations calibrated */
    uint64_t calibration_timestamp;     /**< When calibration was performed */
    char library_version[32];           /**< faster-blaster version */
} fb_calibration_database_t;

/* ============================================================================
 * Calibration API
 * ========================================================================= */

/**
 * @brief Initialize calibration system
 * 
 * @return 0 on success, negative error code on failure
 */
int fb_calibration_init(void);

/**
 * @brief Load calibration data for current hardware
 * 
 * Searches for cached calibration data matching the current hardware
 * fingerprint. Looks in:
 * 1. User's config directory
 * 2. System-wide database
 * 3. Bundled database for common hardware
 * 
 * @param db Output parameter for loaded database
 * @return 0 if data found and loaded, negative error code otherwise
 */
int fb_calibration_load(fb_calibration_database_t **db);

/**
 * @brief Save calibration data to disk
 * 
 * Saves to user's config directory for future use.
 * 
 * @param db Database to save
 * @param path Optional custom path (NULL for default location)
 * @return 0 on success, negative error code on failure
 */
int fb_calibration_save(const fb_calibration_database_t *db, const char *path);

/**
 * @brief Create a new empty calibration database
 * 
 * @return New database, or NULL on failure
 */
fb_calibration_database_t* fb_calibration_database_create(void);

/**
 * @brief Free calibration database
 * 
 * @param db Database to free
 */
void fb_calibration_database_free(fb_calibration_database_t *db);

/**
 * @brief Progress callback for calibration
 * 
 * @param operation_name Current operation being tested
 * @param backend_id Current backend being tested
 * @param progress_percent Overall progress (0-100)
 * @param current_test_info Human-readable description of current test
 * @param user_data User-provided pointer
 */
typedef void (*fb_calibration_progress_fn)(
    const char *operation_name,
    int backend_id,
    int progress_percent,
    const char *current_test_info,
    void *user_data
);

/**
 * @brief Calibration runner configuration
 */
typedef struct {
    fb_calibration_metrics_config_t metrics;  /**< What to measure */
    fb_calibration_progress_fn progress_callback;  /**< Progress updates */
    void *progress_user_data;                /**< User data for callback */
    bool parallel;                            /**< Run tests in parallel */
    int thread_count;                         /**< Threads for parallel (0=auto) */
    bool skip_slow_backends;                  /**< Skip if backend is clearly slower */
    double slow_threshold;                    /**< Threshold for skipping (e.g., 2.0 = 2x slower) */
} fb_calibration_runner_config_t;

/**
 * @brief Get default calibration runner configuration
 * 
 * @param config Output parameter for default configuration
 */
void fb_calibration_runner_config_default(fb_calibration_runner_config_t *config);

/**
 * @brief Run full calibration suite
 * 
 * Benchmarks all registered backends across all standard BLAS operations
 * with a variety of problem sizes and data types.
 * 
 * @param config Calibration configuration
 * @param db Output database with results
 * @return 0 on success, negative error code on failure
 */
int fb_calibration_run_full(
    const fb_calibration_runner_config_t *config,
    fb_calibration_database_t **db
);

/**
 * @brief Run calibration for a specific operation
 * 
 * @param operation_name BLAS operation to calibrate (e.g., "gemm", "axpy")
 * @param backends Array of backend IDs to test
 * @param backend_count Number of backends in array
 * @param problem_sizes Array of problem sizes to test
 * @param problem_count Number of problem sizes
 * @param config Calibration configuration
 * @param results Output array of results
 * @return 0 on success, negative error code on failure
 */
int fb_calibration_run_operation(
    const char *operation_name,
    const int *backends,
    size_t backend_count,
    const fb_problem_size_t *problem_sizes,
    size_t problem_count,
    const fb_calibration_runner_config_t *config,
    fb_calibration_result_t **results
);

/**
 * @brief Standard problem size suites for different operation types
 */

/**
 * @brief Get problem sizes for Level 1 operations (vector-vector)
 * 
 * @param sizes Output array (allocated by caller)
 * @param max_sizes Maximum number of sizes to return
 * @return Number of sizes returned
 */
size_t fb_calibration_get_level1_sizes(fb_problem_size_t *sizes, size_t max_sizes);

/**
 * @brief Get problem sizes for Level 2 operations (matrix-vector)
 * 
 * @param sizes Output array (allocated by caller)
 * @param max_sizes Maximum number of sizes to return
 * @return Number of sizes returned
 */
size_t fb_calibration_get_level2_sizes(fb_problem_size_t *sizes, size_t max_sizes);

/**
 * @brief Get problem sizes for Level 3 operations (matrix-matrix)
 * 
 * Tests a range from small (64x64) to large (8192x8192) matrices.
 * 
 * @param sizes Output array (allocated by caller)
 * @param max_sizes Maximum number of sizes to return
 * @return Number of sizes returned
 */
size_t fb_calibration_get_level3_sizes(fb_problem_size_t *sizes, size_t max_sizes);

/* ============================================================================
 * Calibration Analysis
 * ========================================================================= */

/**
 * @brief Find the best backend for a specific problem
 * 
 * @param db Calibration database
 * @param operation_name Operation name
 * @param problem Problem specification
 * @return Backend ID of best performer, or -1 if not found
 */
int fb_calibration_find_best_backend(
    const fb_calibration_database_t *db,
    const char *operation_name,
    const fb_problem_size_t *problem
);

/**
 * @brief Get speedup factor for a backend vs reference
 * 
 * @param db Calibration database
 * @param operation_name Operation name
 * @param problem Problem specification
 * @param backend_id Backend to check
 * @return Speedup factor (>1.0 = faster than reference), or 0.0 if not found
 */
double fb_calibration_get_speedup(
    const fb_calibration_database_t *db,
    const char *operation_name,
    const fb_problem_size_t *problem,
    int backend_id
);

/**
 * @brief Generate a dispatch table from calibration data
 * 
 * Creates a lookup table that maps (operation, problem_size) to optimal backend.
 * This is used by the dispatch system for zero-overhead function calls.
 * 
 * @param db Calibration database
 * @param table Output dispatch table
 * @return 0 on success, negative error code on failure
 */
int fb_calibration_generate_dispatch_table(
    const fb_calibration_database_t *db,
    void **table  /* Implementation-defined dispatch table structure */
);

/**
 * @brief Print calibration summary
 * 
 * Outputs human-readable summary of calibration results to stdout.
 * 
 * @param db Calibration database
 */
void fb_calibration_print_summary(const fb_calibration_database_t *db);

/**
 * @brief Export calibration data to JSON
 * 
 * @param db Database to export
 * @param output_path Output file path
 * @return 0 on success, negative error code on failure
 */
int fb_calibration_export_json(const fb_calibration_database_t *db, const char *output_path);

/**
 * @brief Import calibration data from JSON
 * 
 * @param input_path Input file path
 * @param db Output database
 * @return 0 on success, negative error code on failure
 */
int fb_calibration_import_json(const char *input_path, fb_calibration_database_t **db);

/**
 * @brief Cleanup calibration system
 */
void fb_calibration_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* FB_CALIBRATION_H */
