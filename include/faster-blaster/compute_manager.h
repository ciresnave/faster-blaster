/**
 * @file compute_manager.h
 * @brief Smart compute manager - load balancing and intelligent device scheduling
 * 
 * The compute manager is the brain of the hybrid dispatch system. It tracks
 * device load in real-time, maintains performance cost models, and makes
 * intelligent decisions about which device should execute each operation.
 * 
 * Key features:
 * - Real-time load balancing across all devices
 * - Multi-factor device selection (speed + load + data locality + power)
 * - Adaptive learning from execution history
 * - Workload-aware scheduling
 * - Power and thermal management
 */

#ifndef FASTER_BLASTER_COMPUTE_MANAGER_H
#define FASTER_BLASTER_COMPUTE_MANAGER_H

#include "compute_device.h"
#include "device_registry.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Type Definitions
 * ========================================================================== */

/**
 * @brief Dispatch strategy (alias for scheduling policy)
 */
typedef enum {
    FB_DISPATCH_FASTEST = 0,
    FB_DISPATCH_LOAD_BALANCED = 1,
    FB_DISPATCH_POWER_EFFICIENT = 2,
    FB_DISPATCH_DATA_LOCALITY = 3,
    FB_DISPATCH_ROUND_ROBIN = 4,
    FB_DISPATCH_ADAPTIVE = 5,
    FB_DISPATCH_CUSTOM = 6
} fb_dispatch_strategy_t;

/**
 * @brief Precision type
 */
typedef enum {
    FB_PRECISION_FP32,
    FB_PRECISION_FP64,
    FB_PRECISION_FP16,
    FB_PRECISION_BF16
} fb_precision_t;

/* ============================================================================
 * Global Initialization (Simplified API)
 * ========================================================================== */

/**
 * @brief Initialize global compute manager with default settings
 * @return 0 on success, -1 on error
 */
int fb_compute_manager_init(void);

/**
 * @brief Shutdown global compute manager
 */
void fb_compute_manager_shutdown(void);

/**
 * @brief Set global dispatch strategy
 * @param strategy Dispatch strategy to use
 */
void fb_set_dispatch_strategy(fb_dispatch_strategy_t strategy);

/**
 * @brief Get current global dispatch strategy
 * @return Current strategy
 */
fb_dispatch_strategy_t fb_get_dispatch_strategy(void);

/**
 * @brief Select device based on strategy, precision, and data locations
 * @param strategy Dispatch strategy to use
 * @param precision Operation precision
 * @param data_ptrs Array of data pointers (for locality analysis)
 * @param num_data_ptrs Number of data pointers
 * @return Selected device, or NULL if none available
 */
fb_compute_device_t* fb_select_device(fb_dispatch_strategy_t strategy,
                                      fb_precision_t precision,
                                      const void** data_ptrs,
                                      uint32_t num_data_ptrs);

/* ============================================================================
 * Scheduling Policies
 * ========================================================================== */

/**
 * @brief Device selection policy
 */
typedef enum {
    FB_POLICY_FASTEST,              /**< Always use fastest device (ignore load) */
    FB_POLICY_LOAD_BALANCED,        /**< Balance load across all devices */
    FB_POLICY_POWER_EFFICIENT,      /**< Minimize power consumption */
    FB_POLICY_DATA_LOCALITY,        /**< Minimize data movement */
    FB_POLICY_ROUND_ROBIN,          /**< Simple round-robin across devices */
    FB_POLICY_ADAPTIVE,             /**< Learn optimal strategy from history */
    FB_POLICY_CUSTOM                /**< User-defined scoring function */
} fb_scheduling_policy_t;

/**
 * @brief Workload characteristics hint
 */
typedef enum {
    FB_WORKLOAD_SMALL,              /**< Small problem (< 1K elements) */
    FB_WORKLOAD_MEDIUM,             /**< Medium problem (1K - 1M elements) */
    FB_WORKLOAD_LARGE,              /**< Large problem (> 1M elements) */
    FB_WORKLOAD_STREAMING,          /**< Streaming/batched operations */
    FB_WORKLOAD_LATENCY_CRITICAL,   /**< Minimize latency (start ASAP) */
    FB_WORKLOAD_THROUGHPUT          /**< Maximize throughput (may queue) */
} fb_workload_hint_t;

/**
 * @brief Data location hint
 */
typedef struct {
    int device_id;                  /**< Device where data currently resides */
    bool is_pinned;                 /**< Memory is pinned/registered */
    size_t size_bytes;              /**< Total data size */
    bool will_reuse;                /**< Data will be used in subsequent ops */
} fb_data_location_t;

/* ============================================================================
 * Cost Model
 * ========================================================================== */

/**
 * @brief Operation cost estimate
 */
typedef struct {
    double compute_time_us;         /**< Estimated computation time (μs) */
    double transfer_time_us;        /**< Estimated data transfer time (μs) */
    double queue_time_us;           /**< Estimated queue wait time (μs) */
    double total_time_us;           /**< Total estimated time (μs) */
    double power_watts;             /**< Estimated power consumption (W) */
    double energy_joules;           /**< Estimated energy (J = W × s) */
} fb_cost_estimate_t;

/**
 * @brief Device scoring factors
 */
typedef struct {
    double speed_score;             /**< Raw performance score (0-1) */
    double load_score;              /**< Current load score (0-1, higher=less loaded) */
    double locality_score;          /**< Data locality score (0-1) */
    double power_score;             /**< Power efficiency score (0-1) */
    double thermal_score;           /**< Thermal headroom score (0-1) */
    double total_score;             /**< Weighted total score */
} fb_device_score_t;

/**
 * @brief Custom scoring function
 * @param device Device to score
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param user_data User-provided data
 * @return Device score (higher = better)
 */
typedef double (*fb_custom_scorer_t)(const fb_compute_device_t* device,
                                     const char* op_name,
                                     int m, int n, int k,
                                     void* user_data);

/* ============================================================================
 * Compute Manager
 * ========================================================================== */

/**
 * @brief Compute manager handle (opaque)
 */
typedef struct fb_compute_manager fb_compute_manager_t;

/**
 * @brief Manager configuration
 */
typedef struct {
    fb_scheduling_policy_t default_policy;  /**< Default scheduling policy */
    
    /* Scoring weights (for multi-factor policies) */
    double speed_weight;            /**< Speed importance (default: 0.4) */
    double load_weight;             /**< Load balance importance (default: 0.3) */
    double locality_weight;         /**< Data locality importance (default: 0.2) */
    double power_weight;            /**< Power efficiency importance (default: 0.1) */
    
    /* Thresholds */
    size_t small_problem_threshold; /**< Size below which CPU preferred (bytes) */
    size_t large_problem_threshold; /**< Size above which GPU preferred (bytes) */
    double load_imbalance_threshold;/**< Max load difference before rebalancing (0-1) */
    
    /* Adaptive learning */
    bool enable_learning;           /**< Learn from execution history */
    int learning_window_size;       /**< Number of recent ops to track */
    
    /* Power management */
    bool enable_power_management;   /**< Enable power-aware scheduling */
    bool prefer_cpu_on_battery;     /**< Prefer CPU when on battery power */
    double thermal_throttle_temp;   /**< Temperature to avoid (°C) */
    
    /* Custom scoring */
    fb_custom_scorer_t custom_scorer; /**< Custom scoring function (optional) */
    void* custom_scorer_data;       /**< User data for custom scorer */
} fb_manager_config_t;

/* ============================================================================
 * Manager Lifecycle
 * ========================================================================== */

/**
 * @brief Create compute manager with default configuration
 * @return Manager handle, or NULL on error
 */
fb_compute_manager_t* fb_manager_create(void);

/**
 * @brief Create compute manager with custom configuration
 * @param config Manager configuration
 * @return Manager handle, or NULL on error
 */
fb_compute_manager_t* fb_manager_create_with_config(const fb_manager_config_t* config);

/**
 * @brief Destroy compute manager
 * @param manager Manager to destroy
 */
void fb_manager_destroy(fb_compute_manager_t* manager);

/**
 * @brief Get default configuration
 * @param config Output: default configuration
 */
void fb_manager_get_default_config(fb_manager_config_t* config);

/**
 * @brief Update manager configuration
 * @param manager Manager to update
 * @param config New configuration
 * @return 0 on success, non-zero on error
 */
int fb_manager_set_config(fb_compute_manager_t* manager, const fb_manager_config_t* config);

/* ============================================================================
 * Device Selection
 * ========================================================================== */

/**
 * @brief Select best device for operation
 * @param manager Compute manager
 * @param op_name Operation name (e.g., "sgemm")
 * @param m Matrix dimension M
 * @param n Matrix dimension N  
 * @param k Matrix dimension K
 * @param workload_hint Workload characteristics (optional)
 * @param data_location Data location hint (optional)
 * @return Best device ID, or -1 on error
 */
int fb_manager_select_device(fb_compute_manager_t* manager,
                              const char* op_name,
                              int m, int n, int k,
                              fb_workload_hint_t workload_hint,
                              const fb_data_location_t* data_location);

/**
 * @brief Select device with explicit policy override
 * @param manager Compute manager
 * @param policy Policy to use for this selection
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Selected device ID, or -1 on error
 */
int fb_manager_select_device_with_policy(fb_compute_manager_t* manager,
                                         fb_scheduling_policy_t policy,
                                         const char* op_name,
                                         int m, int n, int k);

/**
 * @brief Get device scores for all devices
 * @param manager Compute manager
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param scores Output array of device scores (size = device count)
 * @return Number of devices scored
 */
int fb_manager_score_all_devices(fb_compute_manager_t* manager,
                                  const char* op_name,
                                  int m, int n, int k,
                                  fb_device_score_t* scores);

/**
 * @brief Estimate operation cost on specific device
 * @param manager Compute manager
 * @param device_id Device to estimate for
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param estimate Output: cost estimate
 * @return 0 on success, non-zero on error
 */
int fb_manager_estimate_cost(fb_compute_manager_t* manager,
                              int device_id,
                              const char* op_name,
                              int m, int n, int k,
                              fb_cost_estimate_t* estimate);

/* ============================================================================
 * Load Tracking
 * ========================================================================== */

/**
 * @brief Notify manager that operation started on device
 * @param manager Compute manager
 * @param device_id Device ID
 * @param op_name Operation name
 * @param estimated_time_us Estimated execution time (μs)
 * @return Operation tracking ID
 */
int fb_manager_operation_started(fb_compute_manager_t* manager,
                                  int device_id,
                                  const char* op_name,
                                  double estimated_time_us);

/**
 * @brief Notify manager that operation completed
 * @param manager Compute manager
 * @param op_id Operation tracking ID
 * @param actual_time_us Actual execution time (μs)
 */
void fb_manager_operation_completed(fb_compute_manager_t* manager,
                                    int op_id,
                                    double actual_time_us);

/**
 * @brief Get current load for device
 * @param manager Compute manager
 * @param device_id Device ID
 * @return Current load (0-1), or -1 on error
 */
double fb_manager_get_device_load(fb_compute_manager_t* manager, int device_id);

/**
 * @brief Get number of queued operations on device
 * @param manager Compute manager
 * @param device_id Device ID
 * @return Number of queued operations
 */
int fb_manager_get_queue_depth(fb_compute_manager_t* manager, int device_id);

/* ============================================================================
 * Data Locality Tracking
 * ========================================================================== */

/**
 * @brief Register data location
 * @param manager Compute manager
 * @param data_ptr Data pointer
 * @param device_id Device where data resides
 * @param size_bytes Data size
 */
void fb_manager_register_data(fb_compute_manager_t* manager,
                               const void* data_ptr,
                               int device_id,
                               size_t size_bytes);

/**
 * @brief Unregister data location
 * @param manager Compute manager
 * @param data_ptr Data pointer to unregister
 */
void fb_manager_unregister_data(fb_compute_manager_t* manager,
                                const void* data_ptr);

/**
 * @brief Query data location
 * @param manager Compute manager
 * @param data_ptr Data pointer to query
 * @return Device ID where data resides, or -1 if unknown
 */
int fb_manager_query_data_location(fb_compute_manager_t* manager,
                                   const void* data_ptr);

/* ============================================================================
 * Adaptive Learning
 * ========================================================================== */

/**
 * @brief Record execution result for learning
 * @param manager Compute manager
 * @param device_id Device that executed operation
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param actual_time_us Actual execution time (μs)
 */
void fb_manager_record_execution(fb_compute_manager_t* manager,
                                  int device_id,
                                  const char* op_name,
                                  int m, int n, int k,
                                  double actual_time_us);

/**
 * @brief Clear learning history
 * @param manager Compute manager
 */
void fb_manager_clear_history(fb_compute_manager_t* manager);

/**
 * @brief Export learning data to file
 * @param manager Compute manager
 * @param filename Output filename
 * @return 0 on success, non-zero on error
 */
int fb_manager_export_history(fb_compute_manager_t* manager, const char* filename);

/**
 * @brief Import learning data from file
 * @param manager Compute manager
 * @param filename Input filename
 * @return 0 on success, non-zero on error
 */
int fb_manager_import_history(fb_compute_manager_t* manager, const char* filename);

/* ============================================================================
 * Statistics and Monitoring
 * ========================================================================== */

/**
 * @brief Manager statistics
 */
typedef struct {
    uint64_t total_operations;      /**< Total operations dispatched */
    uint64_t cpu_operations;        /**< Operations on CPU */
    uint64_t gpu_operations;        /**< Operations on GPU */
    double total_compute_time_s;    /**< Total compute time (seconds) */
    double total_transfer_time_s;   /**< Total transfer time (seconds) */
    double average_utilization;     /**< Average device utilization (0-1) */
    int policy_changes;             /**< Number of policy changes */
    int load_rebalances;            /**< Number of load rebalances */
} fb_manager_stats_t;

/**
 * @brief Get manager statistics
 * @param manager Compute manager
 * @param stats Output: statistics
 */
void fb_manager_get_stats(fb_compute_manager_t* manager, fb_manager_stats_t* stats);

/**
 * @brief Reset statistics counters
 * @param manager Compute manager
 */
void fb_manager_reset_stats(fb_compute_manager_t* manager);

/**
 * @brief Print manager status and statistics
 * @param manager Compute manager
 */
void fb_manager_print_status(fb_compute_manager_t* manager);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_COMPUTE_MANAGER_H */
