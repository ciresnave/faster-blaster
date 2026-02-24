/**
 * @file dispatch_strategy.h
 * @brief Advanced dispatch strategies and device selection algorithms
 * 
 * This module implements sophisticated device selection algorithms that
 * consider multiple factors simultaneously: performance, load, data locality,
 * power consumption, and thermal state. It provides the intelligence layer
 * for the compute manager.
 */

#ifndef FASTER_BLASTER_DISPATCH_STRATEGY_H
#define FASTER_BLASTER_DISPATCH_STRATEGY_H

#include "compute_device.h"
#include "compute_manager.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Strategy Algorithms
 * ========================================================================== */

/**
 * @brief Select device using FASTEST strategy
 * Always chooses the device with highest peak performance, ignoring current load.
 * 
 * @param devices Array of available devices
 * @param device_count Number of devices
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Device index with best performance
 */
int fb_strategy_fastest(const fb_compute_device_t** devices,
                        int device_count,
                        const char* op_name,
                        int m, int n, int k);

/**
 * @brief Select device using LOAD_BALANCED strategy
 * Balances load across all devices, preferring less-utilized devices.
 * 
 * @param devices Array of available devices
 * @param device_count Number of devices
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Device index with best load balance
 */
int fb_strategy_load_balanced(const fb_compute_device_t** devices,
                               int device_count,
                               const char* op_name,
                               int m, int n, int k);

/**
 * @brief Select device using POWER_EFFICIENT strategy
 * Minimizes power consumption and energy usage.
 * 
 * @param devices Array of available devices
 * @param device_count Number of devices
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Device index with best power efficiency
 */
int fb_strategy_power_efficient(const fb_compute_device_t** devices,
                                 int device_count,
                                 const char* op_name,
                                 int m, int n, int k);

/**
 * @brief Select device using DATA_LOCALITY strategy
 * Minimizes data movement by preferring device where data already resides.
 * 
 * @param devices Array of available devices
 * @param device_count Number of devices
 * @param op_name Operation name
 * @param data_location Data location hint
 * @return Device index minimizing data transfers
 */
int fb_strategy_data_locality(const fb_compute_device_t** devices,
                               int device_count,
                               const char* op_name,
                               const fb_data_location_t* data_location);

/**
 * @brief Select device using ADAPTIVE strategy
 * Learns optimal device selection from execution history using machine learning.
 * 
 * @param devices Array of available devices
 * @param device_count Number of devices
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param history Execution history (opaque)
 * @return Device index based on learned patterns
 */
int fb_strategy_adaptive(const fb_compute_device_t** devices,
                         int device_count,
                         const char* op_name,
                         int m, int n, int k,
                         void* history);

/* ============================================================================
 * Multi-Factor Scoring
 * ========================================================================== */

/**
 * @brief Multi-factor device scoring weights
 */
typedef struct {
    double speed_weight;            /**< Performance weight (0-1) */
    double load_weight;             /**< Load balance weight (0-1) */
    double locality_weight;         /**< Data locality weight (0-1) */
    double power_weight;            /**< Power efficiency weight (0-1) */
    double thermal_weight;          /**< Thermal headroom weight (0-1) */
} fb_scoring_weights_t;

/**
 * @brief Compute multi-factor score for device
 * 
 * @param device Device to score
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param weights Scoring weights
 * @param data_location Data location hint (optional)
 * @param score Output: detailed score breakdown
 * @return Overall score (0-1, higher is better)
 */
double fb_strategy_compute_score(const fb_compute_device_t* device,
                                  const char* op_name,
                                  int m, int n, int k,
                                  const fb_scoring_weights_t* weights,
                                  const fb_data_location_t* data_location,
                                  fb_device_score_t* score);

/**
 * @brief Select device using multi-factor scoring
 * 
 * @param devices Array of available devices
 * @param device_count Number of devices
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param weights Scoring weights
 * @param data_location Data location hint (optional)
 * @param scores Output: scores for all devices (optional)
 * @return Device index with highest score
 */
int fb_strategy_multi_factor(const fb_compute_device_t** devices,
                              int device_count,
                              const char* op_name,
                              int m, int n, int k,
                              const fb_scoring_weights_t* weights,
                              const fb_data_location_t* data_location,
                              fb_device_score_t* scores);

/* ============================================================================
 * Workload-Aware Selection
 * ========================================================================== */

/**
 * @brief Determine optimal device type for workload size
 * 
 * Small workloads often execute faster on CPU due to GPU kernel launch overhead.
 * Large workloads benefit from GPU parallelism.
 * 
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Recommended device type (FB_DEVICE_TYPE_CPU or FB_DEVICE_TYPE_GPU)
 */
fb_device_type_t fb_strategy_recommend_device_type(const char* op_name,
                                                    int m, int n, int k);

/**
 * @brief Estimate GPU kernel launch overhead
 * 
 * @param device GPU device
 * @param op_name Operation name
 * @return Estimated launch overhead (microseconds)
 */
double fb_strategy_estimate_launch_overhead(const fb_compute_device_t* device,
                                            const char* op_name);

/**
 * @brief Calculate problem size (number of operations)
 * 
 * @param op_name Operation name (e.g., "gemm", "getrf")
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Estimated FLOPs
 */
uint64_t fb_strategy_calculate_flops(const char* op_name,
                                     int m, int n, int k);

/**
 * @brief Estimate data transfer time
 * 
 * @param device Device to transfer to
 * @param data_size Total data size (bytes)
 * @param is_pinned Whether memory is pinned
 * @return Estimated transfer time (microseconds)
 */
double fb_strategy_estimate_transfer_time(const fb_compute_device_t* device,
                                          size_t data_size,
                                          bool is_pinned);

/* ============================================================================
 * Power and Thermal Management
 * ========================================================================== */

/**
 * @brief Compute power efficiency score for device
 * 
 * @param device Device to score
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Power efficiency score (0-1, higher is more efficient)
 */
double fb_strategy_power_score(const fb_compute_device_t* device,
                                const char* op_name,
                                int m, int n, int k);

/**
 * @brief Compute thermal headroom score
 * 
 * Devices near thermal limits should be avoided to prevent throttling.
 * 
 * @param device Device to score
 * @return Thermal headroom score (0-1, higher is cooler)
 */
double fb_strategy_thermal_score(const fb_compute_device_t* device);

/**
 * @brief Check if device should be throttled due to thermal/power limits
 * 
 * @param device Device to check
 * @return true if device should be avoided
 */
bool fb_strategy_should_throttle(const fb_compute_device_t* device);

/**
 * @brief Estimate energy consumption for operation
 * 
 * @param device Device to execute on
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Estimated energy (joules)
 */
double fb_strategy_estimate_energy(const fb_compute_device_t* device,
                                   const char* op_name,
                                   int m, int n, int k);

/* ============================================================================
 * Performance Modeling
 * ========================================================================== */

/**
 * @brief Operation performance model
 */
typedef struct {
    double base_latency_us;         /**< Base latency (kernel launch, etc.) */
    double flops_per_us;            /**< Sustained FLOPS per microsecond */
    double memory_bound_factor;     /**< Memory bandwidth limitation (0-1) */
    double efficiency;              /**< Efficiency factor (0-1) */
} fb_perf_model_t;

/**
 * @brief Get performance model for operation on device
 * 
 * @param device Device to model
 * @param op_name Operation name
 * @param model Output: performance model
 * @return 0 on success, non-zero if no model available
 */
int fb_strategy_get_perf_model(const fb_compute_device_t* device,
                                const char* op_name,
                                fb_perf_model_t* model);

/**
 * @brief Estimate execution time using performance model
 * 
 * @param device Device to execute on
 * @param model Performance model
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Estimated execution time (microseconds)
 */
double fb_strategy_estimate_time(const fb_compute_device_t* device,
                                  const fb_perf_model_t* model,
                                  int m, int n, int k);

/* ============================================================================
 * Calibration and Benchmarking
 * ========================================================================== */

/**
 * @brief Benchmark operation on device
 * 
 * @param device Device to benchmark
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param iterations Number of iterations to run
 * @return Average execution time (microseconds), or -1 on error
 */
double fb_strategy_benchmark(fb_compute_device_t* device,
                             const char* op_name,
                             int m, int n, int k,
                             int iterations);

/**
 * @brief Calibrate device for all common operations
 * 
 * Runs benchmarks for typical problem sizes and builds performance models.
 * 
 * @param device Device to calibrate
 * @param quick_mode If true, only benchmark small subset
 * @return 0 on success, non-zero on error
 */
int fb_strategy_calibrate_device(fb_compute_device_t* device, bool quick_mode);

/**
 * @brief Save calibration data to file
 * 
 * @param device Device to save
 * @param filename Output filename
 * @return 0 on success, non-zero on error
 */
int fb_strategy_save_calibration(const fb_compute_device_t* device,
                                  const char* filename);

/**
 * @brief Load calibration data from file
 * 
 * @param device Device to load into
 * @param filename Input filename
 * @return 0 on success, non-zero on error
 */
int fb_strategy_load_calibration(fb_compute_device_t* device,
                                  const char* filename);

/* ============================================================================
 * Decision Explanation (for debugging)
 * ========================================================================== */

/**
 * @brief Device selection explanation
 */
typedef struct {
    int selected_device;            /**< Device ID that was selected */
    char reason[256];               /**< Human-readable explanation */
    fb_device_score_t selected_score; /**< Score of selected device */
    fb_device_score_t runner_up_score; /**< Score of second-best device */
    int runner_up_device;           /**< Second-best device ID */
} fb_selection_explanation_t;

/**
 * @brief Select device and explain decision
 * 
 * @param devices Array of available devices
 * @param device_count Number of devices
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @param policy Scheduling policy
 * @param weights Scoring weights (for multi-factor)
 * @param explanation Output: decision explanation
 * @return Selected device index
 */
int fb_strategy_select_with_explanation(const fb_compute_device_t** devices,
                                         int device_count,
                                         const char* op_name,
                                         int m, int n, int k,
                                         fb_scheduling_policy_t policy,
                                         const fb_scoring_weights_t* weights,
                                         fb_selection_explanation_t* explanation);

/**
 * @brief Print selection explanation
 * 
 * @param explanation Explanation to print
 */
void fb_strategy_print_explanation(const fb_selection_explanation_t* explanation);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_DISPATCH_STRATEGY_H */
