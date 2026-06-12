/**
 * @file data_tracker.h
 * @brief Data location tracking and movement optimization
 * 
 * This module tracks where data resides (CPU vs GPU, which GPU) and
 * optimizes dispatch decisions to minimize expensive data transfers.
 * 
 * Key features:
 * - Track data location across all devices
 * - Estimate transfer costs
 * - Suggest optimal device based on data locality
 * - Auto-register allocations
 */

#ifndef FASTER_BLASTER_DATA_TRACKER_H
#define FASTER_BLASTER_DATA_TRACKER_H

#include "compute_device.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Data Location Types
 * ========================================================================== */

/**
 * @brief Where a data buffer lives.
 */
typedef enum {
    FB_DATA_HOST    = 0,   /**< CPU-accessible host / system memory */
    FB_DATA_DEVICE  = 1,   /**< Device-local memory (GPU VRAM, etc.)  */
    FB_DATA_UNIFIED = 2,   /**< Unified / managed memory (CUDA UVM, etc.) */
    FB_DATA_UNKNOWN = 3,   /**< Location not yet tracked */
} fb_data_location_type_t;

/**
 * @brief Data location descriptor: memory type + device index.
 */
typedef struct {
    fb_data_location_type_t type;  /**< Memory class: host / device / unified */
    int device_id;                 /**< Device index when type == FB_DATA_DEVICE */
} fb_data_location_t;

/* Convenience aliases so data_tracker.c can compare against device->type
 * (fb_compute_device_t uses fb_device_type_t from compute_device.h).  */
#ifndef FB_DEVICE_CPU
#define FB_DEVICE_CPU  FB_DEVICE_TYPE_CPU
#define FB_DEVICE_GPU  FB_DEVICE_TYPE_GPU
#endif

/* ============================================================================
 * Data Tracking
 * ========================================================================== */

/**
 * @brief Data allocation info
 */
typedef struct {
    const void* ptr;                /**< Data pointer */
    size_t size;                    /**< Size in bytes */
    int device_id;                  /**< Device where data resides */
    bool is_pinned;                 /**< Host memory is pinned (fast transfer) */
    bool is_managed;                /**< Unified memory (CUDA managed, etc.) */
    uint64_t last_access_time;      /**< Last access timestamp */
    int access_count;               /**< Number of accesses */
    bool is_temporary;              /**< Temporary allocation (can be freed) */
} fb_data_info_t;

/**
 * @brief Initialize data tracker
 * @return 0 on success, non-zero on error
 */
int fb_data_tracker_init(void);

/**
 * @brief Shutdown data tracker
 */
void fb_data_tracker_shutdown(void);

/**
 * @brief Register data allocation
 * @param ptr Data pointer
 * @param size Data size in bytes
 * @param device_id Device where data resides
 * @param is_pinned Whether memory is pinned
 * @param is_managed Whether memory is managed/unified
 */
void fb_data_register(const void* ptr, size_t size, int device_id,
                     bool is_pinned, bool is_managed);

/**
 * @brief Unregister data allocation
 * @param ptr Data pointer
 */
void fb_data_unregister(const void* ptr);

/**
 * @brief Query data location
 * @param ptr Data pointer
 * @param info Output: data info (optional)
 * @return Device ID, or -1 if unknown
 */
int fb_data_query_location(const void* ptr, fb_data_info_t* info);

/**
 * @brief Update last access time
 * @param ptr Data pointer
 * @param device_id Device that accessed data
 */
void fb_data_record_access(const void* ptr, int device_id);

/* ============================================================================
 * Transfer Cost Estimation
 * ========================================================================== */

/**
 * @brief Estimate cost to transfer data to device
 * @param ptr Data pointer
 * @param target_device Target device ID
 * @return Estimated transfer time (microseconds), or 0 if already on device
 */
double fb_data_estimate_transfer_cost(const void* ptr, int target_device);

/**
 * @brief Estimate total transfer cost for operation
 * @param input_ptrs Array of input data pointers
 * @param input_count Number of inputs
 * @param output_ptrs Array of output data pointers
 * @param output_count Number of outputs
 * @param target_device Target device
 * @return Total estimated transfer time (microseconds)
 */
double fb_data_estimate_operation_cost(const void** input_ptrs, int input_count,
                                       const void** output_ptrs, int output_count,
                                       int target_device);

/* ============================================================================
 * Device Recommendation
 * ========================================================================== */

/**
 * @brief Recommend device based on data locality
 * 
 * Analyzes where input/output data is located and recommends device
 * that minimizes total data movement.
 * 
 * @param input_ptrs Array of input data pointers
 * @param input_count Number of inputs
 * @param output_ptrs Array of output data pointers
 * @param output_count Number of outputs
 * @return Recommended device ID
 */
int fb_data_recommend_device(const void** input_ptrs, int input_count,
                             const void** output_ptrs, int output_count);

/**
 * @brief Score device for data locality
 * @param input_ptrs Array of input data pointers
 * @param input_count Number of inputs
 * @param output_ptrs Array of output data pointers
 * @param output_count Number of outputs
 * @param device_id Device to score
 * @return Locality score (0-1, higher = better locality)
 */
double fb_data_locality_score(const void** input_ptrs, int input_count,
                               const void** output_ptrs, int output_count,
                               int device_id);

/* ============================================================================
 * Prefetching and Migration
 * ========================================================================== */

/**
 * @brief Prefetch data to device
 * @param ptr Data pointer
 * @param target_device Target device
 * @return 0 on success, non-zero on error
 */
int fb_data_prefetch(const void* ptr, int target_device);

/**
 * @brief Suggest data migration
 * 
 * Analyzes access patterns and suggests which data should be migrated
 * to which device for optimal performance.
 * 
 * @param suggestions Output: array of (ptr, target_device) pairs
 * @param max_suggestions Maximum suggestions to return
 * @return Number of suggestions
 */
int fb_data_suggest_migrations(void** suggestions, int max_suggestions);

/* ============================================================================
 * Statistics
 * ========================================================================== */

/**
 * @brief Data tracker statistics
 */
typedef struct {
    int total_allocations;          /**< Total tracked allocations */
    size_t total_bytes_tracked;     /**< Total bytes tracked */
    int cpu_allocations;            /**< Allocations on CPU */
    int gpu_allocations;            /**< Allocations on GPU */
    uint64_t total_transfers;       /**< Total data transfers */
    size_t total_bytes_transferred; /**< Total bytes transferred */
} fb_data_stats_t;

/**
 * @brief Get data tracker statistics
 * @param stats Output: statistics
 */
void fb_data_get_stats(fb_data_stats_t* stats);

/**
 * @brief Print data tracker status
 */
void fb_data_print_status(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_DATA_TRACKER_H */
