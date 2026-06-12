/**
 * @file timing.h
 * @brief High-resolution timing utilities
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_TIMING_H
#define FB_TIMING_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief High-resolution timer handle
 */
typedef struct {
    uint64_t start_time;
    uint64_t end_time;
    bool running;
} fb_timer_t;

/**
 * @brief Initialize timing system
 * 
 * Must be called once before using timers.
 * 
 * @return 0 on success, negative on error
 */
int fb_timing_init(void);

/**
 * @brief Get current time in nanoseconds
 * 
 * @return Current time in nanoseconds since arbitrary epoch
 */
uint64_t fb_timing_now_ns(void);

/**
 * @brief Start a timer
 * 
 * @param timer Timer to start
 */
void fb_timer_start(fb_timer_t *timer);

/**
 * @brief Stop a timer
 * 
 * @param timer Timer to stop
 */
void fb_timer_stop(fb_timer_t *timer);

/**
 * @brief Get elapsed time in nanoseconds
 * 
 * @param timer Timer to query
 * @return Elapsed time in nanoseconds
 */
uint64_t fb_timer_elapsed_ns(const fb_timer_t *timer);

/**
 * @brief Get elapsed time in microseconds
 * 
 * @param timer Timer to query
 * @return Elapsed time in microseconds
 */
double fb_timer_elapsed_us(const fb_timer_t *timer);

/**
 * @brief Get elapsed time in milliseconds
 * 
 * @param timer Timer to query
 * @return Elapsed time in milliseconds
 */
double fb_timer_elapsed_ms(const fb_timer_t *timer);

/**
 * @brief Reset a timer
 * 
 * @param timer Timer to reset
 */
void fb_timer_reset(fb_timer_t *timer);

/**
 * @brief Statistics for multiple timing measurements
 */
typedef struct {
    double *measurements;    /**< Array of timing measurements (ns) */
    size_t count;           /**< Number of measurements */
    size_t capacity;        /**< Allocated capacity */
    double min;             /**< Minimum time */
    double max;             /**< Maximum time */
    double median;          /**< Median time */
    double mean;            /**< Mean time */
    double stddev;          /**< Standard deviation */
} fb_timing_stats_t;

/**
 * @brief Create timing statistics tracker
 * 
 * @param capacity Initial capacity for measurements
 * @return Statistics tracker, or NULL on error
 */
fb_timing_stats_t* fb_timing_stats_create(size_t capacity);

/**
 * @brief Add a measurement to statistics
 * 
 * @param stats Statistics tracker
 * @param time_ns Time measurement in nanoseconds
 * @return 0 on success, negative on error
 */
int fb_timing_stats_add(fb_timing_stats_t *stats, double time_ns);

/**
 * @brief Calculate statistics from measurements
 * 
 * Computes min, max, median, mean, and standard deviation.
 * Must be called after adding all measurements.
 * 
 * @param stats Statistics tracker
 * @return 0 on success, negative on error
 */
int fb_timing_stats_compute(fb_timing_stats_t *stats);

/**
 * @brief Free timing statistics tracker
 * 
 * @param stats Statistics tracker to free
 */
void fb_timing_stats_free(fb_timing_stats_t *stats);

/**
 * @brief Sleep for specified nanoseconds
 * 
 * @param ns Nanoseconds to sleep
 */
void fb_timing_sleep_ns(uint64_t ns);

/**
 * @brief Calculate GFLOPS from operation count and time
 * 
 * @param flops Number of floating-point operations
 * @param time_ns Execution time in nanoseconds
 * @return GFLOPS (billions of operations per second)
 */
double fb_timing_calc_gflops(uint64_t flops, double time_ns);

/**
 * @brief Calculate memory bandwidth in GB/s
 * 
 * @param bytes Number of bytes transferred
 * @param time_ns Execution time in nanoseconds
 * @return Bandwidth in GB/s
 */
double fb_timing_calc_bandwidth(uint64_t bytes, double time_ns);

#ifdef __cplusplus
}
#endif

#endif /* FB_TIMING_H */
