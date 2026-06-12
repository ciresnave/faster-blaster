/**
 * @file power_manager.h
 * @brief Power and thermal management for device scheduling
 * 
 * This module monitors device power consumption and thermal state,
 * enabling power-aware and battery-aware scheduling decisions.
 * 
 * Key features:
 * - Monitor device power consumption
 * - Track thermal state (temperature, throttling)
 * - Detect battery vs AC power
 * - Power-aware device selection
 */

#ifndef FASTER_BLASTER_POWER_MANAGER_H
#define FASTER_BLASTER_POWER_MANAGER_H

#include "compute_device.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Power State Detection
 * ========================================================================== */

/**
 * @brief System power source
 */
typedef enum {
    FB_POWER_SOURCE_UNKNOWN = 0,
    FB_POWER_SOURCE_AC,             /**< Plugged into AC power */
    FB_POWER_SOURCE_BATTERY,        /**< Running on battery */
    FB_POWER_SOURCE_UPS             /**< Running on UPS */
} fb_power_source_t;

/**
 * @brief Initialize power manager
 * @return 0 on success, non-zero on error
 */
int fb_power_init(void);

/**
 * @brief Shutdown power manager
 */
void fb_power_shutdown(void);

/**
 * @brief Get system power source
 * @return Current power source
 */
fb_power_source_t fb_power_get_source(void);

/**
 * @brief Get battery level (if on battery)
 * @return Battery percentage (0-100), or -1 if not on battery
 */
int fb_power_get_battery_level(void);

/**
 * @brief Check if system is in power-saving mode
 * @return true if power-saving mode is active
 */
bool fb_power_is_power_saving_mode(void);

/* ============================================================================
 * Device Power Monitoring
 * ========================================================================== */

/**
 * @brief Device power state
 */
typedef struct {
    double current_power_watts;     /**< Current power consumption (W) */
    double average_power_watts;     /**< Average power over 1 second (W) */
    double peak_power_watts;        /**< Peak power capability (W) */
    double power_limit_watts;       /**< Current power limit (W) */
    bool is_throttled;              /**< Power throttling active */
    double throttle_reason_percent; /**< Throttling severity (0-100%) */
} fb_device_power_state_t;

/**
 * @brief Get device power state
 * @param device_id Device ID
 * @param state Output: power state
 * @return 0 on success, non-zero on error
 */
int fb_power_get_device_state(int device_id, fb_device_power_state_t* state);

/**
 * @brief Estimate power consumption for operation
 * @param device_id Device ID
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Estimated power (watts), or -1 on error
 */
double fb_power_estimate_operation(int device_id, const char* op_name,
                                   int m, int n, int k);

/**
 * @brief Estimate energy consumption for operation
 * @param device_id Device ID
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Estimated energy (joules), or -1 on error
 */
double fb_power_estimate_energy(int device_id, const char* op_name,
                                int m, int n, int k);

/* ============================================================================
 * Thermal Monitoring
 * ========================================================================== */

/**
 * @brief Device thermal state
 */
typedef struct {
    double temperature_celsius;     /**< Current temperature (°C) */
    double max_safe_temp;           /**< Maximum safe temperature (°C) */
    double throttle_temp;           /**< Throttling threshold (°C) */
    bool is_thermal_throttling;     /**< Thermal throttling active */
    int fan_speed_percent;          /**< Fan speed (0-100%), -1 if N/A */
} fb_thermal_state_t;

/**
 * @brief Get device thermal state
 * @param device_id Device ID
 * @param state Output: thermal state
 * @return 0 on success, non-zero on error
 */
int fb_power_get_thermal_state(int device_id, fb_thermal_state_t* state);

/**
 * @brief Check if device should be avoided due to thermal state
 * @param device_id Device ID
 * @return true if device is too hot
 */
bool fb_power_is_thermally_limited(int device_id);

/* ============================================================================
 * Power-Aware Scheduling
 * ========================================================================== */

/**
 * @brief Power efficiency score for device
 * 
 * Computes score based on performance-per-watt and current power state.
 * 
 * @param device_id Device ID
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Efficiency score (0-1, higher = more efficient)
 */
double fb_power_efficiency_score(int device_id, const char* op_name,
                                 int m, int n, int k);

/**
 * @brief Select most power-efficient device
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Most efficient device ID, or -1 on error
 */
int fb_power_select_efficient_device(const char* op_name,
                                     int m, int n, int k);

/**
 * @brief Recommend device considering battery state
 * 
 * If on battery: strongly prefer CPU to save battery
 * If on AC: balance performance and power
 * 
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Recommended device ID
 */
int fb_power_battery_aware_select(const char* op_name,
                                  int m, int n, int k);

/* ============================================================================
 * Power Budgeting
 * ========================================================================== */

/**
 * @brief Set power budget for operations
 * @param max_watts Maximum power budget (watts)
 */
void fb_power_set_budget(double max_watts);

/**
 * @brief Get current power budget
 * @return Power budget (watts)
 */
double fb_power_get_budget(void);

/**
 * @brief Check if operation fits within power budget
 * @param device_id Device ID
 * @param op_name Operation name
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return true if operation is within budget
 */
bool fb_power_check_budget(int device_id, const char* op_name,
                           int m, int n, int k);

/* ============================================================================
 * Statistics
 * ========================================================================== */

/**
 * @brief Power statistics
 */
typedef struct {
    double total_energy_joules;     /**< Total energy consumed (J) */
    double average_power_watts;     /**< Average power consumption (W) */
    double peak_power_watts;        /**< Peak power consumption (W) */
    uint64_t total_operations;      /**< Total operations executed */
    uint64_t throttled_operations;  /**< Operations affected by throttling */
} fb_power_stats_t;

/**
 * @brief Get power statistics
 * @param stats Output: statistics
 */
void fb_power_get_stats(fb_power_stats_t* stats);

/**
 * @brief Reset power statistics
 */
void fb_power_reset_stats(void);

/**
 * @brief Print power status
 */
void fb_power_print_status(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_POWER_MANAGER_H */
