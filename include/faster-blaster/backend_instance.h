/**
 * @file backend_instance.h
 * @brief Backend instance management - connects devices to loaded backends
 * 
 * This module creates the critical link between:
 * 1. Device detection (cpu_detect, gpu_detect)
 * 2. Device selection (compute_manager)
 * 3. Plugin system (plugin_registry)
 * 4. Backend execution (operation dispatch)
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_BACKEND_INSTANCE_H
#define FASTER_BLASTER_BACKEND_INSTANCE_H

#include "compute_device.h"
#include "backend_plugin.h"
#include "backends/backend_interface.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Backend instance - represents a loaded backend for a specific device
 */
typedef struct fb_backend_instance {
    fb_compute_device_t* device;              /**< Device this backend runs on */
    const fb_backend_plugin_t* plugin;        /**< Plugin providing the backend */
    fb_plugin_context_t* plugin_ctx;          /**< Plugin-specific context */
    const fb_backend_vtable_t* vtable;        /**< Backend operations table */
    bool initialized;                         /**< Is this instance ready? */
    uint64_t operation_count;                 /**< Total ops executed */
    double total_time_seconds;                /**< Total execution time */
} fb_backend_instance_t;

/**
 * Backend instance manager - maintains loaded backends for all devices
 */
typedef struct {
    fb_backend_instance_t* instances;         /**< Array of backend instances */
    uint32_t num_instances;                   /**< Number of instances */
    uint32_t capacity;                        /**< Array capacity */
    bool initialized;                         /**< Is manager initialized? */
} fb_backend_instance_manager_t;

/**
 * Initialize the backend instance manager
 * 
 * This scans all detected devices and loads appropriate backends for each.
 * Called automatically by compute_manager_init() if needed.
 * 
 * @return 0 on success, negative on error
 */
int fb_backend_instance_manager_init(void);

/**
 * Shutdown backend instance manager and unload all backends
 */
void fb_backend_instance_manager_shutdown(void);

/**
 * Get or load backend instance for a specific device
 * 
 * This is the main entry point for getting a backend to execute on.
 * If the device already has a loaded backend, returns it immediately.
 * Otherwise, selects and loads the best plugin for that device.
 * 
 * @param device Device to get backend for (from fb_select_device)
 * @return Backend instance, or NULL on error
 */
fb_backend_instance_t* fb_get_backend_for_device(fb_compute_device_t* device);

/**
 * Manually load a specific backend for a device
 * 
 * @param device Device to load backend for
 * @param plugin_name Specific plugin name (e.g., "cublas", "mkl"), NULL for auto
 * @return Backend instance, or NULL on error
 */
fb_backend_instance_t* fb_load_backend_for_device(fb_compute_device_t* device, 
                                                   const char* plugin_name);

/**
 * Unload backend instance for a device
 * 
 * @param device Device to unload backend for
 */
void fb_unload_backend_for_device(fb_compute_device_t* device);

/**
 * Get vtable from a backend instance (convenience function)
 * 
 * @param instance Backend instance
 * @return Vtable pointer, or NULL if instance invalid
 */
const fb_backend_vtable_t* fb_backend_get_vtable(fb_backend_instance_t* instance);

/**
 * Execute operation with fallback chain
 * 
 * Tries to execute an operation on the primary backend. If it fails,
 * tries secondary backends, and finally falls back to reference backend.
 * 
 * @param instance Primary backend instance
 * @param operation_name Name of operation (for error reporting)
 * @param exec_func Function that executes the operation, returns 0 on success
 * @param user_data User data passed to exec_func
 * @return 0 on success, negative on error
 */
typedef int (*fb_operation_exec_func_t)(const struct fb_backend_vtable* vtable, void* user_data);

int fb_execute_with_fallback(fb_backend_instance_t* instance,
                             const char* operation_name,
                             fb_operation_exec_func_t exec_func,
                             void* user_data);

/**
 * Get backend instance statistics
 * 
 * @param instance Backend instance
 * @param out_operation_count Output for operation count
 * @param out_total_time Output for total time in seconds
 * @return 0 on success, negative on error
 */
int fb_backend_get_stats(fb_backend_instance_t* instance,
                        uint64_t* out_operation_count,
                        double* out_total_time);

/**
 * Reset backend instance statistics
 * 
 * @param instance Backend instance
 */
void fb_backend_reset_stats(fb_backend_instance_t* instance);

/**
 * Print backend instance information (debugging)
 * 
 * @param instance Backend instance
 */
void fb_backend_print_instance(const fb_backend_instance_t* instance);

/**
 * List all loaded backend instances
 * 
 * @param instances Output array (can be NULL to query count)
 * @param max_count Size of output array
 * @return Number of instances
 */
int fb_backend_list_instances(fb_backend_instance_t** instances, int max_count);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_BACKEND_INSTANCE_H */
