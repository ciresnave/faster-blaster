/**
 * @file device_registry.h
 * @brief Device discovery, registration, and hotplug management
 * 
 * This module handles automatic discovery of all available compute devices,
 * maintains a registry of devices, and handles dynamic device changes
 * (hotplug, device removal, power state changes).
 */

#ifndef FASTER_BLASTER_DEVICE_REGISTRY_H
#define FASTER_BLASTER_DEVICE_REGISTRY_H

#include "compute_device.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Device Discovery
 * ========================================================================== */

/**
 * @brief Device discovery flags
 */
typedef enum {
    FB_DISCOVERY_CPU      = 1 << 0,  /**< Discover CPU devices */
    FB_DISCOVERY_GPU      = 1 << 1,  /**< Discover GPU devices */
    FB_DISCOVERY_ALL      = 0xFFFF,  /**< Discover all device types */
} fb_discovery_flags_t;

/**
 * @brief Device event type for callbacks
 */
typedef enum {
    FB_DEVICE_EVENT_ADDED,           /**< New device detected */
    FB_DEVICE_EVENT_REMOVED,         /**< Device removed/unavailable */
    FB_DEVICE_EVENT_STATE_CHANGED,   /**< Device state changed (power, thermal) */
    FB_DEVICE_EVENT_LOAD_CHANGED     /**< Device load significantly changed */
} fb_device_event_t;

/**
 * @brief Device event callback function
 * @param event Event type
 * @param device_id Device ID that triggered event
 * @param user_data User-provided data
 */
typedef void (*fb_device_event_callback_t)(fb_device_event_t event,
                                           int device_id,
                                           void* user_data);

/* ============================================================================
 * Registry Management
 * ========================================================================== */

/**
 * @brief Initialize device registry and discover all devices
 * @param flags Discovery flags (which device types to search for)
 * @return Number of devices discovered, or -1 on error
 */
int fb_registry_init(fb_discovery_flags_t flags);

/**
 * @brief Shutdown registry and release all resources
 */
void fb_registry_shutdown(void);

/**
 * @brief Rescan for new devices (hotplug detection)
 * @param flags Discovery flags
 * @return Number of new devices found
 */
int fb_registry_rescan(fb_discovery_flags_t flags);

/**
 * @brief Register event callback for device changes
 * @param callback Callback function
 * @param user_data User data passed to callback
 * @return Callback ID, or -1 on error
 */
int fb_registry_register_callback(fb_device_event_callback_t callback, void* user_data);

/**
 * @brief Unregister event callback
 * @param callback_id Callback ID from registration
 */
void fb_registry_unregister_callback(int callback_id);

/* ============================================================================
 * Device Querying
 * ========================================================================== */

/**
 * @brief Get total number of registered devices
 * @return Total device count
 */
int fb_registry_get_device_count(void);

/**
 * @brief Get device by global ID
 * @param device_id Global device ID (0 to count-1)
 * @return Device handle, or NULL if invalid
 */
fb_compute_device_t* fb_registry_get_device(int device_id);

/**
 * @brief Get device by index (simplified alias)
 * @param index Device index
 * @return Device handle, or NULL if invalid
 */
fb_compute_device_t* fb_get_device(uint32_t index);

/**
 * @brief Get total number of devices (simplified alias)
 * @return Total device count
 */
uint32_t fb_get_device_count(void);

/**
 * @brief Print all detected devices to stdout
 */
void fb_print_devices(void);

/**
 * @brief Get all devices of specific type
 * @param type Device type to query
 * @param devices Output array of device handles
 * @param max_devices Maximum devices to return
 * @return Number of devices returned
 */
int fb_registry_get_devices_by_type(fb_device_type_t type,
                                    fb_compute_device_t** devices,
                                    int max_devices);

/**
 * @brief Find devices matching specific criteria
 * @param min_memory Minimum memory required (bytes, 0 = any)
 * @param min_gflops Minimum GFLOPS required (0 = any)
 * @param required_caps Required capability flags (0 = any)
 * @param devices Output array of matching device handles
 * @param max_devices Maximum devices to return
 * @return Number of matching devices
 */
int fb_registry_find_devices(size_t min_memory,
                              double min_gflops,
                              uint32_t required_caps,
                              fb_compute_device_t** devices,
                              int max_devices);

/* ============================================================================
 * CPU Detection
 * ========================================================================== */

/**
 * @brief Detect all CPU devices/sockets
 * @return Number of CPU devices discovered
 */
int fb_registry_detect_cpus(void);

/**
 * @brief Get CPU vendor from CPUID
 * @return CPU vendor type
 */
fb_cpu_vendor_t fb_registry_get_cpu_vendor(void);

/**
 * @brief Get CPU feature flags (AVX, AVX2, AVX-512, etc.)
 * @param cpu_id CPU ID to query
 * @return Capability flags
 */
uint32_t fb_registry_get_cpu_features(int cpu_id);

/**
 * @brief Get CPU core topology information
 * @param cpu_id CPU ID to query
 * @param properties Output CPU properties
 * @return 0 on success, non-zero on error
 */
int fb_registry_get_cpu_topology(int cpu_id, fb_cpu_properties_t* properties);

/* ============================================================================
 * GPU Detection
 * ========================================================================== */

/**
 * @brief Detect all NVIDIA GPUs (CUDA)
 * @return Number of NVIDIA GPUs discovered
 */
int fb_registry_detect_nvidia_gpus(void);

/**
 * @brief Detect all AMD GPUs (ROCm/HIP)
 * @return Number of AMD GPUs discovered
 */
int fb_registry_detect_amd_gpus(void);

/**
 * @brief Detect all Intel GPUs (oneAPI)
 * @return Number of Intel GPUs discovered
 */
int fb_registry_detect_intel_gpus(void);

/**
 * @brief Detect all Apple GPUs (Metal)
 * @return Number of Apple GPUs discovered
 */
int fb_registry_detect_apple_gpus(void);

/**
 * @brief Get GPU vendor from device
 * @param gpu_id GPU ID to query
 * @return GPU vendor type
 */
fb_gpu_vendor_t fb_registry_get_gpu_vendor(int gpu_id);

/**
 * @brief Get GPU compute capability
 * @param gpu_id GPU ID to query
 * @param major Output: major version
 * @param minor Output: minor version
 * @return 0 on success, non-zero on error
 */
int fb_registry_get_gpu_compute_capability(int gpu_id, int* major, int* minor);

/* ============================================================================
 * Load Monitoring
 * ========================================================================== */

/**
 * @brief Update load metrics for all devices
 * @return Number of devices updated
 */
int fb_registry_update_all_loads(void);

/**
 * @brief Get device with lowest current load
 * @param type Device type to search (or -1 for any)
 * @return Device ID with lowest load, or -1 if none available
 */
int fb_registry_get_least_loaded_device(fb_device_type_t type);

/**
 * @brief Get device with most available memory
 * @param type Device type to search (or -1 for any)
 * @return Device ID with most free memory, or -1 if none available
 */
int fb_registry_get_most_memory_device(fb_device_type_t type);

/* ============================================================================
 * Backend Library Detection
 * ========================================================================== */

/**
 * @brief Detect available CPU BLAS backends
 * @return Bitmask of available backends
 */
uint32_t fb_registry_detect_cpu_backends(void);

/**
 * @brief Detect available GPU BLAS backends
 * @return Bitmask of available backends
 */
uint32_t fb_registry_detect_gpu_backends(void);

/**
 * @brief Check if specific CPU backend is available
 * @param backend Backend type to check
 * @return true if backend library is loaded
 */
bool fb_registry_has_cpu_backend(int backend);

/**
 * @brief Check if specific GPU backend is available
 * @param backend Backend type to check
 * @return true if backend library is loaded
 */
bool fb_registry_has_gpu_backend(int backend);

/* ============================================================================
 * Diagnostic and Debug
 * ========================================================================== */

/**
 * @brief Print registry contents (all devices)
 */
void fb_registry_print(void);

/**
 * @brief Export registry to JSON
 * @param filename Output JSON filename
 * @return 0 on success, non-zero on error
 */
int fb_registry_export_json(const char* filename);

/**
 * @brief Get registry statistics
 * @param total_devices Output: total device count
 * @param cpu_count Output: CPU count
 * @param gpu_count Output: GPU count
 * @param available_count Output: available device count
 */
void fb_registry_get_stats(int* total_devices,
                           int* cpu_count,
                           int* gpu_count,
                           int* available_count);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_DEVICE_REGISTRY_H */
