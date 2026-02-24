/**
 * @file backend_loader.h
 * @brief Dynamic backend loading and management
 * 
 * This module handles runtime detection and loading of optimized BLAS/LAPACK backends
 * including CPU (MKL, OpenBLAS, AOCL, Accelerate) and GPU (cuBLAS, rocBLAS) implementations.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_BACKEND_LOADER_H
#define FASTER_BLASTER_BACKEND_LOADER_H

#include "backend_interface.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Backend type identifiers
 */
typedef enum {
    FB_BACKEND_REFERENCE = 0,  /**< Reference implementation (always available) */
    FB_BACKEND_OPENBLAS,        /**< OpenBLAS (bundled, BSD license) */
    FB_BACKEND_MKL,             /**< Intel MKL (optional installer) */
    FB_BACKEND_ACCELERATE,      /**< Apple Accelerate (macOS/iOS system) */
    FB_BACKEND_AOCL,            /**< AMD AOCL (optional) */
    FB_BACKEND_BLIS,            /**< BLIS (optional) */
    FB_BACKEND_CUBLAS,          /**< NVIDIA cuBLAS (GPU, optional) */
    FB_BACKEND_ROCBLAS,         /**< AMD rocBLAS (GPU, optional) */
    FB_BACKEND_ONEMKL_GPU,      /**< Intel oneMKL GPU (optional) */
    FB_BACKEND_CUSTOM,          /**< User-provided custom backend */
    FB_BACKEND_COUNT
} fb_backend_type_t;

/**
 * Backend loader metadata (distinct from fb_backend_info_t in backend_interface.h)
 */
typedef struct {
    fb_backend_type_t type;
    const char* name;
    const char* version;
    const char* license;
    const char* vendor;
    uint32_t capabilities;          /**< Bitfield of fb_capability_t from backend_interface.h */
    bool available;                 /**< Is this backend currently available? */
    bool loaded;                    /**< Has this backend been loaded? */
    void* handle;                   /**< Dynamic library handle */
    int priority;                   /**< Higher = preferred (for auto-selection) */
} fb_backend_metadata_t;

/**
 * GPU device information
 */
typedef struct {
    int device_id;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    fb_backend_type_t backend_type; /**< Which GPU backend supports this device */
} fb_gpu_device_info_t;

/**
 * Initialize the backend loader system
 * Scans for available backends and collects information
 * 
 * @return 0 on success, negative on error
 */
int fb_backend_loader_init(void);

/**
 * Cleanup and shutdown backend loader
 */
void fb_backend_loader_shutdown(void);

/**
 * Get information about a specific backend
 * 
 * @param type Backend type
 * @param info Output parameter for backend metadata
 * @return 0 on success, negative on error
 */
int fb_backend_get_info(fb_backend_type_t type, fb_backend_metadata_t* info);

/**
 * Get list of all available backends
 * 
 * @param infos Array to fill with backend metadata (can be NULL to query count)
 * @param max_count Size of infos array
 * @return Number of available backends
 */
int fb_backend_list_available(fb_backend_metadata_t* infos, int max_count);

/**
 * Load a specific backend
 * 
 * @param type Backend type to load
 * @param vtable Output parameter for backend vtable
 * @return 0 on success, negative on error
 */
int fb_backend_load(fb_backend_type_t type, fb_backend_vtable_t* vtable);

/**
 * Unload a specific backend
 * 
 * @param type Backend type to unload
 */
void fb_backend_unload(fb_backend_type_t type);

/**
 * Auto-select best available backend based on system capabilities
 * 
 * @param prefer_gpu Prefer GPU backend if available
 * @return Backend type, or FB_BACKEND_REFERENCE if none found
 */
fb_backend_type_t fb_backend_auto_select(bool prefer_gpu);

/**
 * Detect CPU architecture and capabilities
 * Used for selecting optimal CPU backend
 * 
 * @param vendor Output buffer for CPU vendor string (min 64 bytes)
 * @param features Output bitfield of detected CPU features
 * @return 0 on success
 */
int fb_backend_detect_cpu(char* vendor, uint64_t* features);

/**
 * Query available GPU devices
 * 
 * @param devices Array to fill with GPU device info (can be NULL to query count)
 * @param max_devices Size of devices array
 * @return Number of available GPU devices
 */
int fb_backend_query_gpu_devices(fb_gpu_device_info_t* devices, int max_devices);

/**
 * Register a custom backend (for user-provided implementations)
 * 
 * @param name Backend name
 * @param vtable Backend function table
 * @param capabilities Capability flags
 * @return 0 on success, negative on error
 */
int fb_backend_register_custom(const char* name, const fb_backend_vtable_t* vtable, 
                                uint32_t capabilities);

/**
 * Get the currently active backend
 * 
 * @return Current backend type
 */
fb_backend_type_t fb_backend_get_current(void);

/**
 * Set the active backend
 * 
 * @param type Backend type to activate
 * @return 0 on success, negative on error
 */
int fb_backend_set_current(fb_backend_type_t type);

/**
 * Get human-readable error message for last backend operation
 * 
 * @return Error message string (static, do not free)
 */
const char* fb_backend_get_error(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_BACKEND_LOADER_H */
