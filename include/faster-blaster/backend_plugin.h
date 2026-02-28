/**
 * @file backend_plugin.h
 * @brief Plugin architecture for dynamically loadable BLAS/LAPACK backends
 * 
 * This interface enables runtime discovery and loading of backend implementations.
 * Each backend (AOCL BLIS, standard BLIS, OpenBLAS, MKL, etc.) is a self-contained
 * plugin that can be developed, tested, and deployed independently.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_BACKEND_PLUGIN_H
#define FASTER_BLASTER_BACKEND_PLUGIN_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration - full definition and typedef in backend_interface.h */
struct fb_backend_vtable;

/**
 * Plugin capability flags
 */
typedef enum {
    FB_PLUGIN_CAP_CPU          = 1 << 0,  /**< CPU execution */
    FB_PLUGIN_CAP_GPU          = 1 << 1,  /**< GPU execution */
    FB_PLUGIN_CAP_LEVEL1       = 1 << 2,  /**< BLAS Level 1 operations */
    FB_PLUGIN_CAP_LEVEL2       = 1 << 3,  /**< BLAS Level 2 operations */
    FB_PLUGIN_CAP_LEVEL3       = 1 << 4,  /**< BLAS Level 3 operations */
    FB_PLUGIN_CAP_LAPACK       = 1 << 5,  /**< LAPACK operations */
    FB_PLUGIN_CAP_SINGLE_PREC  = 1 << 6,  /**< Single precision */
    FB_PLUGIN_CAP_DOUBLE_PREC  = 1 << 7,  /**< Double precision */
    FB_PLUGIN_CAP_COMPLEX      = 1 << 8,  /**< Complex types */
    FB_PLUGIN_CAP_THREADSAFE   = 1 << 9,  /**< Thread-safe operations */
} fb_plugin_capability_t;

/**
 * Plugin metadata
 */
typedef struct {
    const char* name;           /**< Plugin name (e.g., "aocl-blis") */
    const char* version;        /**< Plugin version */
    const char* vendor;         /**< Vendor (e.g., "AMD", "Intel") */
    const char* description;    /**< Human-readable description */
    uint32_t api_version;       /**< faster-blaster API version */
    uint32_t capabilities;      /**< Bitfield of fb_plugin_capability_t */
} fb_plugin_metadata_t;

/**
 * Plugin context - opaque handle to plugin-specific state
 */
typedef struct fb_plugin_context fb_plugin_context_t;

/**
 * Library handle - opaque handle to loaded DLL/SO
 */
typedef void* fb_lib_handle_t;

/**
 * Plugin probe result
 */
typedef struct {
    int score;                  /**< Compatibility score (0-100), 0 = incompatible */
    const char* library_path;   /**< Full path to library if found, NULL otherwise */
    const char* reason;         /**< Explanation of score (for diagnostics) */
} fb_plugin_probe_result_t;

/**
 * Plugin interface - all plugins must implement this
 */
typedef struct {
    /** Plugin metadata */
    const fb_plugin_metadata_t* metadata;
    
    /**
     * Probe for compatible library
     * 
     * Searches for a compatible library and returns compatibility score.
     * Higher score = better match. Plugin with highest score wins.
     * 
     * @param lib_handle If non-NULL, probe this specific library handle.
     *                   If NULL, search system for compatible library.
     * @param search_paths Array of paths to search (NULL-terminated), may be NULL
     * @return Probe result with score and library path
     */
    fb_plugin_probe_result_t (*probe)(fb_lib_handle_t lib_handle, const char** search_paths);
    
    /**
     * Initialize plugin with a library
     * 
     * @param lib_handle Handle to loaded library (from probe)
     * @param ctx_out Output parameter for plugin context
     * @return 0 on success, negative on error
     */
    int (*init)(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out);
    
    /**
     * Get the backend trait vtable
     * 
     * @param ctx Plugin context from init()
     * @return Pointer to vtable, NULL on error
     */
    const struct fb_backend_vtable* (*get_vtable)(fb_plugin_context_t* ctx);
    
    /**
     * Get plugin context (for backend-specific operations)
     * 
     * @param ctx Plugin context from init()
     * @return Opaque pointer to plugin-specific state
     */
    void* (*get_context)(fb_plugin_context_t* ctx);
    
    /**
     * Shutdown and cleanup plugin
     * 
     * @param ctx Plugin context to cleanup
     */
    void (*shutdown)(fb_plugin_context_t* ctx);
    
    /**
     * Set number of threads (optional)
     * 
     * @param ctx Plugin context
     * @param num_threads Number of threads, 0 for auto
     */
    void (*set_num_threads)(fb_plugin_context_t* ctx, int num_threads);
    
    /**
     * Get number of threads (optional)
     * 
     * @param ctx Plugin context
     * @return Number of threads, 1 if not threaded
     */
    int (*get_num_threads)(fb_plugin_context_t* ctx);
    
} fb_backend_plugin_t;

/**
 * Plugin registry entry
 */
typedef struct fb_plugin_registry_entry {
    const fb_backend_plugin_t* plugin;
    struct fb_plugin_registry_entry* next;
} fb_plugin_registry_entry_t;

/**
 * Register a backend plugin
 * 
 * Plugins are typically registered via static initialization or explicit calls.
 * 
 * @param plugin Plugin interface to register
 * @return 0 on success, negative on error
 */
int fb_register_plugin(const fb_backend_plugin_t* plugin);

/**
 * Discover and load best matching plugin for a backend type
 * 
 * Probes all registered plugins and selects the one with highest score.
 * 
 * @param backend_name Preferred backend name (e.g., "blis", "openblas"), NULL for auto
 * @param search_paths Array of paths to search (NULL-terminated), may be NULL
 * @param ctx_out Output parameter for plugin context
 * @return Plugin interface on success, NULL on failure
 */
const fb_backend_plugin_t* fb_load_best_plugin(
    const char* backend_name,
    const char** search_paths,
    fb_plugin_context_t** ctx_out
);

/**
 * Get list of all registered plugins
 * 
 * @return Head of plugin registry linked list
 */
const fb_plugin_registry_entry_t* fb_get_registered_plugins(void);

/**
 * Utility: Load library from search paths
 * 
 * @param library_names Array of library name patterns (NULL-terminated)
 * @param search_paths Array of directory paths (NULL-terminated), may be NULL
 * @return Library handle on success, NULL on failure
 */
fb_lib_handle_t fb_plugin_load_library(const char** library_names, const char** search_paths);

/**
 * Utility: Get function pointer from library
 * 
 * @param lib_handle Library handle
 * @param symbol_name Function name
 * @return Function pointer, NULL if not found
 */
void* fb_plugin_get_symbol(fb_lib_handle_t lib_handle, const char* symbol_name);

/**
 * Utility: Unload library
 * 
 * @param lib_handle Library handle to unload
 */
void fb_plugin_unload_library(fb_lib_handle_t lib_handle);

/**
 * Built-in plugin registration functions
 * These must be called explicitly during library initialization
 */
/* CPU Plugins */
void fb_register_aocl_plugin(void);
void fb_register_standard_blis_plugin(void);
void fb_register_openblas_plugin(void);
void fb_register_mkl_plugin(void);
void fb_register_accelerate_plugin(void);

/* GPU Plugins */
void fb_register_cublas_plugin(void);
void fb_register_rocblas_plugin(void);
void fb_register_onemkl_plugin(void);
void fb_register_metal_plugin(void);

/* Fallback / correctness oracle */
void fb_register_reference_plugin(void);

/**
 * Initialize plugin system and register all built-in plugins
 */
void fb_init_plugins(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_BACKEND_PLUGIN_H */
