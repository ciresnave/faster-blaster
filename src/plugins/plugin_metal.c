/**
 * @file plugin_metal.c
 * @brief Plugin implementation for Apple Metal Performance Shaders
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <string.h>

/* Plugin metadata */
static const fb_plugin_metadata_t metal_metadata = {
    .name = "metal",
    .version = "3.0",
    .vendor = "Apple Inc.",
    .description = "Metal Performance Shaders - GPU-accelerated compute for Apple Silicon and AMD GPUs on macOS",
    .capabilities = FB_PLUGIN_CAP_GPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_THREADSAFE
};

/**
 * @brief Check if Metal GPU is available
 * 
 * Detects Metal-capable GPUs on macOS (Apple Silicon or AMD GPUs).
 * Returns 1 if Metal GPU is available, 0 otherwise.
 */
static int is_metal_gpu_available(void) {
#if defined(__APPLE__) && defined(FB_ENABLE_METAL)
    /* Try to check Metal availability via Metal framework */
    #if __has_include(<Metal/Metal.h>)
        /* On macOS 10.11+ with Metal support */
        /* We could use MTLCreateSystemDefaultDevice() but that requires linking Metal framework */
        /* For now, check if we're on macOS and assume Metal is available */
        /* A more robust check would dynamically load Metal.framework */
        
        #ifdef __aarch64__
            /* Apple Silicon always has Metal GPU */
            return 1;
        #else
            /* Intel Mac - check for discrete GPU via system_profiler or IOKit */
            /* Simplified: assume Metal is available on modern macOS */
            void* metal_handle = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_NOW | RTLD_LOCAL);
            if (metal_handle) {
                /* Check for MTLCreateSystemDefaultDevice symbol */
                void* create_device = dlsym(metal_handle, "MTLCreateSystemDefaultDevice");
                dlclose(metal_handle);
                return create_device != NULL;
            }
        #endif
    #endif
#endif
    return 0;
}

/**
 * @brief Probe for Metal Performance Shaders library
 */
static fb_plugin_probe_result_t metal_probe(fb_lib_handle_t unused_lib_handle, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
#ifndef __APPLE__
    result.score = 0;
    result.reason = "Metal Performance Shaders only available on macOS";
    return result;
#endif

#ifndef FB_ENABLE_METAL
    result.score = 0;
    result.reason = "Metal not enabled in build (FB_ENABLE_METAL not defined)";
    return result;
#else
    
    /* Check for Metal GPU availability */
    if (!is_metal_gpu_available()) {
        result.score = 0;
        result.reason = "No Metal-capable GPU detected";
        return result;
    }
    
    const char* lib_names[] = {
        "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders",
        "/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/MetalPerformanceShaders",
        NULL
    };
    
    const char* default_paths[] = { NULL }; /* Use absolute paths */
    
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names,
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        /* Metal Performance Shaders uses different API than CBLAS */
        /* Check for MPS symbols */
        void* mps_matrix_sym = FB_GET_PROC_ADDRESS(test_handle, "MPSMatrixDescriptor");
        
        if (mps_matrix_sym) {
            #ifdef __aarch64__
                /* Apple Silicon - highest priority for GPU compute */
                result.score = 99;
                result.reason = "Found Metal GPU on Apple Silicon";
            #else
                /* Intel Mac with discrete GPU */
                result.score = 92;
                result.reason = "Found Metal GPU on Intel Mac";
            #endif
            result.library_path = NULL; /* Init will load the library */
        } else {
            result.score = 0;
            result.reason = "Metal framework found but missing MPS symbols";
        }
        
        FB_CLOSE_LIBRARY(test_handle);
    } else {
        result.score = 0;
        result.reason = "Metal Performance Shaders framework not found";
    }
    
    return result;
#endif
}

/**
 * @brief Initialize Metal Performance Shaders backend
 * 
 * Note: Returns empty vtable as GPU operations use trait interface.
 * This plugin handles detection/scoring; actual GPU operations use existing traits.
 */
static int metal_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    (void)lib_handle;
    (void)ctx_out;
    
#if defined(__APPLE__) && defined(FB_ENABLE_METAL)
    /* Metal GPU backend initialization would go here */
    /* Currently using trait interface for GPU operations */
    return 0; /* Success */
#else
    return -1; /* Not enabled */
#endif
}

/**
 * @brief Shutdown Metal Performance Shaders backend
 */
static void metal_shutdown(fb_plugin_context_t* ctx) {
    (void)ctx;
#if defined(__APPLE__) && defined(FB_ENABLE_METAL)
    /* Cleanup would go here */
#endif
}

/**
 * @brief Get vtable for Metal Performance Shaders backend
 * 
 * Returns empty vtable as GPU operations use trait interface.
 */
static const fb_backend_vtable_t* metal_get_vtable(fb_plugin_context_t* ctx) {
    (void)ctx;
    
    static const fb_backend_vtable_t vtable = {0};
    return &vtable;
}

/**
 * @brief Get plugin context
 */
static void* metal_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

/* Plugin interface */
static const fb_backend_plugin_t g_metal_plugin = {
    .metadata = &metal_metadata,
    .probe = metal_probe,
    .init = metal_init,
    .get_vtable = metal_get_vtable,
    .get_context = metal_get_context,
    .shutdown = metal_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL
};

void fb_register_metal_plugin(void) {
    fb_register_plugin(&g_metal_plugin);
}
