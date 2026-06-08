/**
 * @file plugin_clblast.c
 * @brief CLBlast OpenCL BLAS Plugin
 * 
 * CLBlast is a modern, lightweight, performant and tunable OpenCL BLAS library.
 * Supports a wide variety of GPU devices via OpenCL.
 * 
 * Homepage: https://github.com/CNugteren/CLBlast
 */

#include "faster-blaster/backend_plugin.h"
#include "faster-blaster/gpu_backend_trait.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CLBlast availability check */
#if defined(FB_ENABLE_OPENCL)
#define CLBLAST_AVAILABLE 1
#else
#define CLBLAST_AVAILABLE 0
#endif

#if CLBLAST_AVAILABLE
#include <CL/cl.h>
#endif

#ifdef _WIN32
    #include <windows.h>
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* OpenCL GPU detection */
static int is_opencl_gpu_available(void) {
#if CLBLAST_AVAILABLE
    cl_platform_id platform_id = NULL;
    cl_uint num_platforms = 0;
    
    /* Try to get at least one OpenCL platform */
    cl_int err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (err == CL_SUCCESS && num_platforms > 0) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        return (err == CL_SUCCESS && num_devices > 0);
    }
#endif
    return 0;
}

/* Plugin context */
typedef struct {
    fb_lib_handle_t lib_handle;
    int device_count;
    void* backend_handle;  /* CLBlast backend handle from trait */
    fb_backend_vtable_t* vtable;  /* Per-instance vtable */
} clblast_plugin_context_t;

/* Forward declarations for wrapper functions */
#if CLBLAST_AVAILABLE
extern void clblast_register_instance(const fb_backend_vtable_t* vtable, void* backend_handle);
extern void clblast_unregister_instance(const fb_backend_vtable_t* vtable);
extern void clblast_populate_vtable(fb_backend_vtable_t* vtable);
#endif

static const fb_plugin_metadata_t g_clblast_metadata = {
    .name = "clblast",
    .version = "1.6.0",
    .vendor = "Cedric Nugteren",
    .description = "CLBlast - Tuned OpenCL BLAS library",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_GPU | FB_PLUGIN_CAP_LEVEL1 |
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC
};

/* Probe function */
static fb_plugin_probe_result_t clblast_probe(fb_lib_handle_t unused_lib_handle, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
#if !CLBLAST_AVAILABLE
    result.score = 0;
    result.reason = "CLBlast not enabled in build (FB_ENABLE_OPENCL not defined)";
    return result;
#else
    
    /* Check if OpenCL GPU is available */
    if (!is_opencl_gpu_available()) {
        result.score = 0;
        result.reason = "No OpenCL GPU detected";
        return result;
    }
    
    /* Try to find CLBlast library */
    const char* lib_names[] = {
#ifdef _WIN32
        "clblast.dll",
#elif defined(__APPLE__)
        "libclblast.dylib",
#else
        "libclblast.so",
        "libclblast.so.1",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#ifdef _WIN32
        "C:\\libraries\\vcpkg\\installed\\x64-windows\\bin",
        "C:\\vcpkg\\installed\\x64-windows\\bin",
        "C:\\Program Files\\CLBlast\\lib",
        "C:\\Program Files (x86)\\CLBlast\\lib",
        ".",  /* Current directory */
#elif defined(__APPLE__)
        "/usr/local/lib",
        "/opt/homebrew/lib",
#else
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
#endif
        NULL
    };
    
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names,
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        result.score = 85; /* High score for OpenCL GPU with CLBlast */
        result.library_path = NULL;
        result.reason = "Found OpenCL GPU with CLBlast library";
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "OpenCL GPU detected but CLBlast library not found";
    }
    
    return result;
#endif
}

static int clblast_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    if (!ctx_out) {
        return -1;
    }
    
#if !CLBLAST_AVAILABLE
    return -1;
#else
    
    /* If lib_handle not provided, load the library ourselves */
    fb_lib_handle_t actual_lib_handle = lib_handle;
    bool we_loaded_it = false;
    
    if (!actual_lib_handle) {
        const char* lib_names[] = {
#ifdef _WIN32
            "clblast.dll",
#elif defined(__APPLE__)
            "libclblast.dylib",
#else
            "libclblast.so",
            "libclblast.so.1",
#endif
            NULL
        };
        
        const char* default_paths[] = {
#ifdef _WIN32
            "C:\\\\libraries\\\\vcpkg\\\\installed\\\\x64-windows\\\\bin",
            "C:\\\\vcpkg\\\\installed\\\\x64-windows\\\\bin",
            "C:\\\\Program Files\\\\CLBlast\\\\lib",
            "C:\\\\Program Files (x86)\\\\CLBlast\\\\lib",
            ".",
#elif defined(__APPLE__)
            "/usr/local/lib",
            "/opt/homebrew/lib",
#else
            "/usr/local/lib",
            "/usr/lib/x86_64-linux-gnu",
#endif
            NULL
        };
        
        actual_lib_handle = fb_plugin_load_library(lib_names, default_paths);
        we_loaded_it = true;
        
        if (!actual_lib_handle) {
            fprintf(stderr, "CLBlast plugin: Failed to load CLBlast library\\n");
            return -1;
        }
    }
    
    clblast_plugin_context_t* ctx = (clblast_plugin_context_t*)calloc(1, sizeof(clblast_plugin_context_t));
    if (!ctx) {
        if (we_loaded_it && actual_lib_handle) {
            fb_plugin_unload_library(actual_lib_handle);
        }
        return -1;
    }
    
    ctx->lib_handle = actual_lib_handle;
    ctx->device_count = 0;
    ctx->backend_handle = NULL;
    
    /* Initialize CLBlast backend using trait - assume device 0 for now */
    int result = fb_clblast_trait.init(0, actual_lib_handle, &ctx->backend_handle);
    if (result != 0) {
        if (we_loaded_it && actual_lib_handle) {
            fb_plugin_unload_library(actual_lib_handle);
        }
        free(ctx);
        return -1;
    }
    
    /* Allocate per-instance vtable */
    ctx->vtable = (fb_backend_vtable_t*)calloc(1, sizeof(fb_backend_vtable_t));
    if (!ctx->vtable) {
        if (we_loaded_it && actual_lib_handle) {
            fb_plugin_unload_library(actual_lib_handle);
        }
        free(ctx);
        return -1;
    }
    
    /* Populate vtable with wrapper functions */
    clblast_populate_vtable(ctx->vtable);
    
    /* Register this instance in the vtable->handle registry */
    clblast_register_instance(ctx->vtable, ctx->backend_handle);
    
    fprintf(stderr, "[CLBlast] Instance initialized: vtable=%p, backend_handle=%p\n", 
           ctx->vtable, ctx->backend_handle);
    
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0;
#endif
}

static const fb_backend_vtable_t* clblast_get_vtable(fb_plugin_context_t* ctx) {
#if CLBLAST_AVAILABLE
    if (!ctx) {
        return NULL;
    }
    clblast_plugin_context_t* clblast_ctx = (clblast_plugin_context_t*)ctx;
    /* No need to set thread-local handle - wrappers will look it up from registry */
    return clblast_ctx->vtable;
#else
    (void)ctx;
    return NULL;
#endif
}

static void* clblast_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

static void clblast_shutdown(fb_plugin_context_t* ctx) {
    if (!ctx) {
        return;
    }
#if CLBLAST_AVAILABLE
    clblast_plugin_context_t* clblast_ctx = (clblast_plugin_context_t*)ctx;
    
    /* Unregister from registry */
    clblast_unregister_instance(clblast_ctx->vtable);
    
    /* Shutdown CLBlast backend via trait */
    if (clblast_ctx->backend_handle && fb_clblast_trait.shutdown) {
        fb_clblast_trait.shutdown(clblast_ctx->backend_handle);
        clblast_ctx->backend_handle = NULL;
    }
    
    /* Free per-instance vtable */
    if (clblast_ctx->vtable) {
        free(clblast_ctx->vtable);
        clblast_ctx->vtable = NULL;
    }
    
    /* Unload library if we loaded it */
    if (clblast_ctx->lib_handle) {
        /* Note: Don't unload the library here - it may be used by other instances */
        clblast_ctx->lib_handle = NULL;
    }
#endif
    free(ctx);
}

static const fb_backend_plugin_t g_clblast_plugin = {
    .metadata = &g_clblast_metadata,
    .probe = clblast_probe,
    .init = clblast_init,
    .get_vtable = clblast_get_vtable,
    .get_context = clblast_get_context,
    .shutdown = clblast_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL
};

/* Plugin registration function */
void fb_register_clblast_plugin(void) {
    fb_register_plugin(&g_clblast_plugin);
}

#ifdef __cplusplus
}
#endif
