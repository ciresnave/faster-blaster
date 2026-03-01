/**
 * @file plugin_aocl_blis.c
 * @brief AMD AOCL BLIS plugin implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include "../backends/backend_auto_detect.h"   /* fb_enumerate_and_populate  */
#include "faster-blaster/vtable_autofill.h"    /* fb_vtable_sync_ext_ops     */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef _WIN32
    #include <windows.h>
    #include <intrin.h>
    #define FB_LOAD_LIBRARY(path) LoadLibraryA(path)
    #define FB_GET_PROC_ADDRESS(handle, name) GetProcAddress((HMODULE)(handle), name)
#else
    #include <dlfcn.h>
    #include <cpuid.h>
    #define FB_LOAD_LIBRARY(path) dlopen(path, RTLD_LAZY)
    #define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#endif

/* CPU vendor detection */
static int is_amd_cpu(void) {
    int cpu_info[4];
    char vendor[13];
    
#ifdef _WIN32
    __cpuid(cpu_info, 0);
#else
    __cpuid(0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#endif
    
    memcpy(vendor, &cpu_info[1], 4);
    memcpy(vendor + 4, &cpu_info[3], 4);
    memcpy(vendor + 8, &cpu_info[2], 4);
    vendor[12] = '\0';
    
    return strcmp(vendor, "AuthenticAMD") == 0;
}

/* Plugin context - holds library handle */
typedef struct {
    fb_lib_handle_t lib_handle;
} aocl_plugin_context_t;

/* Static backend vtable that will be populated during init */
static fb_backend_vtable_t g_aocl_vtable;

/* File-scoped context - set during init */
static aocl_plugin_context_t* g_aocl_context = NULL;

/* Plugin metadata */
static const fb_plugin_metadata_t g_aocl_metadata = {
    .name = "aocl-blis",
    .version = "4.2.1",
    .vendor = "AMD",
    .description = "AMD Optimizing CPU Libraries - BLIS (CBLAS interface)",
    .api_version = 1,
    .capabilities = FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 | 
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_THREADSAFE
};

/* BLAS operations are dispatched directly via ext_ops[op][conv] populated by
 * fb_enumerate_and_populate() — no per-operation wrapper functions needed. */

/* Backend capability functions */
static uint32_t fb_aocl_get_capabilities_wrapper(void* handle) {
    (void)handle;
    return FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
           FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_COMPLEX;
}

static int fb_aocl_get_num_threads_wrapper(void* handle) {
    (void)handle;
    /* AOCL BLIS may support thread control - return default */
    return 1;
}

static void fb_aocl_set_num_threads_wrapper(void* handle, int num_threads) {
    (void)handle;
    (void)num_threads;
    /* TODO: Call AOCL thread control if available */
}

/* Plugin probe function - search for AOCL BLIS and return compatibility score */
static fb_plugin_probe_result_t aocl_probe(fb_lib_handle_t unused_lib_handle, const char** search_paths) {
    fb_plugin_probe_result_t result = {0};
    
    const char* lib_names[] = {
#ifdef _WIN32
        "AOCL-LibBlis-Win-dll.dll",
        "AOCL-LibBlis-Win-MT-dll.dll",
#elif defined(__APPLE__)
        "libaocl_blis.dylib",
#else
        "libaocl_blis.so",
#endif
        NULL
    };
    
    const char* default_paths[] = {
#ifdef _WIN32
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64",
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib",
#elif defined(__APPLE__)
        "/opt/AMD/aocl/lib",
        "/usr/local/lib",
#else
        "/opt/AMD/aocl/lib",
        "/usr/lib/x86_64-linux-gnu",
#endif
        NULL
    };
    
    /* Try to find the library */
    fb_lib_handle_t test_handle = fb_plugin_load_library(lib_names, 
                                                          search_paths ? search_paths : default_paths);
    
    if (test_handle) {
        /* Check for CBLAS symbols to verify it's AOCL BLIS */
        void* cblas_sdot_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sdot");
        void* cblas_sgemm_sym = FB_GET_PROC_ADDRESS(test_handle, "cblas_sgemm");
        
        if (cblas_sdot_sym && cblas_sgemm_sym) {
            /* Adjust score based on CPU vendor */
            if (is_amd_cpu()) {
                result.score = 95; /* AOCL is highly optimized for AMD CPUs */
                result.reason = "Found AOCL BLIS with CBLAS interface (AMD CPU detected)";
            } else {
                result.score = 70; /* Lower priority on non-AMD CPUs */
                result.reason = "Found AOCL BLIS with CBLAS interface (non-AMD CPU - not optimal)";
            }
            result.library_path = NULL; /* Init will load the library */
        } else {
            result.score = 0;
            result.reason = "Library found but missing required CBLAS symbols";
        }
        
        fb_plugin_unload_library(test_handle);
    } else {
        result.score = 0;
        result.reason = "AOCL BLIS library not found in search paths";
    }
    
    return result;
}

/* Plugin init function - load library and map function pointers */
static int aocl_init(fb_lib_handle_t lib_handle, fb_plugin_context_t** ctx_out) {
    printf("[AOCL] Init called\n");
    if (!ctx_out) {
        printf("[AOCL] ctx_out is NULL\n");
        return -1;
    }
    
    /* Load the library if not provided */
    if (!lib_handle) {
        printf("[AOCL] Loading library...\n");
        const char* lib_names[] = {
#ifdef _WIN32
            /* Prioritize multi-threaded version */
            "AOCL-LibBlis-Win-MT-dll.dll",
            "AOCL-LibBlis-Win-dll.dll",
#elif defined(__APPLE__)
            "libaocl_blis.dylib",
#else
            "libaocl_blis.so",
#endif
            NULL
        };
        
        const char* default_paths[] = {
#ifdef _WIN32
            /* Try MT version paths first */
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64",
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\ILP64",
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib",
#elif defined(__APPLE__)
            "/opt/AMD/aocl/lib",
            "/usr/local/lib",
#else
            "/opt/AMD/aocl/lib",
            "/usr/lib/x86_64-linux-gnu",
#endif
            NULL
        };
        
        /* Try to load with full paths for MT version first */
#ifdef _WIN32
        const char* mt_paths[] = {
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64\\AOCL-LibBlis-Win-MT-dll.dll",
            "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\ILP64\\AOCL-LibBlis-Win-MT-dll.dll",
            NULL
        };
        for (int i = 0; mt_paths[i]; i++) {
            lib_handle = FB_LOAD_LIBRARY(mt_paths[i]);
            if (lib_handle) {
                printf("[AOCL] Loaded MT version from: %s\n", mt_paths[i]);
                break;
            }
        }
#endif
        
        if (!lib_handle) {
            lib_handle = fb_plugin_load_library(lib_names, default_paths);
        }
        printf("[AOCL] Library load result: %p\n", lib_handle);
        if (!lib_handle) {
            printf("[AOCL] Failed to load library\n");
            return -1;
        }
    }
    
    printf("[AOCL] Allocating context...\n");
    /* Allocate plugin context */
    aocl_plugin_context_t* ctx = (aocl_plugin_context_t*)calloc(1, sizeof(aocl_plugin_context_t));
    printf("[AOCL] Context allocated at %p\n", ctx);
    if (!ctx) {
        printf("[AOCL] Allocation failed!\n");
        return -2; /* Out of memory */
    }
    
    printf("[AOCL] Setting lib_handle...\n");
    ctx->lib_handle = lib_handle;

    /* Enumerate all exported BLAS/LAPACK symbols and fill ext_ops[op][conv]. */
    fb_enumerate_and_populate(&g_aocl_vtable, lib_handle);
    fb_vtable_sync_ext_ops(&g_aocl_vtable);

    printf("[AOCL] Checking critical functions...\n");
    /* Require sdot and sgemm to consider the backend usable. */
    if (g_aocl_vtable.sdot == NULL || g_aocl_vtable.sgemm == NULL) {
        printf("[AOCL] Missing critical symbols!\n");
        free(ctx);
        return -3;
    }

    printf("[AOCL] Populating vtable...\n");
    g_aocl_context = ctx;

    /* CPU backend properties - null operations for memory/stream management */
    g_aocl_vtable.mem_alloc = NULL;
    g_aocl_vtable.mem_free = NULL;
    g_aocl_vtable.mem_upload = NULL;
    g_aocl_vtable.mem_download = NULL;
    g_aocl_vtable.mem_copy = NULL;
    g_aocl_vtable.stream_create = NULL;
    g_aocl_vtable.stream_destroy = NULL;
    g_aocl_vtable.stream_sync = NULL;
    g_aocl_vtable.stream_set = NULL;
    
    /* Backend capabilities */
    g_aocl_vtable.get_capabilities = fb_aocl_get_capabilities_wrapper;
    g_aocl_vtable.get_num_threads = fb_aocl_get_num_threads_wrapper;
    g_aocl_vtable.set_num_threads = fb_aocl_set_num_threads_wrapper;
    
    /* Return initialized context */
    *ctx_out = (fb_plugin_context_t*)ctx;
    return 0; /* Success */
}

/* Shutdown plugin */
static void aocl_shutdown(fb_plugin_context_t* ctx) {
    if (ctx) {
        aocl_plugin_context_t* aocl_ctx = (aocl_plugin_context_t*)ctx;
        /* Library handle will be freed by plugin registry */
        free(aocl_ctx);
    }
}

/* Get backend vtable */
static const fb_backend_vtable_t* aocl_get_vtable(fb_plugin_context_t* ctx) {
    (void)ctx;
    return &g_aocl_vtable;
}

/* Get plugin context */
static void* aocl_get_context(fb_plugin_context_t* ctx) {
    return ctx;
}

/* Plugin interface */
static const fb_backend_plugin_t g_aocl_plugin = {
    .metadata = &g_aocl_metadata,
    .probe = aocl_probe,
    .init = aocl_init,
    .get_vtable = aocl_get_vtable,
    .get_context = aocl_get_context,
    .shutdown = aocl_shutdown,
    .set_num_threads = NULL, /* TODO: Implement if AOCL supports threading control */
    .get_num_threads = NULL
};

/* Plugin registration function - must be called explicitly */
void fb_register_aocl_plugin(void) {
    fb_register_plugin(&g_aocl_plugin);
}
