/**
 * @file dispatch_unified.c
 * @brief Implementation of unified dispatch API
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/dispatch_unified.h"
#include "faster-blaster/backend_instance.h"
#include "faster-blaster/backend_plugin.h"
#include "faster-blaster/device_registry.h"
#include "faster-blaster/compute_manager.h"
#include "faster-blaster/judge_select.h"  /* fb_judge_build_dispatch_table, FB_SELECT_BALANCED */
#include "faster-blaster/backend_ids.h"   /* FB_BACKEND_ID_* stable constants */
#include "device_detection/cpu_detect.h"
#include "device_detection/gpu_detect.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/* Global state */
static bool g_initialized = false;
static bool g_verbose = false;
static fb_compute_device_t* g_pinned_device = NULL;  /* Manual device selection */

/* Judge-driven dispatch table — populated from stored profiles at startup.
 * g_judge_profile_dir may be overridden via fb_judge_set_profile_dir()
 * before calling fb_init(). */
static uint32_t g_judge_dispatch[FB_JUDGE_MAX_OPERATIONS];
static bool     g_judge_dispatch_loaded = false;
static char     g_judge_profile_dir[512] = "judge_profiles";

/* External plugin registration functions */
extern void fb_register_all_plugins(void);
extern const fb_backend_vtable_t* fb_reference_get_vtable(void);

/**
 * Map a plugin metadata name to its stable FB_BACKEND_ID_* constant.
 * Unknown names get a djb2 hash shifted above the reserved range so that
 * stored profiles don't collide with known backends.
 */
static uint32_t backend_name_to_id(const char *name)
{
    if (!name) return FB_BACKEND_ID_NONE;
    if (strcmp(name, "aocl-blis")  == 0) return FB_BACKEND_ID_AOCL_BLIS;
    if (strcmp(name, "blis")       == 0) return FB_BACKEND_ID_BLIS;
    if (strcmp(name, "openblas")   == 0) return FB_BACKEND_ID_OPENBLAS;
    if (strcmp(name, "mkl")        == 0) return FB_BACKEND_ID_MKL;
    if (strcmp(name, "accelerate") == 0) return FB_BACKEND_ID_ACCELERATE;
    if (strcmp(name, "cublas")     == 0) return FB_BACKEND_ID_CUBLAS;
    if (strcmp(name, "rocblas")    == 0) return FB_BACKEND_ID_ROCBLAS;
    if (strcmp(name, "onemkl")     == 0) return FB_BACKEND_ID_ONEMKL;
    if (strcmp(name, "metal")      == 0) return FB_BACKEND_ID_METAL;
    if (strcmp(name, "clblast")    == 0) return FB_BACKEND_ID_CLBLAST;
    if (strcmp(name, "clblas")     == 0) return FB_BACKEND_ID_CLBLAS;
    if (strcmp(name, "reference")  == 0) return FB_BACKEND_ID_REFERENCE;
    /* Unknown plugin — produce a stable hash in the range [0x80000000, 0xFFFFFFFE]. */
    uint32_t h = 5381u;
    for (const char *p = name; *p; p++)
        h = h * 33u ^ (uint32_t)(unsigned char)*p;
    return (h & 0x0FFFFFFFu) | 0x80000000u;
}

/* ============================================================================
 * Initialization
 * ========================================================================== */

int fb_init(void) {
    if (g_initialized) {
        if (g_verbose) {
            printf("[INFO] faster-blaster already initialized\n");
        }
        return 0;
    }
    
    printf("=============================================================\n");
    printf(" faster-blaster: Hybrid CPU/GPU Dispatch System\n");
    printf(" Version: 0.1.0-alpha\n");
    printf("=============================================================\n\n");
    
    /* Step 1: Register all plugins */
    printf("[1/5] Registering backend plugins...\n");
    fb_register_all_plugins();
    
    /* Step 2: Initialize device detection */
    printf("[2/5] Detecting compute devices...\n");
    if (fb_registry_init(FB_DISCOVERY_ALL) < 0) {
        fprintf(stderr, "[ERROR] Failed to initialize device registry\n");
        return -1;
    }
    
    /* Step 3: Initialize compute manager */
    printf("[3/5] Initializing compute manager...\n");
    if (fb_compute_manager_init() != 0) {
        fprintf(stderr, "[ERROR] Failed to initialize compute manager\n");
        return -1;
    }
    
    /* Step 4: Initialize backend instance manager */
    printf("[4/5] Initializing backend loader...\n");
    if (fb_backend_instance_manager_init() != 0) {
        fprintf(stderr, "[ERROR] Failed to initialize backend instance manager\n");
        return -1;
    }
    
    /* Step 5: Print detected devices */
    printf("[5/5] System ready!\n\n");
    fb_print_devices();

    /* Step 6: Build judge-driven per-operation dispatch table from stored
     * profiles.  Best-effort: failure does not block initialization.  On
     * first run (no profiles) probe-score selection remains in effect. */
    {
        fb_judge_init(g_judge_profile_dir);

        /* Collect IDs for all currently registered plugins. */
        uint32_t backend_ids[32];
        uint32_t n_backends = 0;
        const fb_plugin_registry_entry_t *pe = fb_get_registered_plugins();
        while (pe && n_backends < 31u) {
            if (pe->plugin && pe->plugin->metadata && pe->plugin->metadata->name)
                backend_ids[n_backends++] =
                    backend_name_to_id(pe->plugin->metadata->name);
            pe = pe->next;
        }
        /* Always include the reference backend as the guaranteed fallback. */
        backend_ids[n_backends++] = FB_BACKEND_ID_REFERENCE;

        if (g_verbose)
            printf("[Judge] Building dispatch table from '%s' (%u backends)...\n",
                   g_judge_profile_dir, n_backends);

        /* FB_DTYPE_F32 = 0 (see src/judge/judge_types.h). */
        fb_select_status_t jst = fb_judge_build_dispatch_table(
            g_judge_profile_dir,
            /*device_id=*/0,
            /*primary_dtype=*/(uint8_t)0,  /* FB_DTYPE_F32 */
            &FB_SELECT_BALANCED,
            /*overrides=*/NULL, /*n_overrides=*/0u,
            backend_ids, n_backends,
            /*fallback=*/FB_BACKEND_ID_REFERENCE,
            g_judge_dispatch);

        if (jst == FB_SELECT_OK || jst == FB_SELECT_WARN_DEGRADED) {
            g_judge_dispatch_loaded = true;
            if (g_verbose || jst == FB_SELECT_WARN_DEGRADED)
                printf("[Judge] Dispatch table loaded%s.\n",
                       jst == FB_SELECT_WARN_DEGRADED
                           ? " (some ops fell back to reference)" : "");
        } else if (jst == FB_SELECT_ERR_NO_PROFILES) {
            if (g_verbose)
                printf("[Judge] No profiles in '%s' — probe-score selection active.\n",
                       g_judge_profile_dir);
        } else {
            fprintf(stderr, "[WARN] fb_judge_build_dispatch_table() returned %d — "
                    "using probe-score selection.\n", (int)jst);
        }
    }

    g_initialized = true;
    printf("\n[SUCCESS] faster-blaster initialized successfully\n\n");

    return 0;
}

void fb_shutdown(void) {
    if (!g_initialized) {
        return;
    }
    
    if (g_verbose) {
        printf("\n[INFO] Shutting down faster-blaster...\n");
    }
    
    fb_judge_shutdown();
    g_judge_dispatch_loaded = false;

    fb_backend_instance_manager_shutdown();
    fb_compute_manager_shutdown();
    fb_registry_shutdown();

    g_initialized = false;
    g_pinned_device = NULL;
    
    if (g_verbose) {
        printf("[INFO] Shutdown complete\n");
    }
}

/* Alias for old API compatibility */
void fb_finalize(void) {
    fb_shutdown();
}

/* ============================================================================
 * Configuration
 * ========================================================================== */

void fb_set_policy(fb_scheduling_policy_t policy) {
    if (!g_initialized) {
        fb_init();
    }
    fb_set_dispatch_strategy((fb_dispatch_strategy_t)policy);
}

fb_scheduling_policy_t fb_get_policy(void) {
    if (!g_initialized) {
        fb_init();
    }
    return (fb_scheduling_policy_t)fb_get_dispatch_strategy();
}

/* ============================================================================
 * Backend Retrieval for Operations
 * ========================================================================== */

fb_backend_instance_t* fb_get_backend_for_operation(
    fb_operation_type_t op_type,
    fb_precision_t precision,
    size_t problem_size,
    const void** data_ptrs,
    uint32_t num_data_ptrs)
{
    if (!g_initialized) {
        fprintf(stderr, "[ERROR] System not initialized\n");
        return NULL;
    }
    
    (void)op_type;        /* TODO: Use for multi-backend selection */
    (void)problem_size;   /* TODO: Use for size-based backend selection */
    (void)data_ptrs;      /* TODO: Use for data locality */
    (void)num_data_ptrs;
    (void)precision;      /* TODO: Use for precision-specific backends */
    
    /* Step 1: Device selection */
    fb_compute_device_t* device = NULL;
    
    if (g_pinned_device) {
        /* Manual device pinning */
        device = g_pinned_device;
    } else {
        /* Automatic device selection */
        /* For now, just use first device until compute_manager is fully implemented */
        uint32_t device_count = fb_get_device_count();
        if (device_count == 0) {
            fprintf(stderr, "[ERROR] No devices available\n");
            return NULL;
        }
        device = fb_get_device(0);
    }
    
    if (!device) {
        fprintf(stderr, "[ERROR] No device selected\n");
        return NULL;
    }
    
    /* Step 2: Get backend for this device */
    fb_backend_instance_t* instance = fb_get_backend_for_device(device);
    
    if (!instance) {
        if (g_verbose) {
            printf("[INFO] No backend available for device %s\n", 
                   device->properties.name);
        }
        return NULL;
    }
    
    return instance;
}

fb_backend_instance_t* fb_get_current_backend(fb_precision_t precision,
                                              const void** data_ptrs,
                                              uint32_t num_data_ptrs)
{
    /* Default to Level 3 BLAS with arbitrary problem size */
    return fb_get_backend_for_operation(
        FB_OP_LEVEL3,
        precision,
        1024 * 1024,  /* Default problem size */
        data_ptrs,
        num_data_ptrs
    );
}

/* ============================================================================
 * Manual Device Control
 * ============================================================================ */

int fb_use_device(uint32_t device_id) {
    if (!g_initialized) {
        if (fb_init() != 0) {
            return -1;
        }
    }
    
    uint32_t device_count = fb_get_device_count();
    if (device_id >= device_count) {
        fprintf(stderr, "[ERROR] Invalid device ID: %u (only %u devices available)\n", 
                device_id, device_count);
        return -1;
    }
    
    g_pinned_device = fb_get_device(device_id);
    if (!g_pinned_device) {
        fprintf(stderr, "[ERROR] Failed to get device %u\n", device_id);
        return -1;
    }
    
    printf("[INFO] Pinned execution to device %u: %s\n", 
           device_id, g_pinned_device->properties.name);
    
    return 0;
}

void fb_use_auto(void) {
    g_pinned_device = NULL;
    if (g_verbose) {
        printf("[INFO] Switched to automatic device selection\n");
    }
}

void fb_set_verbose(bool enable) {
    g_verbose = enable;
}

/* ============================================================================
 * Backend Selection
 * ========================================================================== */

/* fb_get_current_backend() is now implemented above using fb_get_backend_for_operation() */

/* ============================================================================
 * Status and Debugging
 * ========================================================================== */

void fb_print_status(void) {
    if (!g_initialized) {
        printf("faster-blaster is not initialized\n");
        return;
    }
    
    printf("\n");
    printf("=============================================================\n");
    printf(" faster-blaster Status\n");
    printf("=============================================================\n\n");
    
    /* Device information */
    printf("--- Detected Devices ---\n");
    fb_print_devices();
    
    /* Current policy */
    printf("\n--- Current Configuration ---\n");
    fb_dispatch_strategy_t strategy = fb_get_dispatch_strategy();
    const char* strategy_names[] = {
        "FASTEST", "LOAD_BALANCED", "POWER_EFFICIENT", 
        "DATA_LOCALITY", "ROUND_ROBIN", "ADAPTIVE", "CUSTOM"
    };
    printf("Dispatch strategy: %s\n", strategy_names[strategy]);
    
    if (g_pinned_device) {
        printf("Device selection: MANUAL (device %u: %s)\n",
               g_pinned_device->properties.device_id, g_pinned_device->properties.name);
    } else {
        printf("Device selection: AUTOMATIC\n");
    }
    
    printf("Verbose logging: %s\n", g_verbose ? "ON" : "OFF");
    
    /* Backend instances */
    printf("\n--- Loaded Backends ---\n");
    fb_backend_instance_t* instances[32];
    int num_instances = fb_backend_list_instances(instances, 32);
    
    if (num_instances == 0) {
        printf("  No backends loaded yet (lazy initialization)\n");
    } else {
        for (int i = 0; i < num_instances; i++) {
            printf("\nBackend %d:\n", i + 1);
            fb_backend_print_instance(instances[i]);
        }
    }
    
    /* Judge dispatch */
    printf("\n--- Judge Dispatch Table ---\n");
    if (g_judge_dispatch_loaded) {
        printf("  Profile-driven dispatch: ACTIVE  (profile dir: %s)\n",
               g_judge_profile_dir);
    } else {
        printf("  Profile-driven dispatch: NOT LOADED\n");
        printf("  Profile dir: %s\n", g_judge_profile_dir);
        printf("  (Run bench to generate profiles; dispatch uses probe-score selection.)\n");
    }

    printf("\n");
    printf("=============================================================\n\n");
}

/* ============================================================================
 * Judge Dispatch Accessors
 * ========================================================================== */

/**
 * Override the profile directory before calling fb_init().
 * Has no effect after initialization.
 */
void fb_judge_set_profile_dir(const char *dir)
{
    if (!dir || g_initialized) return;
    snprintf(g_judge_profile_dir, sizeof(g_judge_profile_dir), "%s", dir);
}

/** Return true if a judge dispatch table was successfully loaded at startup. */
bool fb_judge_dispatch_is_loaded(void)
{
    return g_judge_dispatch_loaded;
}

/**
 * Return the judge-selected backend ID for a specific operation.
 *
 * Returns FB_BACKEND_ID_NONE when the dispatch table is not loaded or
 * the op_id is out of range.  The caller falls back to probe-score
 * selection in that case.
 */
uint32_t fb_judge_get_routed_backend_id(uint32_t op_id)
{
    if (!g_judge_dispatch_loaded || op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS)
        return FB_BACKEND_ID_NONE;
    return g_judge_dispatch[op_id];
}

/* ============================================================================
 * Version Information
 * ========================================================================== */

#define FB_VERSION_MAJOR 0
#define FB_VERSION_MINOR 1
#define FB_VERSION_PATCH 0

const char* fb_get_version_string(void) {
    return "0.1.0-alpha";
}

void fb_get_version(int* major, int* minor, int* patch) {
    if (major) *major = FB_VERSION_MAJOR;
    if (minor) *minor = FB_VERSION_MINOR;
    if (patch) *patch = FB_VERSION_PATCH;
}
