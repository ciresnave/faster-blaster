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
#include "device_detection/cpu_detect.h"
#include "device_detection/gpu_detect.h"
#include <stdio.h>
#include <stdbool.h>

/* Global state */
static bool g_initialized = false;
static bool g_verbose = false;
static fb_compute_device_t* g_pinned_device = NULL;  /* Manual device selection */

/* External plugin registration functions */
extern void fb_register_all_plugins(void);
extern const fb_backend_vtable_t* fb_reference_get_vtable(void);

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
    
    printf("\n");
    printf("=============================================================\n\n");
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
