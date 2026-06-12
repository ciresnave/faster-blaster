/**
 * @file compute_manager.c
 * @brief Real compute manager with intelligent device selection
 */

#include "faster-blaster/compute_manager.h"
#include "faster-blaster/device_registry.h"
#include "faster-blaster/compute_device.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Global state */
static bool g_manager_initialized = false;
static fb_dispatch_strategy_t g_current_strategy = FB_DISPATCH_FASTEST;
static int g_round_robin_index = 0;

/* ============================================================================
 * Helper Functions
 * ========================================================================== */

/**
 * @brief Check if device supports required precision
 */
static bool device_supports_precision(const fb_compute_device_t* device, fb_precision_t precision) {
    if (!device) return false;
    
    switch (precision) {
        case FB_PRECISION_FP32:
            return true; // All devices support FP32
        case FB_PRECISION_FP64:
            return (device->properties.capabilities & FB_DEVICE_CAP_FP64) != 0;
        case FB_PRECISION_FP16:
            return (device->properties.capabilities & FB_DEVICE_CAP_FP16) != 0;
        case FB_PRECISION_BF16:
            return (device->properties.capabilities & FB_DEVICE_CAP_BF16) != 0;
        default:
            return false;
    }
}

static bool device_is_eligible(const fb_compute_device_t* device, fb_precision_t precision) {
    return device && device->is_available && device_supports_precision(device, precision);
}

/**
 * @brief Select device using specified strategy.
 *
 * This implementation uses availability and compatibility only.
 * Device-level scoring is intentionally removed from scheduling.
 */
static fb_compute_device_t* select_device_internal(fb_dispatch_strategy_t strategy,
                                                   fb_precision_t precision,
                                                   const void** data_ptrs,
                                                   uint32_t num_data_ptrs) {
    int device_count = fb_get_device_count();
    if (device_count == 0) {
        fprintf(stderr, "No devices available\n");
        return NULL;
    }
    
    if (strategy == FB_DISPATCH_ROUND_ROBIN) {
        for (int attempts = 0; attempts < device_count; attempts++) {
            g_round_robin_index = (g_round_robin_index + 1) % device_count;
            fb_compute_device_t* device = fb_get_device(g_round_robin_index);
            if (device_is_eligible(device, precision)) {
                return device;
            }
        }
        return NULL;
    }
    
    /* For all other strategies, choose the first available compatible device.
     * This avoids score-based device ranking in the scheduling layer. */
    for (int i = 0; i < device_count; i++) {
        fb_compute_device_t* device = fb_get_device(i);
        if (device_is_eligible(device, precision)) {
            return device;
        }
    }
    
    return NULL;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

int fb_compute_manager_init(void) {
    if (g_manager_initialized) {
        fprintf(stderr, "Compute manager already initialized\n");
        return 0;
    }
    
    printf("Initializing compute manager...\n");
    
    /* Initialize device registry if not already done */
    int device_count = fb_get_device_count();
    if (device_count == 0) {
        printf("  Device registry not initialized, initializing now...\n");
        device_count = fb_registry_init(FB_DISCOVERY_ALL);
        if (device_count <= 0) {
            fprintf(stderr, "Failed to initialize device registry\n");
            return -1;
        }
    }
    
    /* Set default strategy */
    g_current_strategy = FB_DISPATCH_FASTEST;
    g_round_robin_index = 0;
    
    g_manager_initialized = true;
    printf("Compute manager initialized with %d device(s)\n", device_count);
    printf("Default dispatch strategy: FASTEST\n");
    
    return 0;
}

void fb_compute_manager_shutdown(void) {
    if (!g_manager_initialized) {
        return;
    }
    
    printf("Shutting down compute manager...\n");
    
    g_manager_initialized = false;
    g_current_strategy = FB_DISPATCH_FASTEST;
    g_round_robin_index = 0;
}

void fb_set_dispatch_strategy(fb_dispatch_strategy_t strategy) {
    if (strategy < FB_DISPATCH_FASTEST || strategy > FB_DISPATCH_CUSTOM) {
        fprintf(stderr, "Invalid dispatch strategy: %d\n", strategy);
        return;
    }
    
    const char* strategy_names[] = {
        "FASTEST",
        "LOAD_BALANCED",
        "POWER_EFFICIENT",
        "DATA_LOCALITY",
        "ROUND_ROBIN",
        "ADAPTIVE",
        "CUSTOM"
    };
    
    g_current_strategy = strategy;
    printf("Dispatch strategy set to: %s\n", strategy_names[strategy]);
}

fb_dispatch_strategy_t fb_get_dispatch_strategy(void) {
    return g_current_strategy;
}

fb_compute_device_t* fb_select_device(fb_dispatch_strategy_t strategy,
                                      fb_precision_t precision,
                                      const void** data_ptrs,
                                      uint32_t num_data_ptrs) {
    if (!g_manager_initialized) {
        /* Auto-initialize if not done */
        if (fb_compute_manager_init() != 0) {
            return NULL;
        }
    }
    
    fb_compute_device_t* device = select_device_internal(strategy, precision, data_ptrs, num_data_ptrs);
    
    if (!device) {
        fprintf(stderr, "Failed to select device (strategy=%d, precision=%d)\n", strategy, precision);
    }
    
    return device;
}

/* ============================================================================
 * Advanced Scheduling Functions (Placeholder implementations)
 * ========================================================================== */

/**
 * @brief Set workload hint for adaptive scheduling
 */
void fb_set_workload_hint(fb_workload_hint_t hint) {
    /* TODO: Implement adaptive learning based on workload hints */
    (void)hint;
}

/**
 * @brief Get performance statistics
 */
void fb_get_performance_stats(fb_compute_device_t* device, 
                              double* avg_utilization,
                              double* total_operations) {
    if (!device) return;
    
    if (avg_utilization) {
        *avg_utilization = device->load.utilization_percent;
    }
    if (total_operations) {
        *total_operations = (double)device->load.active_operations;
    }
}

/**
 * @brief Print compute manager status
 */
void fb_print_manager_status(void) {
    printf("\n=== Compute Manager Status ===\n");
    
    if (!g_manager_initialized) {
        printf("Status: NOT INITIALIZED\n");
        printf("================================\n\n");
        return;
    }
    
    const char* strategy_names[] = {
        "FASTEST",
        "LOAD_BALANCED",
        "POWER_EFFICIENT",
        "DATA_LOCALITY",
        "ROUND_ROBIN",
        "ADAPTIVE",
        "CUSTOM"
    };
    
    printf("Status: INITIALIZED\n");
    printf("Current Strategy: %s\n", strategy_names[g_current_strategy]);
    printf("Devices: %d\n", fb_get_device_count());
    
    /* Show device load */
    int device_count = fb_get_device_count();
    for (int i = 0; i < device_count; i++) {
        fb_compute_device_t* device = fb_get_device(i);
        if (device) {
            printf("  [%d] %s: %.1f%% load, %d active ops\n",
                   i, device->properties.name,
                   device->load.utilization_percent,
                   device->load.active_operations);
        }
    }
    
    printf("================================\n\n");
}
