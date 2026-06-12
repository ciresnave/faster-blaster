/**
 * @file compute_manager_stub.c
 * @brief Compute Manager Stub Implementation
 * 
 * Temporary stub implementations until compute_manager.c is updated to new API
 */

#include "faster-blaster/compute_manager.h"
#include "faster-blaster/compute_device.h"
#include <stdio.h>
#include <stdint.h>

// Global state (minimal for stub)
static fb_dispatch_strategy_t g_strategy = FB_DISPATCH_FASTEST;
static bool g_initialized = false;

int fb_compute_manager_init(void) {
    if (g_initialized) {
        return 0;
    }
    g_initialized = true;
    g_strategy = FB_DISPATCH_FASTEST;
    return 0;
}

void fb_compute_manager_shutdown(void) {
    g_initialized = false;
}

void fb_set_dispatch_strategy(fb_dispatch_strategy_t strategy) {
    g_strategy = strategy;
}

fb_dispatch_strategy_t fb_get_dispatch_strategy(void) {
    return g_strategy;
}

fb_compute_device_t* fb_select_device(
    fb_dispatch_strategy_t strategy,
    fb_precision_t precision,
    const void** data_ptrs,
    uint32_t num_data_ptrs
) {
    (void)strategy;
    (void)precision;
    (void)data_ptrs;
    (void)num_data_ptrs;
    
    // Stub: return NULL (caller should handle gracefully)
    return NULL;
}
