/**
 * @file device_registry_stub.c
 * @brief Temporary stub for device registry until implementation matches new API
 */

#include "faster-blaster/device_registry.h"
#include "faster-blaster/compute_device.h"
#include <string.h>
#include <stdio.h>

// Stub implementation - returns minimal CPU device
static fb_compute_device_t g_stub_cpu_device = {0};
static bool g_initialized = false;

int fb_registry_init(fb_discovery_flags_t flags) {
    (void)flags;
    if (!g_initialized) {
        memset(&g_stub_cpu_device, 0, sizeof(g_stub_cpu_device));
        g_stub_cpu_device.properties.type = FB_DEVICE_TYPE_CPU;
        g_stub_cpu_device.properties.device_id = 0;
        strncpy(g_stub_cpu_device.properties.name, "CPU (stub)", 
                sizeof(g_stub_cpu_device.properties.name) - 1);
        g_initialized = true;
        return 1; // 1 device
    }
    return 0;
}

void fb_registry_shutdown(void) {
    g_initialized = false;
}

void fb_print_devices(void) {
    printf("Device Registry (Stub):\n");
    printf("  [0] CPU (stub)\n");
}

uint32_t fb_get_device_count(void) {
    return g_initialized ? 1 : 0;
}

fb_compute_device_t* fb_get_device(uint32_t index) {
    if (g_initialized && index == 0) {
        return &g_stub_cpu_device;
    }
    return NULL;
}

int fb_registry_get_device_count(void) {
    return g_initialized ? 1 : 0;
}

fb_compute_device_t* fb_registry_get_device(int device_id) {
    if (device_id == 0 && g_initialized) {
        return &g_stub_cpu_device;
    }
    return NULL;
}

int fb_registry_get_devices_by_type(fb_device_type_t type,
                                    fb_compute_device_t** devices,
                                    int max_devices) {
    if (type == FB_DEVICE_TYPE_CPU && g_initialized && max_devices > 0) {
        devices[0] = &g_stub_cpu_device;
        return 1;
    }
    return 0;
}

int fb_registry_find_devices(size_t min_memory,
                             double min_gflops,
                             uint32_t required_caps,
                             fb_compute_device_t** devices,
                             int max_devices) {
    (void)min_memory;
    (void)min_gflops;
    (void)required_caps;
    if (g_initialized && max_devices > 0) {
        devices[0] = &g_stub_cpu_device;
        return 1;
    }
    return 0;
}

int fb_registry_rescan(fb_discovery_flags_t flags) {
    (void)flags;
    return 0;
}

int fb_registry_register_callback(fb_device_event_callback_t callback, void* user_data) {
    (void)callback;
    (void)user_data;
    return -1; // Not supported in stub
}

void fb_registry_unregister_callback(int callback_id) {
    (void)callback_id;
}
