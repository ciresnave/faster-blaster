/**
 * @file test_device_detection.c
 * @brief Test program for device detection
 */

#include "core/device_registry.h"
#include <stdio.h>

int main(void) {
    printf("faster-blaster Device Detection Test\n");
    printf("====================================\n\n");
    
    // Initialize registry (detects all devices)
    printf("Initializing device registry...\n");
    if (fb_registry_init() != 0) {
        fprintf(stderr, "Failed to initialize device registry\n");
        return 1;
    }
    
    printf("Success!\n\n");
    
    // Print all detected devices
    fb_print_devices();
    
    // Find fastest FP32 device
    printf("\n=== Device Selection Tests ===\n");
    fb_compute_device_t* fastest = fb_find_fastest_device(FB_PRECISION_FP32);
    if (fastest) {
        printf("Fastest FP32 device: ");
        fb_print_device_summary(fastest);
    }
    
    // Find least loaded device
    fb_compute_device_t* least_loaded = fb_find_least_loaded_device();
    if (least_loaded) {
        printf("Least loaded device: ");
        fb_print_device_summary(least_loaded);
    }
    
    // Test capability queries
    printf("\n=== Capability Tests ===\n");
    uint32_t device_count = fb_get_device_count();
    for (uint32_t i = 0; i < device_count; i++) {
        fb_compute_device_t* dev = fb_get_device(i);
        printf("Device %u (%s):\n", i, dev->name);
        printf("  FP64: %s\n", fb_device_supports_precision(dev, FB_PRECISION_FP64) ? "Yes" : "No");
        printf("  FP16: %s\n", fb_device_supports_precision(dev, FB_PRECISION_FP16) ? "Yes" : "No");
        printf("  Tensor Cores: %s\n", fb_device_has_tensor_cores(dev) ? "Yes" : "No");
        printf("\n");
    }
    
    // Cleanup
    fb_registry_shutdown();
    
    printf("Test complete!\n");
    return 0;
}
