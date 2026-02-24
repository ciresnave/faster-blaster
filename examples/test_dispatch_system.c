/**
 * @file test_dispatch_system.c
 * @brief Comprehensive test of the hybrid dispatch system
 */

#include "core/compute_manager.h"
#include "core/device_registry.h"
#include "core/data_tracker.h"
#include "core/power_manager.h"
#include <stdio.h>
#include <stdlib.h>

void print_separator(const char* title) {
    printf("\n");
    printf("==================================================\n");
    printf(" %s\n", title);
    printf("==================================================\n\n");
}

int main(void) {
    printf("faster-blaster Hybrid Dispatch System Test\n");
    printf("==========================================\n\n");
    
    // ========================================================================
    // STEP 1: Initialize System
    // ========================================================================
    
    print_separator("STEP 1: System Initialization");
    
    printf("Initializing compute manager...\n");
    if (fb_compute_manager_init() != 0) {
        fprintf(stderr, "Failed to initialize compute manager\n");
        return 1;
    }
    printf("✓ Compute manager initialized\n\n");
    
    // ========================================================================
    // STEP 2: Device Discovery
    // ========================================================================
    
    print_separator("STEP 2: Device Discovery");
    
    fb_print_devices();
    
    uint32_t device_count = fb_get_device_count();
    uint32_t gpu_count = fb_get_gpu_count();
    
    printf("Summary: %u total devices (%u GPUs)\n", device_count, gpu_count);
    
    // ========================================================================
    // STEP 3: Power Management
    // ========================================================================
    
    print_separator("STEP 3: Power Management");
    
    fb_print_power_status();
    
    // Set power mode
    printf("\nSetting power mode to ADAPTIVE...\n");
    fb_set_power_mode(FB_POWER_MODE_AUTO);
    
    if (fb_should_prefer_cpu()) {
        printf("→ System recommends using CPU (power saving mode)\n");
    } else {
        printf("→ System allows GPU usage (AC power available)\n");
    }
    
    // ========================================================================
    // STEP 4: Device Selection - Different Strategies
    // ========================================================================
    
    print_separator("STEP 4: Device Selection Strategies");
    
    // Test FASTEST strategy
    printf("Strategy: FASTEST\n");
    fb_set_dispatch_strategy(FB_STRATEGY_FASTEST);
    fb_compute_device_t* fastest = fb_select_device_auto();
    if (fastest) {
        printf("  Selected: ");
        fb_print_device_summary(fastest);
    }
    printf("\n");
    
    // Test LOAD_BALANCED strategy
    printf("Strategy: LOAD_BALANCED\n");
    fb_set_dispatch_strategy(FB_STRATEGY_LOAD_BALANCED);
    fb_compute_device_t* balanced = fb_select_device_auto();
    if (balanced) {
        printf("  Selected: ");
        fb_print_device_summary(balanced);
    }
    printf("\n");
    
    // Test POWER_EFFICIENT strategy
    printf("Strategy: POWER_EFFICIENT\n");
    fb_set_dispatch_strategy(FB_STRATEGY_POWER_EFFICIENT);
    fb_compute_device_t* efficient = fb_select_device_auto();
    if (efficient) {
        printf("  Selected: ");
        fb_print_device_summary(efficient);
    }
    printf("\n");
    
    // Test ADAPTIVE strategy (multi-factor scoring)
    printf("Strategy: ADAPTIVE (Multi-Factor Scoring)\n");
    fb_set_dispatch_strategy(FB_STRATEGY_ADAPTIVE);
    fb_compute_device_t* adaptive = fb_select_device_auto();
    if (adaptive) {
        printf("  Selected: ");
        fb_print_device_summary(adaptive);
    }
    printf("\n");
    
    // ========================================================================
    // STEP 5: Data Locality Awareness
    // ========================================================================
    
    print_separator("STEP 5: Data Locality Tracking");
    
    // Simulate some data allocations
    float* host_data = (float*)malloc(1024 * 1024 * sizeof(float));
    float* device_data = (float*)malloc(2048 * 2048 * sizeof(float));
    
    // Register data locations
    fb_data_location_t host_loc = { FB_DATA_HOST, 0 };
    fb_data_location_t gpu_loc = { FB_DATA_DEVICE, 0 };
    
    fb_data_register(host_data, host_loc, 1024 * 1024 * sizeof(float));
    printf("✓ Registered host_data (4 MB) on CPU\n");
    
    if (gpu_count > 0) {
        fb_data_register(device_data, gpu_loc, 2048 * 2048 * sizeof(float));
        printf("✓ Registered device_data (16 MB) on GPU 0\n");
    }
    
    fb_data_print_stats();
    
    // Test data locality-based selection
    printf("\nTesting DATA_LOCALITY strategy:\n");
    const void* data_ptrs[] = { host_data, device_data };
    fb_set_dispatch_strategy(FB_STRATEGY_DATA_LOCALITY);
    fb_compute_device_t* locality_dev = fb_select_device(
        FB_STRATEGY_DATA_LOCALITY, FB_PRECISION_FP32, 
        data_ptrs, 2);
    
    if (locality_dev) {
        printf("Selected device based on data location: ");
        fb_print_device_summary(locality_dev);
    }
    
    // Estimate transfer costs
    printf("\nTransfer cost estimates:\n");
    fb_compute_device_t* cpu = fb_get_cpu_device();
    if (cpu && gpu_count > 0) {
        fb_compute_device_t* gpu = fb_get_gpu_device(0);
        
        double host_to_gpu = fb_estimate_transfer_cost(host_data, gpu);
        double device_to_cpu = fb_estimate_transfer_cost(device_data, cpu);
        
        printf("  host_data → GPU: %.3f ms\n", host_to_gpu * 1000.0);
        printf("  device_data → CPU: %.3f ms\n", device_to_cpu * 1000.0);
    }
    
    // ========================================================================
    // STEP 6: Operation Cost Estimation
    // ========================================================================
    
    print_separator("STEP 6: Operation Cost Estimation");
    
    // Estimate GEMM operation
    size_t m = 4096, n = 4096, k = 4096;
    double gemm_gflops = fb_estimate_operation_cost(FB_OP_GEMM, m, n, k, 
                                                     FB_PRECISION_FP32);
    
    printf("GEMM(%zu, %zu, %zu) requires %.2f GFLOPS\n", m, n, k, gemm_gflops);
    
    printf("\nEstimated execution time on each device:\n");
    for (uint32_t i = 0; i < device_count; i++) {
        fb_compute_device_t* dev = fb_get_device(i);
        double exec_time = fb_estimate_execution_time(dev, gemm_gflops);
        printf("  Device %u (%s): %.3f ms\n", i, dev->name, exec_time * 1000.0);
    }
    
    // ========================================================================
    // STEP 7: Custom Scoring Weights
    // ========================================================================
    
    print_separator("STEP 7: Custom Scoring Configuration");
    
    // Create custom weights favoring speed
    fb_scoring_weights_t custom_weights;
    custom_weights.speed_weight = 0.7f;
    custom_weights.load_weight = 0.1f;
    custom_weights.locality_weight = 0.1f;
    custom_weights.power_weight = 0.05f;
    custom_weights.thermal_weight = 0.05f;
    
    fb_set_scoring_weights(&custom_weights);
    
    printf("Custom weights (speed-focused):\n");
    printf("  Speed: %.2f\n", custom_weights.speed_weight);
    printf("  Load: %.2f\n", custom_weights.load_weight);
    printf("  Locality: %.2f\n", custom_weights.locality_weight);
    printf("  Power: %.2f\n", custom_weights.power_weight);
    printf("  Thermal: %.2f\n", custom_weights.thermal_weight);
    
    fb_compute_device_t* custom_dev = fb_select_device_scored(
        FB_PRECISION_FP32, NULL, 0, &custom_weights);
    
    if (custom_dev) {
        printf("\nSelected with custom weights: ");
        fb_print_device_summary(custom_dev);
    }
    
    // ========================================================================
    // STEP 8: Manager Statistics
    // ========================================================================
    
    print_separator("STEP 8: System Statistics");
    
    fb_print_manager_stats();
    
    // ========================================================================
    // STEP 9: Simulated Workload
    // ========================================================================
    
    print_separator("STEP 9: Simulated Workload Dispatch");
    
    printf("Simulating 5 GEMM operations with adaptive dispatch...\n\n");
    fb_set_dispatch_strategy(FB_STRATEGY_ADAPTIVE);
    
    for (int i = 0; i < 5; i++) {
        fb_compute_device_t* selected = fb_select_device_auto();
        
        if (selected) {
            printf("Operation %d → %s (Load: %.1f%%)\n", 
                   i + 1, selected->name, selected->current_load * 100.0f);
            
            // Simulate execution
            double exec_time = 0.05; // 50ms
            fb_record_execution(selected, FB_OP_GEMM, exec_time);
        }
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    print_separator("Cleanup");
    
    free(host_data);
    free(device_data);
    
    fb_compute_manager_shutdown();
    
    printf("✓ System shutdown complete\n\n");
    printf("Test completed successfully!\n");
    
    return 0;
}
