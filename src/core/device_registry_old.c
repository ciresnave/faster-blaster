/**
 * @file device_registry.c
 * @brief Device Registry Implementation
 * 
 * Manages all compute devices (CPUs and GPUs) in the system
 */

#include "faster-blaster/device_registry.h"
#include "device_detection/cpu_detect.h"
#include "device_detection/gpu_detect.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Global registry
static fb_device_registry_t g_registry = {0};
static bool g_registry_initialized = false;

// ============================================================================
// Registry Initialization
// ============================================================================

int fb_registry_init(fb_discovery_flags_t flags) {
    if (g_registry_initialized) {
        return g_registry.num_devices; // Already initialized, return device count
    }
    
    (void)flags; // TODO: Use flags to control which devices to discover
    
    memset(&g_registry, 0, sizeof(g_registry));
    
    // Detect CPU
    fb_cpu_info_t cpu_info;
    if (fb_detect_cpu(&cpu_info) == 0) {
        // Create CPU device entry
        fb_compute_device_t* cpu_dev = &g_registry.devices[g_registry.num_devices];
        cpu_dev->type = FB_DEVICE_CPU;
        cpu_dev->device_id = 0;
        cpu_dev->vendor_id = cpu_info.vendor;
        
        snprintf(cpu_dev->name, sizeof(cpu_dev->name), "%s", cpu_info.brand_string);
        
        // Capabilities
        cpu_dev->capabilities.fp32 = true;
        cpu_dev->capabilities.fp64 = true;
        cpu_dev->capabilities.fp16 = cpu_info.simd.x86.avx512_fp16 || 
                                      cpu_info.simd.arm.fp16;
        cpu_dev->capabilities.int8 = true;
        
        // Performance
        cpu_dev->performance.fp32_gflops = cpu_info.estimated_gflops_sp;
        cpu_dev->performance.fp64_gflops = cpu_info.estimated_gflops_dp;
        cpu_dev->performance.memory_bandwidth_gbps = 50.0; // Typical DDR4/DDR5
        
        // Memory
        cpu_dev->memory.total_bytes = 0; // TODO: System RAM detection
        cpu_dev->memory.free_bytes = 0;
        cpu_dev->memory.is_unified = true;
        
        // Topology
        cpu_dev->topology.compute_units = cpu_info.topology.physical_cores;
        cpu_dev->topology.threads_per_cu = cpu_info.topology.threads_per_core;
        
        // State
        cpu_dev->state = FB_DEVICE_AVAILABLE;
        cpu_dev->current_load = 0.0f;
        
        // Store detailed CPU info
        g_registry.cpu_info = cpu_info;
        g_registry.has_cpu = true;
        
        g_registry.num_devices++;
    }
    
    // Detect GPUs
    fb_gpu_info_t gpus[FB_MAX_DEVICES];
    uint32_t num_gpus = 0;
    
    if (fb_detect_gpus(gpus, FB_MAX_DEVICES - g_registry.num_devices, &num_gpus) == 0) {
        for (uint32_t i = 0; i < num_gpus; i++) {
            fb_compute_device_t* gpu_dev = &g_registry.devices[g_registry.num_devices];
            fb_gpu_info_t* gpu_info = &gpus[i];
            
            gpu_dev->type = FB_DEVICE_GPU;
            gpu_dev->device_id = gpu_info->device_id;
            gpu_dev->vendor_id = gpu_info->vendor;
            
            strncpy(gpu_dev->name, gpu_info->name, sizeof(gpu_dev->name) - 1);
            strncpy(gpu_dev->pci_bus_id, gpu_info->pci_bus_id, 
                    sizeof(gpu_dev->pci_bus_id) - 1);
            
            // Capabilities
            gpu_dev->capabilities.fp32 = gpu_info->compute.fp32;
            gpu_dev->capabilities.fp64 = gpu_info->compute.fp64;
            gpu_dev->capabilities.fp16 = gpu_info->compute.fp16;
            gpu_dev->capabilities.bf16 = gpu_info->compute.bf16;
            gpu_dev->capabilities.int8 = gpu_info->compute.int8;
            gpu_dev->capabilities.int4 = gpu_info->compute.int4;
            gpu_dev->capabilities.has_tensor_cores = gpu_info->compute.has_tensor_cores;
            gpu_dev->capabilities.has_unified_memory = gpu_info->compute.has_unified_memory;
            
            // Performance
            gpu_dev->performance.fp32_gflops = gpu_info->fp32_tflops * 1000.0;
            gpu_dev->performance.fp64_gflops = gpu_info->fp64_tflops * 1000.0;
            gpu_dev->performance.fp16_gflops = gpu_info->fp16_tflops * 1000.0;
            gpu_dev->performance.tensor_gflops = gpu_info->tensor_tflops * 1000.0;
            gpu_dev->performance.memory_bandwidth_gbps = gpu_info->memory_bandwidth_gbps;
            
            // Memory
            gpu_dev->memory.total_bytes = gpu_info->total_memory_bytes;
            gpu_dev->memory.free_bytes = gpu_info->free_memory_bytes;
            gpu_dev->memory.is_unified = (gpu_info->vendor == GPU_VENDOR_APPLE);
            
            // Topology
            gpu_dev->topology.compute_units = gpu_info->compute_units;
            gpu_dev->topology.cores_per_cu = gpu_info->cores_per_cu;
            
            // Power
            gpu_dev->power.tdp_watts = gpu_info->tdp_watts;
            gpu_dev->power.current_watts = gpu_info->current_power_watts;
            
            // Thermal
            gpu_dev->thermal.current_temp_c = gpu_info->current_temp_c;
            gpu_dev->thermal.max_temp_c = gpu_info->max_temp_c;
            
            // State
            gpu_dev->state = FB_DEVICE_AVAILABLE;
            gpu_dev->current_load = gpu_info->gpu_utilization_percent / 100.0f;
            
            // Store detailed GPU info
            g_registry.gpu_info[i] = *gpu_info;
            
            g_registry.num_devices++;
            g_registry.num_gpus++;
        }
    }
    
    g_registry_initialized = true;
    
    return 0;
}

void fb_registry_shutdown(void) {
    if (!g_registry_initialized) return;
    
    memset(&g_registry, 0, sizeof(g_registry));
    g_registry_initialized = false;
}

// ============================================================================
// Device Query
// ============================================================================

uint32_t fb_get_device_count(void) {
    if (!g_registry_initialized) {
        fb_registry_init(FB_DISCOVERY_ALL);
    }
    return g_registry.num_devices;
}

fb_compute_device_t* fb_get_device(uint32_t index) {
    if (!g_registry_initialized) {
        fb_registry_init(FB_DISCOVERY_ALL);
    }
    
    if (index >= g_registry.num_devices) {
        return NULL;
    }
    
    return &g_registry.devices[index];
}

fb_compute_device_t* fb_get_device_by_type(fb_device_type_t type, uint32_t index) {
    if (!g_registry_initialized) {
        fb_registry_init(FB_DISCOVERY_ALL);
    }
    
    uint32_t found = 0;
    for (uint32_t i = 0; i < g_registry.num_devices; i++) {
        if (g_registry.devices[i].type == type) {
            if (found == index) {
                return &g_registry.devices[i];
            }
            found++;
        }
    }
    
    return NULL;
}

fb_compute_device_t* fb_get_cpu_device(void) {
    return fb_get_device_by_type(FB_DEVICE_CPU, 0);
}

uint32_t fb_get_gpu_count(void) {
    if (!g_registry_initialized) {
        fb_registry_init(FB_DISCOVERY_ALL);
    }
    return g_registry.num_gpus;
}

fb_compute_device_t* fb_get_gpu_device(uint32_t gpu_index) {
    return fb_get_device_by_type(FB_DEVICE_GPU, gpu_index);
}

// ============================================================================
// Device Update
// ============================================================================

int fb_update_device_state(fb_compute_device_t* device) {
    if (!device) return -1;
    
    if (device->type == FB_DEVICE_GPU) {
        // Find corresponding GPU info
        for (uint32_t i = 0; i < g_registry.num_gpus; i++) {
            if (g_registry.gpu_info[i].device_id == device->device_id) {
                fb_gpu_info_t* gpu_info = &g_registry.gpu_info[i];
                
                // Update utilization
                fb_update_gpu_utilization(gpu_info);
                device->current_load = gpu_info->gpu_utilization_percent / 100.0f;
                
                // Update memory
                fb_update_gpu_memory(gpu_info);
                device->memory.free_bytes = gpu_info->free_memory_bytes;
                
                // Update temperature
                fb_update_gpu_temperature(gpu_info);
                device->thermal.current_temp_c = gpu_info->current_temp_c;
                
                // Update power
                fb_update_gpu_power(gpu_info);
                device->power.current_watts = gpu_info->current_power_watts;
                
                return 0;
            }
        }
    } else if (device->type == FB_DEVICE_CPU) {
        // TODO: Update CPU utilization, temperature
        // This requires platform-specific code
    }
    
    return -1;
}

int fb_update_all_devices(void) {
    fb_get_registry();
    
    for (uint32_t i = 0; i < g_registry.num_devices; i++) {
        fb_update_device_state(&g_registry.devices[i]);
    }
    
    return 0;
}

// ============================================================================
// Device State Management
// ============================================================================

int fb_set_device_state(fb_compute_device_t* device, fb_device_state_t state) {
    if (!device) return -1;
    
    device->state = state;
    return 0;
}

int fb_enable_device(fb_compute_device_t* device) {
    return fb_set_device_state(device, FB_DEVICE_AVAILABLE);
}

int fb_disable_device(fb_compute_device_t* device) {
    return fb_set_device_state(device, FB_DEVICE_DISABLED);
}

bool fb_is_device_available(const fb_compute_device_t* device) {
    return device && device->state == FB_DEVICE_AVAILABLE;
}

// ============================================================================
// Capability Queries
// ============================================================================

bool fb_device_supports_precision(const fb_compute_device_t* device, 
                                  fb_precision_t precision) {
    if (!device) return false;
    
    switch (precision) {
        case FB_PRECISION_FP64: return device->capabilities.fp64;
        case FB_PRECISION_FP32: return device->capabilities.fp32;
        case FB_PRECISION_FP16: return device->capabilities.fp16;
        case FB_PRECISION_BF16: return device->capabilities.bf16;
        case FB_PRECISION_INT8: return device->capabilities.int8;
        case FB_PRECISION_INT4: return device->capabilities.int4;
        default: return false;
    }
}

bool fb_device_has_tensor_cores(const fb_compute_device_t* device) {
    return device && device->capabilities.has_tensor_cores;
}

double fb_device_get_gflops(const fb_compute_device_t* device, 
                           fb_precision_t precision) {
    if (!device) return 0.0;
    
    switch (precision) {
        case FB_PRECISION_FP64: return device->performance.fp64_gflops;
        case FB_PRECISION_FP32: return device->performance.fp32_gflops;
        case FB_PRECISION_FP16: return device->performance.fp16_gflops;
        case FB_PRECISION_BF16: return device->performance.fp16_gflops; // Approx
        case FB_PRECISION_INT8: return device->performance.tensor_gflops;
        default: return device->performance.fp32_gflops;
    }
}

// ============================================================================
// Device Selection Helpers
// ============================================================================

fb_compute_device_t* fb_find_fastest_device(fb_precision_t precision) {
    fb_get_registry();
    
    fb_compute_device_t* fastest = NULL;
    double max_gflops = 0.0;
    
    for (uint32_t i = 0; i < g_registry.num_devices; i++) {
        fb_compute_device_t* device = &g_registry.devices[i];
        
        if (!fb_is_device_available(device)) continue;
        if (!fb_device_supports_precision(device, precision)) continue;
        
        double gflops = fb_device_get_gflops(device, precision);
        if (gflops > max_gflops) {
            max_gflops = gflops;
            fastest = device;
        }
    }
    
    return fastest;
}

fb_compute_device_t* fb_find_least_loaded_device(void) {
    fb_get_registry();
    
    // Update all device states first
    fb_update_all_devices();
    
    fb_compute_device_t* least_loaded = NULL;
    float min_load = 2.0f; // > 100%
    
    for (uint32_t i = 0; i < g_registry.num_devices; i++) {
        fb_compute_device_t* device = &g_registry.devices[i];
        
        if (!fb_is_device_available(device)) continue;
        
        if (device->current_load < min_load) {
            min_load = device->current_load;
            least_loaded = device;
        }
    }
    
    return least_loaded;
}

fb_compute_device_t* fb_find_device_with_data(const void* ptr) {
    // TODO: Query data tracker for pointer location
    // For now, assume data is on CPU
    return fb_get_cpu_device();
}

// ============================================================================
// Debug/Print Functions
// ============================================================================

void fb_print_devices(void) {
    fb_get_registry();
    
    printf("=== Compute Devices ===\n");
    printf("Total: %u devices (%u GPUs)\n\n", 
           g_registry.num_devices, g_registry.num_gpus);
    
    for (uint32_t i = 0; i < g_registry.num_devices; i++) {
        fb_compute_device_t* dev = &g_registry.devices[i];
        
        printf("[%u] %s: %s\n", i,
               dev->type == FB_DEVICE_CPU ? "CPU" : "GPU",
               dev->name);
        
        if (dev->type == FB_DEVICE_GPU) {
            printf("    PCI Bus: %s\n", dev->pci_bus_id);
        }
        
        printf("    Compute Units: %u (x%u threads) = %u total\n",
               dev->topology.compute_units,
               dev->topology.threads_per_cu,
               dev->topology.compute_units * dev->topology.threads_per_cu);
        
        printf("    Memory: %.1f GB (%.1f GB free)\n",
               dev->memory.total_bytes / (1024.0*1024.0*1024.0),
               dev->memory.free_bytes / (1024.0*1024.0*1024.0));
        
        printf("    Performance:\n");
        printf("      FP32: %.1f GFLOPS\n", dev->performance.fp32_gflops);
        printf("      FP64: %.1f GFLOPS\n", dev->performance.fp64_gflops);
        if (dev->performance.fp16_gflops > 0) {
            printf("      FP16: %.1f GFLOPS\n", dev->performance.fp16_gflops);
        }
        if (dev->performance.tensor_gflops > 0) {
            printf("      Tensor: %.1f GFLOPS\n", dev->performance.tensor_gflops);
        }
        printf("      Memory BW: %.1f GB/s\n", dev->performance.memory_bandwidth_gbps);
        
        if (dev->type == FB_DEVICE_GPU) {
            printf("    Power: %u W (TDP: %u W)\n", 
                   dev->power.current_watts, dev->power.tdp_watts);
            printf("    Temperature: %u°C\n", dev->thermal.current_temp_c);
            printf("    Utilization: %.1f%%\n", dev->current_load * 100.0f);
        }
        
        printf("    Capabilities:");
        if (dev->capabilities.fp64) printf(" FP64");
        if (dev->capabilities.fp32) printf(" FP32");
        if (dev->capabilities.fp16) printf(" FP16");
        if (dev->capabilities.bf16) printf(" BF16");
        if (dev->capabilities.int8) printf(" INT8");
        if (dev->capabilities.has_tensor_cores) printf(" TensorCores");
        printf("\n");
        
        printf("    State: %s\n", 
               dev->state == FB_DEVICE_AVAILABLE ? "Available" :
               dev->state == FB_DEVICE_BUSY ? "Busy" :
               dev->state == FB_DEVICE_ERROR ? "Error" : "Disabled");
        
        printf("\n");
    }
}

void fb_print_device_summary(const fb_compute_device_t* device) {
    if (!device) return;
    
    printf("%s: %s (%.1f GFLOPS FP32, %.1f%% load)\n",
           device->type == FB_DEVICE_CPU ? "CPU" : "GPU",
           device->name,
           device->performance.fp32_gflops,
           device->current_load * 100.0f);
}
