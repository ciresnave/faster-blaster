/**
 * @file gpu_detect.c
 * @brief GPU Detection Implementation - Main Dispatcher
 * 
 * This file coordinates GPU detection across different platforms.
 * Platform-specific detection is in separate files to avoid header conflicts:
 * - cuda_detect.c: NVIDIA CUDA/cuBLAS
 * - hip_detect.c: AMD ROCm/HIP
 * - (Intel and Metal detection remain here as they don't conflict)
 */

#include "device_detection/gpu_detect.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Platform-specific detectors are implemented in separate files:
// - fb_detect_cuda_gpus() -> cuda_detect.c
// - fb_detect_rocm_gpus() -> hip_detect.c

// Forward declarations for platform-specific functions
#ifdef FB_ENABLE_CUDA
extern int fb_detect_cuda_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus);
extern int fb_update_nvidia_utilization(fb_gpu_info_t* gpu_info);
extern int fb_update_nvidia_temperature(fb_gpu_info_t* gpu_info);
extern int fb_update_nvidia_power(fb_gpu_info_t* gpu_info);
extern int fb_update_nvidia_memory(fb_gpu_info_t* gpu_info);
#endif

#ifdef FB_ENABLE_HIP
extern int fb_detect_rocm_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus);
extern int fb_update_amd_memory(fb_gpu_info_t* gpu_info);
#else
// Stubs when HIP not available (HIP compiler not found)
int fb_detect_rocm_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    (void)gpus; (void)max_gpus;
    *num_gpus = 0;
    return -1;
}
int fb_update_amd_memory(fb_gpu_info_t* gpu_info) {
    (void)gpu_info;
    return -1;
}
#endif

#ifdef FB_ENABLE_OPENCL
extern int fb_detect_opencl_gpus(fb_gpu_info_t* gpus, int max_gpus, int* count_out);
extern int fb_update_opencl_memory(fb_gpu_info_t* gpu);
#endif

// Intel Level Zero
#ifdef FB_ENABLE_SYCL
#include <level_zero/ze_api.h>
#endif

// Apple Metal
#ifdef __APPLE__
#include <Metal/Metal.h>
#endif

// ============================================================================
// Intel oneAPI/Level Zero Detection
// ============================================================================

#ifdef FB_ENABLE_SYCL

int fb_detect_intel_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    *num_gpus = 0;
    
    // Initialize Level Zero
    if (zeInit(0) != ZE_RESULT_SUCCESS) {
        return -1;
    }
    
    // Get drivers
    uint32_t driver_count = 0;
    zeDriverGet(&driver_count, NULL);
    if (driver_count == 0) return -1;
    
    ze_driver_handle_t* drivers = (ze_driver_handle_t*)malloc(
        driver_count * sizeof(ze_driver_handle_t));
    zeDriverGet(&driver_count, drivers);
    
    for (uint32_t d = 0; d < driver_count; d++) {
        uint32_t device_count = 0;
        zeDeviceGet(drivers[d], &device_count, NULL);
        
        if (device_count == 0) continue;
        
        ze_device_handle_t* devices = (ze_device_handle_t*)malloc(
            device_count * sizeof(ze_device_handle_t));
        zeDeviceGet(drivers[d], &device_count, devices);
        
        for (uint32_t i = 0; i < device_count && *num_gpus < max_gpus; i++) {
            ze_device_properties_t props;
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            zeDeviceGetProperties(devices[i], &props);
            
            // Only GPU devices
            if (props.type != ZE_DEVICE_TYPE_GPU) continue;
            
            fb_gpu_info_t* gpu = &gpus[*num_gpus];
            memset(gpu, 0, sizeof(*gpu));
            
            gpu->vendor = GPU_VENDOR_INTEL;
            gpu->device_id = *num_gpus;
            strncpy(gpu->name, props.name, sizeof(gpu->name) - 1);
            
            // Compute units
            gpu->compute_units = props.numSlices * props.numSubslicesPerSlice;
            gpu->cores_per_cu = props.numEUsPerSubslice;
            gpu->total_cores = gpu->compute_units * gpu->cores_per_cu;
            
            // Memory properties
            ze_device_memory_properties_t mem_props;
            uint32_t mem_count = 1;
            zeDeviceGetMemoryProperties(devices[i], &mem_count, &mem_props);
            gpu->total_memory_bytes = mem_props.totalSize;
            
            // Capabilities
            gpu->compute.fp64 = true;
            gpu->compute.fp32 = true;
            gpu->compute.fp16 = true;
            gpu->compute.has_xmx = true; // Intel XMX (Matrix Extensions)
            
            // Architecture detection (simplified)
            if (strstr(props.name, "Arc")) {
                gpu->architecture = GPU_ARCH_INTEL_XE_HPG;
            } else if (strstr(props.name, "Ponte Vecchio")) {
                gpu->architecture = GPU_ARCH_INTEL_XE_HPC;
            } else {
                gpu->architecture = GPU_ARCH_INTEL_XE;
            }
            
            gpu->sycl_available = true;
            
            fb_estimate_gpu_tflops(gpu);
            
            (*num_gpus)++;
        }
        
        free(devices);
    }
    
    free(drivers);
    return 0;
}

#else

int fb_detect_intel_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    (void)gpus;
    (void)max_gpus;
    *num_gpus = 0;
    return -1;
}

#endif // FB_ENABLE_SYCL

// ============================================================================
// Apple Metal Detection
// ============================================================================

#ifdef __APPLE__

int fb_detect_metal_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    *num_gpus = 0;
    
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        
        for (id<MTLDevice> device in devices) {
            if (*num_gpus >= max_gpus) break;
            
            fb_gpu_info_t* gpu = &gpus[*num_gpus];
            memset(gpu, 0, sizeof(*gpu));
            
            gpu->vendor = GPU_VENDOR_APPLE;
            gpu->device_id = *num_gpus;
            
            const char* name = [device.name UTF8String];
            strncpy(gpu->name, name, sizeof(gpu->name) - 1);
            
            // Memory
            gpu->total_memory_bytes = device.recommendedMaxWorkingSetSize;
            gpu->memory_type = GPU_MEMORY_UNIFIED;
            
            // Capabilities
            gpu->compute.fp32 = true;
            gpu->compute.fp16 = true;
            gpu->compute.has_unified_memory = true;
            
            // Architecture from name
            if (strstr(name, "M4")) {
                gpu->architecture = GPU_ARCH_APPLE_M4;
            } else if (strstr(name, "M3")) {
                gpu->architecture = GPU_ARCH_APPLE_M3;
            } else if (strstr(name, "M2")) {
                gpu->architecture = GPU_ARCH_APPLE_M2;
            } else if (strstr(name, "M1")) {
                gpu->architecture = GPU_ARCH_APPLE_M1;
            }
            
            gpu->metal_available = true;
            
            fb_estimate_gpu_tflops(gpu);
            
            (*num_gpus)++;
        }
    }
    
    return 0;
}

#else

int fb_detect_metal_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    (void)gpus;
    (void)max_gpus;
    *num_gpus = 0;
    return -1;
}

#endif // __APPLE__

// ============================================================================
// Main Detection Function
// ============================================================================

int fb_detect_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    *num_gpus = 0;
    
    uint32_t cuda_count = 0, rocm_count = 0, intel_count = 0, metal_count = 0, opencl_count = 0;
    
    // Try CUDA
#ifdef FB_ENABLE_CUDA
    fb_detect_cuda_gpus(gpus + *num_gpus, max_gpus - *num_gpus, &cuda_count);
    *num_gpus += cuda_count;
#endif
    
    // Try ROCm
    fb_detect_rocm_gpus(gpus + *num_gpus, max_gpus - *num_gpus, &rocm_count);
    *num_gpus += rocm_count;
    
    // Try Intel
    fb_detect_intel_gpus(gpus + *num_gpus, max_gpus - *num_gpus, &intel_count);
    *num_gpus += intel_count;
    
    // Try Metal
    fb_detect_metal_gpus(gpus + *num_gpus, max_gpus - *num_gpus, &metal_count);
    *num_gpus += metal_count;
    
    // Try OpenCL (can detect devices already found by vendor APIs + additional devices)
    printf("[DEBUG] About to call OpenCL detection, FB_ENABLE_OPENCL=%s\n", 
#ifdef FB_ENABLE_OPENCL
        "defined"
#else
        "NOT defined"
#endif
    );
#ifdef FB_ENABLE_OPENCL
    int opencl_count_int = 0;
    fb_detect_opencl_gpus(gpus + *num_gpus, (int)(max_gpus - *num_gpus), &opencl_count_int);
    printf("[DEBUG] OpenCL detection returned, count=%d\n", opencl_count_int);
    opencl_count = (uint32_t)opencl_count_int;
    *num_gpus += opencl_count;
#endif
    
    return (*num_gpus > 0) ? 0 : -1;
}

// ============================================================================
// TFLOPS Estimation
// ============================================================================

void fb_estimate_gpu_tflops(fb_gpu_info_t* gpu_info) {
    // Base calculation: Cores * Clock * 2 (FMA)
    double base_gflops = gpu_info->total_cores * 
                         (gpu_info->boost_clock_mhz / 1000.0) * 2.0;
    
    gpu_info->fp32_tflops = base_gflops / 1000.0;
    
    // FP64: typically 1/2 for AMD, 1/32 for NVIDIA consumer, 1/2 for datacenter
    if (gpu_info->vendor == GPU_VENDOR_NVIDIA) {
        if (gpu_info->architecture == GPU_ARCH_NVIDIA_VOLTA ||
            gpu_info->architecture == GPU_ARCH_NVIDIA_AMPERE ||
            gpu_info->architecture == GPU_ARCH_NVIDIA_HOPPER) {
            // Datacenter GPUs
            gpu_info->fp64_tflops = gpu_info->fp32_tflops / 2.0;
        } else {
            // Consumer GPUs
            gpu_info->fp64_tflops = gpu_info->fp32_tflops / 32.0;
        }
    } else {
        // AMD, Intel: typically 1/2
        gpu_info->fp64_tflops = gpu_info->fp32_tflops / 2.0;
    }
    
    // FP16: 2x FP32 (or more with specialized hardware)
    gpu_info->fp16_tflops = gpu_info->fp32_tflops * 2.0;
    
    // Tensor cores: can be 8-16x FP32 for specific operations
    if (gpu_info->compute.has_tensor_cores || 
        gpu_info->compute.has_matrix_cores ||
        gpu_info->compute.has_xmx) {
        gpu_info->tensor_tflops = gpu_info->fp32_tflops * 8.0;
    }
}

// ============================================================================
// Monitoring Functions
// ============================================================================

int fb_update_gpu_utilization(fb_gpu_info_t* gpu_info) {
#ifdef FB_ENABLE_CUDA
    if (gpu_info->vendor == GPU_VENDOR_NVIDIA && gpu_info->cuda_available) {
        return fb_update_nvidia_utilization(gpu_info);
    }
#endif
    return -1;
}

int fb_update_gpu_temperature(fb_gpu_info_t* gpu_info) {
#ifdef FB_ENABLE_CUDA
    if (gpu_info->vendor == GPU_VENDOR_NVIDIA && gpu_info->cuda_available) {
        return fb_update_nvidia_temperature(gpu_info);
    }
#endif
    return -1;
}

int fb_update_gpu_power(fb_gpu_info_t* gpu_info) {
#ifdef FB_ENABLE_CUDA
    if (gpu_info->vendor == GPU_VENDOR_NVIDIA && gpu_info->cuda_available) {
        return fb_update_nvidia_power(gpu_info);
    }
#endif
    return -1;
}

int fb_update_gpu_memory(fb_gpu_info_t* gpu_info) {
#ifdef FB_ENABLE_CUDA
    if (gpu_info->vendor == GPU_VENDOR_NVIDIA && gpu_info->cuda_available) {
        return fb_update_nvidia_memory(gpu_info);
    }
#endif
#ifdef FB_ENABLE_HIP
    if (gpu_info->vendor == GPU_VENDOR_AMD && gpu_info->hip_available) {
        return fb_update_amd_memory(gpu_info);
    }
#endif
    return -1;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool fb_gpu_supports_compute_capability(const fb_gpu_info_t* gpu_info, 
                                        int major, int minor) {
    if (gpu_info->vendor != GPU_VENDOR_NVIDIA) return false;
    
    if (gpu_info->compute.cuda_major > major) return true;
    if (gpu_info->compute.cuda_major == major && 
        gpu_info->compute.cuda_minor >= minor) return true;
    
    return false;
}

void fb_print_gpu_info(const fb_gpu_info_t* gpu_info) {
    printf("GPU Information:\n");
    printf("  Name: %s\n", gpu_info->name);
    printf("  Vendor: %d\n", gpu_info->vendor);
    printf("  Device ID: %d\n", gpu_info->device_id);
    printf("  PCI Bus ID: %s\n", gpu_info->pci_bus_id);
    
    printf("\nCompute:\n");
    if (gpu_info->vendor == GPU_VENDOR_NVIDIA) {
        printf("  CUDA Compute: %d.%d\n", 
               gpu_info->compute.cuda_major, gpu_info->compute.cuda_minor);
    }
    printf("  Compute Units: %u\n", gpu_info->compute_units);
    printf("  Total Cores: %u\n", gpu_info->total_cores);
    printf("  Tensor Cores: %s\n", gpu_info->compute.has_tensor_cores ? "Yes" : "No");
    
    printf("\nMemory:\n");
    printf("  Total: %.1f GB\n", gpu_info->total_memory_bytes / (1024.0*1024.0*1024.0));
    printf("  Free: %.1f GB\n", gpu_info->free_memory_bytes / (1024.0*1024.0*1024.0));
    printf("  Bandwidth: %.1f GB/s\n", gpu_info->memory_bandwidth_gbps);
    
    printf("\nPerformance:\n");
    printf("  FP32: %.1f TFLOPS\n", gpu_info->fp32_tflops);
    printf("  FP64: %.1f TFLOPS\n", gpu_info->fp64_tflops);
    printf("  FP16: %.1f TFLOPS\n", gpu_info->fp16_tflops);
    if (gpu_info->tensor_tflops > 0) {
        printf("  Tensor: %.1f TFLOPS\n", gpu_info->tensor_tflops);
    }
    
    printf("\nMonitoring:\n");
    printf("  Utilization: %u%%\n", gpu_info->gpu_utilization_percent);
    printf("  Temperature: %u°C\n", gpu_info->current_temp_c);
    printf("  Power: %u W (limit: %u W)\n", 
           gpu_info->current_power_watts, gpu_info->power_limit_watts);
}
