/**
 * @file cuda_detect.c
 * @brief NVIDIA CUDA GPU Detection
 * 
 * Separated from gpu_detect.c to avoid header conflicts with HIP.
 * This file can only be compiled when FB_ENABLE_CUDA is defined.
 */

#include "device_detection/gpu_detect.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef FB_ENABLE_CUDA

#include <cuda_runtime.h>
#include <driver_types.h>
#include <nvml.h>

static fb_gpu_arch_t cuda_arch_from_compute_capability(int major, int minor) {
    if (major == 2) return GPU_ARCH_NVIDIA_FERMI;
    if (major == 3) return GPU_ARCH_NVIDIA_KEPLER;
    if (major == 5) return GPU_ARCH_NVIDIA_MAXWELL;
    if (major == 6) return GPU_ARCH_NVIDIA_PASCAL;
    if (major == 7 && minor == 0) return GPU_ARCH_NVIDIA_VOLTA;
    if (major == 7 && minor == 5) return GPU_ARCH_NVIDIA_TURING;
    if (major == 8 && minor == 0) return GPU_ARCH_NVIDIA_AMPERE;
    if (major == 8 && minor == 6) return GPU_ARCH_NVIDIA_AMPERE;
    if (major == 8 && minor == 9) return GPU_ARCH_NVIDIA_ADA_LOVELACE;
    if (major == 9 && minor == 0) return GPU_ARCH_NVIDIA_HOPPER;
    if (major >= 10) return GPU_ARCH_NVIDIA_BLACKWELL;
    return GPU_ARCH_UNKNOWN;
}

int fb_detect_cuda_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    *num_gpus = 0;
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        return -1; // No CUDA devices or CUDA not available
    }
    
    // Initialize NVML for detailed monitoring
    bool nvml_available = false;
    if (nvmlInit() == NVML_SUCCESS) {
        nvml_available = true;
    }
    
    for (int i = 0; i < device_count && i < (int)max_gpus; i++) {
        struct cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            continue;
        }
        
        fb_gpu_info_t* gpu = &gpus[*num_gpus];
        memset(gpu, 0, sizeof(*gpu));
        
        // Basic info
        gpu->vendor = GPU_VENDOR_NVIDIA;
        gpu->device_id = i;
        strncpy(gpu->name, prop.name, sizeof(gpu->name) - 1);
        
        // Architecture from compute capability
        gpu->architecture = cuda_arch_from_compute_capability(
            prop.major, prop.minor);
        
        // Compute capabilities
        gpu->compute.cuda_major = prop.major;
        gpu->compute.cuda_minor = prop.minor;
        gpu->compute.fp64 = true;
        gpu->compute.fp32 = true;
        gpu->compute.fp16 = (prop.major >= 6); // Pascal+
        gpu->compute.bf16 = (prop.major >= 8); // Ampere+
        gpu->compute.int8 = (prop.major >= 6);
        gpu->compute.has_tensor_cores = (prop.major >= 7); // Volta+
        gpu->compute.has_unified_memory = (prop.major >= 3);
        gpu->compute.has_peer_access = (prop.major >= 2);
        gpu->compute.has_ecc = prop.ECCEnabled;
        
        // Compute units
        gpu->compute_units = prop.multiProcessorCount;
        gpu->cores_per_cu = 64; // Typical for modern NVIDIA GPUs
        if (prop.major == 8 && prop.minor == 6) gpu->cores_per_cu = 128; // Ada
        if (prop.major >= 9) gpu->cores_per_cu = 128; // Hopper+
        gpu->total_cores = gpu->compute_units * gpu->cores_per_cu;
        
        // Clocks (clockRate and memoryClockRate deprecated in CUDA 13+)
        int clock_khz = 0, mem_clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, i);
        cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, i);
        gpu->base_clock_mhz = clock_khz / 1000;
        gpu->boost_clock_mhz = clock_khz / 1000; // Base approximation
        gpu->memory_clock_mhz = mem_clock_khz / 1000;
        
        // Memory
        gpu->total_memory_bytes = prop.totalGlobalMem;
        gpu->l2_cache_bytes = prop.l2CacheSize;
        gpu->memory_bus_width = prop.memoryBusWidth;
        
        // Memory type estimation
        if (strstr(prop.name, "H100") || strstr(prop.name, "A100")) {
            gpu->memory_type = GPU_MEMORY_HBM2E;
        } else if (prop.major == 8 && prop.minor == 9) {
            gpu->memory_type = GPU_MEMORY_GDDR6X; // Ada (RTX 40 series)
        } else if (prop.major >= 8) {
            gpu->memory_type = GPU_MEMORY_GDDR6;
        } else {
            gpu->memory_type = GPU_MEMORY_GDDR5;
        }
        
        // Bandwidth calculation
        // Bandwidth (GB/s) = (Memory Clock MHz * 2) * (Bus Width / 8) / 1000
        gpu->memory_bandwidth_gbps = 
            (gpu->memory_clock_mhz * 2.0 * gpu->memory_bus_width / 8.0) / 1000.0;
        
        // PCIe bandwidth (estimate)
        gpu->pcie_bandwidth_gbps = 16.0; // PCIe 4.0 x16 = 32 GB/s bidirectional
        
        // PCI Bus ID
        snprintf(gpu->pci_bus_id, sizeof(gpu->pci_bus_id),
                 "%04x:%02x:%02x.%x",
                 prop.pciDomainID, prop.pciBusID, 
                 prop.pciDeviceID, 0);
        
        // Runtime availability
        gpu->cuda_available = true;
        
        // NVML-based monitoring (if available)
        if (nvml_available) {
            nvmlDevice_t nvml_dev;
            if (nvmlDeviceGetHandleByIndex(i, &nvml_dev) == NVML_SUCCESS) {
                // Power
                unsigned int power_mw;
                if (nvmlDeviceGetPowerUsage(nvml_dev, &power_mw) == NVML_SUCCESS) {
                    gpu->current_power_watts = power_mw / 1000;
                }
                
                unsigned int power_limit_mw;
                if (nvmlDeviceGetPowerManagementLimit(nvml_dev, &power_limit_mw) == NVML_SUCCESS) {
                    gpu->power_limit_watts = power_limit_mw / 1000;
                    gpu->tdp_watts = power_limit_mw / 1000;
                }
                
                // Temperature
                unsigned int temp;
                if (nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
                    gpu->current_temp_c = temp;
                }
                
                // Max temperature
                unsigned int max_temp;
                if (nvmlDeviceGetTemperatureThreshold(nvml_dev, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, &max_temp) == NVML_SUCCESS) {
                    gpu->max_temp_c = max_temp;
                }
                
                // Utilization
                nvmlUtilization_t util;
                if (nvmlDeviceGetUtilizationRates(nvml_dev, &util) == NVML_SUCCESS) {
                    gpu->gpu_utilization_percent = util.gpu;
                    gpu->memory_utilization_percent = util.memory;
                }
                
                // Memory info
                nvmlMemory_t mem_info;
                if (nvmlDeviceGetMemoryInfo(nvml_dev, &mem_info) == NVML_SUCCESS) {
                    gpu->free_memory_bytes = mem_info.free;
                }
            }
        } else {
            // Fallback: use cudaMemGetInfo for free memory
            size_t free_mem, total_mem;
            cudaSetDevice(i);
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                gpu->free_memory_bytes = free_mem;
            }
        }
        
        // Estimate TFLOPS
        fb_estimate_gpu_tflops(gpu);
        
        (*num_gpus)++;
    }
    
    if (nvml_available) {
        nvmlShutdown();
    }
    
    return 0;
}

// ============================================================================
// NVIDIA Monitoring Functions
// ============================================================================

int fb_update_nvidia_utilization(fb_gpu_info_t* gpu_info) {
    if (nvmlInit() == NVML_SUCCESS) {
        nvmlDevice_t dev;
        if (nvmlDeviceGetHandleByIndex(gpu_info->device_id, &dev) == NVML_SUCCESS) {
            nvmlUtilization_t util;
            if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
                gpu_info->gpu_utilization_percent = util.gpu;
                gpu_info->memory_utilization_percent = util.memory;
            }
        }
        nvmlShutdown();
        return 0;
    }
    return -1;
}

int fb_update_nvidia_temperature(fb_gpu_info_t* gpu_info) {
    if (nvmlInit() == NVML_SUCCESS) {
        nvmlDevice_t dev;
        if (nvmlDeviceGetHandleByIndex(gpu_info->device_id, &dev) == NVML_SUCCESS) {
            unsigned int temp;
            if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
                gpu_info->current_temp_c = temp;
            }
        }
        nvmlShutdown();
        return 0;
    }
    return -1;
}

int fb_update_nvidia_power(fb_gpu_info_t* gpu_info) {
    if (nvmlInit() == NVML_SUCCESS) {
        nvmlDevice_t dev;
        if (nvmlDeviceGetHandleByIndex(gpu_info->device_id, &dev) == NVML_SUCCESS) {
            unsigned int power_mw;
            if (nvmlDeviceGetPowerUsage(dev, &power_mw) == NVML_SUCCESS) {
                gpu_info->current_power_watts = power_mw / 1000;
            }
        }
        nvmlShutdown();
        return 0;
    }
    return -1;
}

int fb_update_nvidia_memory(fb_gpu_info_t* gpu_info) {
    cudaSetDevice(gpu_info->device_id);
    size_t free_mem, total_mem;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        gpu_info->free_memory_bytes = free_mem;
        return 0;
    }
    return -1;
}

#else

// Stub when CUDA not enabled
int fb_detect_cuda_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    (void)gpus;
    (void)max_gpus;
    *num_gpus = 0;
    return -1;
}

int fb_update_nvidia_utilization(fb_gpu_info_t* gpu_info) {
    (void)gpu_info;
    return -1;
}

int fb_update_nvidia_temperature(fb_gpu_info_t* gpu_info) {
    (void)gpu_info;
    return -1;
}

int fb_update_nvidia_power(fb_gpu_info_t* gpu_info) {
    (void)gpu_info;
    return -1;
}

int fb_update_nvidia_memory(fb_gpu_info_t* gpu_info) {
    (void)gpu_info;
    return -1;
}

#endif // FB_ENABLE_CUDA
