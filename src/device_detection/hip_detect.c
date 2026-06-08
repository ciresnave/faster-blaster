/**
 * @file hip_detect.c
 * @brief AMD ROCm/HIP GPU Detection
 * 
 * Separated from gpu_detect.c to avoid header conflicts with CUDA.
 * This file can only be compiled when FB_ENABLE_HIP is defined.
 */

#include "device_detection/gpu_detect.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef FB_ENABLE_HIP

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int fb_detect_rocm_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    *num_gpus = 0;
    
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    
    if (err != hipSuccess || device_count == 0) {
        return -1;
    }
    
    for (int i = 0; i < device_count && i < (int)max_gpus; i++) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, i) != hipSuccess) {
            continue;
        }

        const char *gcn_arch_name = prop.gcnArchName;
        
        fb_gpu_info_t* gpu = &gpus[*num_gpus];
        memset(gpu, 0, sizeof(*gpu));
        
        gpu->vendor = GPU_VENDOR_AMD;
        gpu->device_id = i;
        snprintf(gpu->name, sizeof(gpu->name), "%s", prop.name);
        
        // Parse GCN architecture number from gcnArchName (e.g., "gfx906", "gfx1030")
        int gcn_arch = 0;
        if (gcn_arch_name[0] != '\0' && strlen(gcn_arch_name) > 3) {
            // Skip "gfx" prefix and parse the number
            gcn_arch = atoi(gcn_arch_name + 3);
        }
        
        // HIP architecture string
        snprintf(gpu->compute.hip_arch, sizeof(gpu->compute.hip_arch),
                 "%s", gcn_arch_name[0] != '\0' ? gcn_arch_name : "unknown");
        
        // Architecture from GCN version
        if (gcn_arch >= 1000 && gcn_arch < 1100) {
            gpu->architecture = GPU_ARCH_AMD_RDNA;
        } else if (gcn_arch >= 1030) {
            gpu->architecture = GPU_ARCH_AMD_RDNA2;
        } else if (gcn_arch >= 9000) {
            gpu->architecture = GPU_ARCH_AMD_CDNA;
        } else {
            gpu->architecture = GPU_ARCH_AMD_GCN;
        }
        
        // Compute capabilities
        gpu->compute.fp64 = true;
        gpu->compute.fp32 = true;
        gpu->compute.fp16 = true;
        gpu->compute.bf16 = (gcn_arch >= 908); // CDNA+
        gpu->compute.int8 = true;
        gpu->compute.has_matrix_cores = (gcn_arch >= 908); // CDNA+
        
        // Compute units
        gpu->compute_units = prop.multiProcessorCount;
        gpu->cores_per_cu = 64; // Standard for AMD
        gpu->total_cores = gpu->compute_units * gpu->cores_per_cu;
        
        // Clocks
        gpu->base_clock_mhz = prop.clockRate / 1000;
        gpu->boost_clock_mhz = prop.clockRate / 1000; // Base approximation (HIP doesn't expose boost separately)
        gpu->memory_clock_mhz = prop.memoryClockRate / 1000;
        
        // Memory
        gpu->total_memory_bytes = prop.totalGlobalMem;
        gpu->l2_cache_bytes = prop.l2CacheSize;
        gpu->memory_bus_width = prop.memoryBusWidth;
        
        // Memory type
        if (strstr(prop.name, "MI") || gcn_arch >= 908) {
            gpu->memory_type = GPU_MEMORY_HBM2;
        } else {
            gpu->memory_type = GPU_MEMORY_GDDR6;
        }
        
        // Bandwidth
        gpu->memory_bandwidth_gbps = 
            (gpu->memory_clock_mhz * 2.0 * gpu->memory_bus_width / 8.0) / 1000.0;
        
        // PCI Bus ID
        snprintf(gpu->pci_bus_id, sizeof(gpu->pci_bus_id),
                 "%04x:%02x:%02x.%x",
                 prop.pciDomainID, prop.pciBusID,
                 prop.pciDeviceID, 0);
        
        // Runtime
        gpu->hip_available = true;
        
        // Memory info
        size_t free_mem, total_mem;
        hipSetDevice(i);
        if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
            gpu->free_memory_bytes = free_mem;
        }
        
        fb_estimate_gpu_tflops(gpu);
        
        (*num_gpus)++;
    }
    
    return 0;
}

int fb_update_amd_memory(fb_gpu_info_t* gpu_info) {
    hipSetDevice(gpu_info->device_id);
    size_t free_mem, total_mem;
    if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
        gpu_info->free_memory_bytes = free_mem;
        return 0;
    }
    return -1;
}
#ifdef __cplusplus
}
#endif
#else

// Stub when HIP not enabled
int fb_detect_rocm_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus) {
    (void)gpus;
    (void)max_gpus;
    *num_gpus = 0;
    return -1;
}

int fb_update_amd_memory(fb_gpu_info_t* gpu_info) {
    (void)gpu_info;
    return -1;
}

#endif // FB_ENABLE_HIP
