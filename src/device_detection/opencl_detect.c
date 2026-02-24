/**
 * @file opencl_detect.c
 * @brief OpenCL GPU Detection Implementation
 * 
 * Detects OpenCL-capable devices including GPUs from AMD, NVIDIA, Intel, and others.
 * OpenCL provides broader hardware support than vendor-specific APIs (CUDA/HIP).
 */

#include "device_detection/gpu_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef FB_ENABLE_OPENCL
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

/**
 * @brief Detect all OpenCL-capable GPUs
 */
int fb_detect_opencl_gpus(fb_gpu_info_t* gpus, int max_gpus, int* count_out) {
    cl_int err;
    cl_uint num_platforms = 0;
    
    *count_out = 0;
    
    printf("[OpenCL] Detection starting...\n");
    
    // Get number of platforms
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    printf("[OpenCL] clGetPlatformIDs returned %d, num_platforms=%u\n", err, num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        printf("[OpenCL] No platforms found\n");
        return 0; // No OpenCL platforms
    }
    
    // Get platforms
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    if (!platforms) return -1;
    
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("[OpenCL] Failed to get platform IDs\n");
        free(platforms);
        return -1;
    }
    
    int total_devices = 0;
    
    // Iterate through platforms
    for (cl_uint p = 0; p < num_platforms && total_devices < max_gpus; p++) {
        cl_uint num_devices = 0;
        
        char platform_name[256] = {0};
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        printf("[OpenCL] Platform %u: %s\n", p, platform_name);
        
        // Get GPU devices for this platform
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        printf("[OpenCL]   clGetDeviceIDs returned %d, num_devices=%u\n", err, num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            continue; // No GPUs on this platform
        }
        
        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
        if (!devices) continue;
        
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
        if (err != CL_SUCCESS) {
            free(devices);
            continue;
        }
        
        // Query each device
        for (cl_uint d = 0; d < num_devices && total_devices < max_gpus; d++) {
            fb_gpu_info_t* gpu = &gpus[total_devices];
            memset(gpu, 0, sizeof(fb_gpu_info_t));
            
            gpu->device_id = total_devices;
            gpu->opencl_available = true;
            
            // Device name (try board name first, fall back to device name)
            char board_name[256] = {0};
            cl_int board_err = clGetDeviceInfo(devices[d], 0x4038, sizeof(board_name), board_name, NULL); // CL_DEVICE_BOARD_NAME_AMD
            if (board_err == CL_SUCCESS && strlen(board_name) > 0) {
                strncpy(gpu->name, board_name, sizeof(gpu->name) - 1);
            } else {
                clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(gpu->name), gpu->name, NULL);
            }
            
            // Vendor
            char vendor_name[256] = {0};
            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
            
            if (strstr(vendor_name, "NVIDIA")) {
                gpu->vendor = GPU_VENDOR_NVIDIA;
            } else if (strstr(vendor_name, "AMD") || strstr(vendor_name, "Advanced Micro Devices")) {
                gpu->vendor = GPU_VENDOR_AMD;
            } else if (strstr(vendor_name, "Intel")) {
                gpu->vendor = GPU_VENDOR_INTEL;
            } else if (strstr(vendor_name, "Apple")) {
                gpu->vendor = GPU_VENDOR_APPLE;
            } else {
                gpu->vendor = GPU_VENDOR_UNKNOWN;
            }
            
            // Try to get PCI bus ID using vendor-specific extensions
            // This is critical for deduplication across CUDA/ROCm/OpenCL
            gpu->pci_bus_id[0] = '\0';
            
            if (gpu->vendor == GPU_VENDOR_NVIDIA) {
                // NVIDIA: cl_nv_device_attribute_query extension
                // CL_DEVICE_PCI_BUS_ID_NV = 0x4008, CL_DEVICE_PCI_SLOT_ID_NV = 0x4009
                cl_uint pci_bus_id = 0, pci_slot_id = 0;
                if (clGetDeviceInfo(devices[d], 0x4008, sizeof(pci_bus_id), &pci_bus_id, NULL) == CL_SUCCESS &&
                    clGetDeviceInfo(devices[d], 0x4009, sizeof(pci_slot_id), &pci_slot_id, NULL) == CL_SUCCESS) {
                    // Format: domain:bus:device.function
                    // pci_bus_id = bus, pci_slot_id = (device << 3) | function
                    int device_num = (pci_slot_id >> 3) & 0x1F;
                    int function = pci_slot_id & 0x7;
                    snprintf(gpu->pci_bus_id, sizeof(gpu->pci_bus_id),
                             "0000:%02x:%02x.%x", pci_bus_id, device_num, function);
                }
            } else if (gpu->vendor == GPU_VENDOR_AMD) {
                // AMD: cl_amd_device_attribute_query extension
                // CL_DEVICE_TOPOLOGY_AMD = 0x4037
                typedef union {
                    struct { cl_uint type; cl_uint data[5]; } raw;
                    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
                } cl_device_topology_amd;
                
                cl_device_topology_amd topology;
                if (clGetDeviceInfo(devices[d], 0x4037, sizeof(topology), &topology, NULL) == CL_SUCCESS) {
                    if (topology.raw.type == 1) { // CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD
                        snprintf(gpu->pci_bus_id, sizeof(gpu->pci_bus_id),
                                 "0000:%02x:%02x.%x", 
                                 (unsigned char)topology.pcie.bus,
                                 (unsigned char)topology.pcie.device,
                                 (unsigned char)topology.pcie.function);
                    }
                }
            }
            
            // Try standard OpenCL extensions for device identification (works for Intel and others)
            if (gpu->pci_bus_id[0] == '\0') {
                // Try cl_khr_device_uuid extension (OpenCL 3.0+)
                // CL_DEVICE_UUID_KHR = 0x106A, CL_DEVICE_LUID_KHR = 0x106D (Windows)
                cl_uchar device_uuid[16] = {0};
                if (clGetDeviceInfo(devices[d], 0x106A, sizeof(device_uuid), device_uuid, NULL) == CL_SUCCESS) {
                    // UUID is a 128-bit unique identifier - convert to hex string for PCI bus ID field
                    // Format as "UUID:xxxx-xxxx-xxxx-xxxx" to distinguish from real PCI IDs
                    snprintf(gpu->pci_bus_id, sizeof(gpu->pci_bus_id),
                             "UUID:%02x%02x-%02x%02x-%02x%02x-%02x%02x",
                             device_uuid[0], device_uuid[1], device_uuid[2], device_uuid[3],
                             device_uuid[4], device_uuid[5], device_uuid[6], device_uuid[7]);
                }
                
                // Try Windows LUID (Local Unique Identifier) if UUID not available
                #ifdef _WIN32
                if (gpu->pci_bus_id[0] == '\0') {
                    cl_uchar device_luid[8] = {0};
                    if (clGetDeviceInfo(devices[d], 0x106D, sizeof(device_luid), device_luid, NULL) == CL_SUCCESS) {
                        // LUID is a 64-bit Windows identifier
                        snprintf(gpu->pci_bus_id, sizeof(gpu->pci_bus_id),
                                 "LUID:%02x%02x%02x%02x%02x%02x%02x%02x",
                                 device_luid[0], device_luid[1], device_luid[2], device_luid[3],
                                 device_luid[4], device_luid[5], device_luid[6], device_luid[7]);
                    }
                }
                #endif
            }
            
            // If still no unique ID, fall back to name+memory deduplication (logged as warning)
            
            // OpenCL version
            char version[256] = {0};
            clGetDeviceInfo(devices[d], CL_DEVICE_VERSION, sizeof(version), version, NULL);
            // Parse "OpenCL M.m ..." format
            if (sscanf(version, "OpenCL %d.%d", &gpu->compute.opencl_major, &gpu->compute.opencl_minor) != 2) {
                gpu->compute.opencl_major = 1;
                gpu->compute.opencl_minor = 0;
            }
            
            // Compute units (AMD: CUs, NVIDIA: SMs, Intel: EUs)
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(gpu->compute_units), &gpu->compute_units, NULL);
            
            // Clock frequency (MHz)
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(gpu->base_clock_mhz), &gpu->base_clock_mhz, NULL);
            gpu->boost_clock_mhz = gpu->base_clock_mhz; // OpenCL doesn't expose boost clocks
            
            // Memory
            cl_ulong mem_size = 0;
            clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
            gpu->total_memory_bytes = (uint64_t)mem_size;
            
            // Memory bandwidth (estimate from bus width and clock if available)
            cl_uint mem_bus_width = 0;
            cl_uint mem_clock_mhz = 0;
            
            // Try vendor-specific extensions for detailed memory info
            // Note: These may not be available on all implementations
            clGetDeviceInfo(devices[d], 0x4008, sizeof(mem_bus_width), &mem_bus_width, NULL); // CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD
            
            if (mem_bus_width > 0) {
                gpu->memory_bus_width = mem_bus_width;
                // Rough estimate: (bus_width / 8) * clock * 2 (DDR)
                gpu->memory_bandwidth_gbps = (mem_bus_width / 8.0) * gpu->base_clock_mhz * 2.0 / 1000.0;
            }
            
            // Capabilities
            cl_device_fp_config fp_config = 0;
            clGetDeviceInfo(devices[d], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, NULL);
            gpu->compute.fp64 = (fp_config != 0);
            
            clGetDeviceInfo(devices[d], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fp_config), &fp_config, NULL);
            gpu->compute.fp32 = (fp_config != 0);
            
            // Half precision
            #ifdef CL_DEVICE_HALF_FP_CONFIG
            clGetDeviceInfo(devices[d], CL_DEVICE_HALF_FP_CONFIG, sizeof(fp_config), &fp_config, NULL);
            gpu->compute.fp16 = (fp_config != 0);
            #else
            gpu->compute.fp16 = false; // OpenCL < 1.2
            #endif
            
            // Unified memory
            cl_bool unified_mem = CL_FALSE;
            clGetDeviceInfo(devices[d], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified_mem), &unified_mem, NULL);
            gpu->compute.has_unified_memory = (unified_mem == CL_TRUE);
            
            // Performance estimation (very rough)
            // GFLOPS ≈ compute_units * cores_per_cu * clock * 2 (FMA)
            int cores_per_cu = 64; // Conservative estimate
            if (gpu->vendor == GPU_VENDOR_NVIDIA) cores_per_cu = 64;
            else if (gpu->vendor == GPU_VENDOR_AMD) cores_per_cu = 64;
            else if (gpu->vendor == GPU_VENDOR_INTEL) cores_per_cu = 8; // EU execution units
            
            gpu->cores_per_cu = cores_per_cu;
            gpu->total_cores = gpu->compute_units * cores_per_cu;
            
            double gflops = gpu->compute_units * cores_per_cu * (gpu->base_clock_mhz / 1000.0) * 2.0;
            gpu->fp32_tflops = gflops / 1000.0;
            gpu->fp64_tflops = gpu->compute.fp64 ? (gflops / 1000.0 / 2.0) : 0.0; // Typically 1/2 rate
            
            total_devices++;
        }
        
        free(devices);
    }
    
    free(platforms);
    *count_out = total_devices;
    return 0;
}

/**
 * @brief Update memory usage for OpenCL GPU
 */
int fb_update_opencl_memory(fb_gpu_info_t* gpu) {
    // OpenCL doesn't provide standard memory usage queries
    // Would require platform-specific extensions
    (void)gpu;
    return -1; // Not supported
}

#else /* !FB_ENABLE_OPENCL */

int fb_detect_opencl_gpus(fb_gpu_info_t* gpus, int max_gpus, int* count_out) {
    (void)gpus;
    (void)max_gpus;
    *count_out = 0;
    return 0; // OpenCL not enabled
}

int fb_update_opencl_memory(fb_gpu_info_t* gpu) {
    (void)gpu;
    return -1;
}

#endif /* FB_ENABLE_OPENCL */
