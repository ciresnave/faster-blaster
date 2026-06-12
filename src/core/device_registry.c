/**
 * @file device_registry.c
 * @brief Real device registry with CPU and GPU detection
 */

#include "faster-blaster/device_registry.h"
#include "faster-blaster/compute_device.h"
#include "device_detection/cpu_detect.h"
#include "device_detection/gpu_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declaration: defined in opencl_detect.c, conditionally compiled */
extern int fb_detect_opencl_gpus(fb_gpu_info_t* gpus, int max_gpus, int* count_out);

/* Maximum devices we can support */
#define MAX_DEVICES 32
#define MAX_CALLBACKS 16

/* Device storage */
static fb_compute_device_t g_devices[MAX_DEVICES];
static int g_device_count = 0;
static bool g_registry_initialized = false;

/* Callback storage */
typedef struct {
    fb_device_event_callback_t callback;
    void* user_data;
    bool active;
} callback_entry_t;

static callback_entry_t g_callbacks[MAX_CALLBACKS];
static int g_next_callback_id = 0;

/* ============================================================================
 * Helper Functions
 * ========================================================================== */

/**
 * @brief Convert CPU info to compute device
 */
static void cpu_info_to_device(const fb_cpu_info_t* cpu_info, 
                               fb_compute_device_t* device,
                               int device_id) {
    memset(device, 0, sizeof(fb_compute_device_t));
    
    /* Set basic properties */
    device->properties.type = FB_DEVICE_TYPE_CPU;
    device->properties.device_id = device_id;
    
    /* Set CPU name */
    if (strlen(cpu_info->brand_string) > 0) {
        snprintf(device->properties.name, sizeof(device->properties.name), 
                "%s", cpu_info->brand_string);
    } else {
        snprintf(device->properties.name, sizeof(device->properties.name), 
                "CPU #%d", device_id);
    }
    
    /* CPU-specific properties */
    switch (cpu_info->vendor) {
        case CPU_VENDOR_INTEL:
            device->properties.cpu.vendor = FB_CPU_VENDOR_INTEL;
            break;
        case CPU_VENDOR_AMD:
            device->properties.cpu.vendor = FB_CPU_VENDOR_AMD;
            break;
        case CPU_VENDOR_ARM:
        case CPU_VENDOR_APPLE:
        case CPU_VENDOR_AMPERE:
        case CPU_VENDOR_QUALCOMM:
        case CPU_VENDOR_NVIDIA:
            device->properties.cpu.vendor = FB_CPU_VENDOR_ARM;
            break;
        default:
            device->properties.cpu.vendor = FB_CPU_VENDOR_UNKNOWN;
            break;
    }
    
    device->properties.cpu.num_physical_cores = cpu_info->topology.physical_cores;
    device->properties.cpu.num_logical_cores = cpu_info->topology.logical_cores;
    
    /* Cache properties */
    device->properties.cpu.l1_cache_size = cpu_info->cache.l1_data.size_kb * 1024;
    device->properties.cpu.l2_cache_size = cpu_info->cache.l2.size_kb * 1024;
    device->properties.cpu.l3_cache_size = cpu_info->cache.l3.size_kb * 1024;
    device->properties.cpu.cache_line_size = cpu_info->cache.l1_data.line_size;
    
    /* SIMD capabilities */
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_info->architecture == CPU_ARCH_X86_64) {
        device->properties.cpu.has_avx = cpu_info->simd.x86.avx;
        device->properties.cpu.has_avx2 = cpu_info->simd.x86.avx2;
        device->properties.cpu.has_avx512 = cpu_info->simd.x86.avx512f;
        device->properties.cpu.has_fma = cpu_info->simd.x86.fma;
        
        /* Determine SIMD width */
        if (cpu_info->simd.x86.avx512f) {
            device->properties.cpu.simd_width = 512;
        } else if (cpu_info->simd.x86.avx || cpu_info->simd.x86.avx2) {
            device->properties.cpu.simd_width = 256;
        } else {
            device->properties.cpu.simd_width = 128; // SSE
        }
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (cpu_info->architecture == CPU_ARCH_ARM64) {
        device->properties.cpu.has_neon = cpu_info->simd.arm.neon;
        device->properties.cpu.simd_width = 128; // NEON is 128-bit
    }
#endif
    
    /* Capabilities */
    device->properties.capabilities = FB_DEVICE_CAP_FP64; // CPUs always support FP64
    
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_info->simd.x86.avx512_bf16) {
        device->properties.capabilities |= FB_DEVICE_CAP_BF16;
    }
    if (cpu_info->simd.x86.amx_bf16 || cpu_info->simd.x86.amx_int8) {
        device->properties.capabilities |= FB_DEVICE_CAP_AMX;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (cpu_info->simd.arm.sve) {
        device->properties.capabilities |= FB_DEVICE_CAP_SVE;
    }
    if (cpu_info->simd.arm.bf16) {
        device->properties.capabilities |= FB_DEVICE_CAP_BF16;
    }
#endif
    
    /* Estimate performance (very rough) */
    int cores = cpu_info->topology.physical_cores > 0 ? cpu_info->topology.physical_cores : 1;
    double freq_ghz = cpu_info->base_freq_mhz / 1000.0;
    
    /* Rough GFLOPS estimate: cores * freq * ops_per_cycle */
    int ops_per_cycle = 16; // AVX-512 = 16 FP32 ops/cycle, AVX2 = 8, etc.
    if (device->properties.cpu.has_avx512) {
        ops_per_cycle = 16;
    } else if (device->properties.cpu.has_avx2) {
        ops_per_cycle = 8;
    } else if (device->properties.cpu.has_avx) {
        ops_per_cycle = 8;
    } else {
        ops_per_cycle = 4; // SSE
    }
    
    device->properties.peak_gflops_fp32 = cores * freq_ghz * ops_per_cycle * 2; // 2 for FMA
    device->properties.peak_gflops_fp64 = device->properties.peak_gflops_fp32 / 2;
    
    /* Memory bandwidth (rough estimate based on typical DDR4/DDR5) */
    device->properties.memory_bandwidth_gbps = 50.0; // Conservative estimate
    
    /* Total memory - use system RAM */
    device->properties.total_memory = (size_t)8ULL * 1024 * 1024 * 1024; // Default 8GB
    
    /* Initialize state */
    device->is_available = true;
    device->is_initialized = true;
    device->load.utilization_percent = 0.0;
    device->load.memory_used = 0;
    device->load.memory_available = device->properties.total_memory;
    device->load.active_operations = 0;
    device->load.power_state = FB_POWER_STATE_PERFORMANCE;
}

/**
 * @brief Convert GPU info to compute device
 */
static void gpu_info_to_device(const fb_gpu_info_t* gpu_info, 
                               fb_compute_device_t* device,
                               int device_id) {
    memset(device, 0, sizeof(fb_compute_device_t));
    
    /* Set basic properties */
    device->properties.type = FB_DEVICE_TYPE_GPU;
    device->properties.device_id = device_id;
    
    /* Set GPU name */
    snprintf(device->properties.name, sizeof(device->properties.name), 
            "%s", gpu_info->name);
    
    /* GPU-specific properties */
    switch (gpu_info->vendor) {
        case GPU_VENDOR_NVIDIA:
            device->properties.gpu.vendor = FB_GPU_VENDOR_NVIDIA;
            break;
        case GPU_VENDOR_AMD:
            device->properties.gpu.vendor = FB_GPU_VENDOR_AMD;
            break;
        case GPU_VENDOR_INTEL:
            device->properties.gpu.vendor = FB_GPU_VENDOR_INTEL;
            break;
        case GPU_VENDOR_APPLE:
            device->properties.gpu.vendor = FB_GPU_VENDOR_APPLE;
            break;
        default:
            device->properties.gpu.vendor = FB_GPU_VENDOR_UNKNOWN;
            break;
    }
    
    device->properties.gpu.compute_capability_major = gpu_info->compute.cuda_major;
    device->properties.gpu.compute_capability_minor = gpu_info->compute.cuda_minor;
    device->properties.gpu.multiprocessor_count = gpu_info->compute_units;
    device->properties.gpu.max_threads_per_block = 1024; // Standard for modern GPUs
    device->properties.gpu.total_global_memory = gpu_info->total_memory_bytes;
    device->properties.gpu.warp_size = 32; // Standard for NVIDIA/AMD
    device->properties.gpu.pci_device_id = gpu_info->device_id;
    
    /* Store PCI bus ID for deduplication - parse bus number from string like "0000:01:00.0" */
    if (gpu_info->pci_bus_id[0] != '\0') {
        /* Extract bus number from PCI ID string (format: domain:bus:device.function) */
        int domain, bus, dev, func;
        if (sscanf(gpu_info->pci_bus_id, "%x:%x:%x.%x", &domain, &bus, &dev, &func) == 4) {
            device->properties.gpu.pci_bus_id = (domain << 16) | (bus << 8) | (dev << 3) | func;
        } else {
            device->properties.gpu.pci_bus_id = 0;
        }
    } else {
        device->properties.gpu.pci_bus_id = 0;
    }
    
    /* Performance metrics */
    device->properties.peak_gflops_fp32 = gpu_info->fp32_tflops * 1000.0; // Convert TFLOPS to GFLOPS
    device->properties.peak_gflops_fp64 = gpu_info->fp64_tflops * 1000.0;
    device->properties.memory_bandwidth_gbps = gpu_info->memory_bandwidth_gbps;
    device->properties.total_memory = gpu_info->total_memory_bytes;
    
    /* Capabilities */
    device->properties.capabilities = 0;
    
    if (gpu_info->compute.fp64) {
        device->properties.capabilities |= FB_DEVICE_CAP_FP64;
    }
    if (gpu_info->compute.fp16) {
        device->properties.capabilities |= FB_DEVICE_CAP_FP16;
    }
    if (gpu_info->compute.bf16) {
        device->properties.capabilities |= FB_DEVICE_CAP_BF16;
    }
    if (gpu_info->compute.int8) {
        device->properties.capabilities |= FB_DEVICE_CAP_INT8;
    }
    if (gpu_info->compute.has_tensor_cores) {
        device->properties.capabilities |= FB_DEVICE_CAP_TENSOR_CORES;
    }
    if (gpu_info->compute.has_ecc) {
        device->properties.capabilities |= FB_DEVICE_CAP_ECC_MEMORY;
    }
    
    /* Initialize state */
    device->is_available = true;
    device->is_initialized = true;
    device->load.utilization_percent = 0.0;
    device->load.memory_used = 0;
    device->load.memory_available = gpu_info->total_memory_bytes;
    device->load.active_operations = 0;
    device->load.power_state = FB_POWER_STATE_PERFORMANCE;
}

/**
 * @brief Trigger device event callbacks
 */
static void trigger_callbacks(fb_device_event_t event, int device_id) {
    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (g_callbacks[i].active && g_callbacks[i].callback) {
            g_callbacks[i].callback(event, device_id, g_callbacks[i].user_data);
        }
    }
}

/* ============================================================================
 * Registry Management
 * ========================================================================== */

int fb_registry_init(fb_discovery_flags_t flags) {
    if (g_registry_initialized) {
        fprintf(stderr, "Device registry already initialized\n");
        return g_device_count;
    }
    
    g_device_count = 0;
    memset(g_devices, 0, sizeof(g_devices));
    memset(g_callbacks, 0, sizeof(g_callbacks));
    
    printf("Detecting compute devices...\n");
    
    /* Detect CPUs */
    if (flags & FB_DISCOVERY_CPU) {
        fb_cpu_info_t cpu_info;
        memset(&cpu_info, 0, sizeof(cpu_info));
        
        int result = fb_detect_cpu(&cpu_info);
        if (result == 0) {
            cpu_info_to_device(&cpu_info, &g_devices[g_device_count], g_device_count);
            printf("  [%d] CPU: %s (%d cores, %.0f GFLOPS)\n",
                   g_device_count,
                   g_devices[g_device_count].properties.name,
                   g_devices[g_device_count].properties.cpu.num_physical_cores,
                   g_devices[g_device_count].properties.peak_gflops_fp32);
            g_device_count++;
        } else {
            fprintf(stderr, "  Warning: CPU detection failed (error %d), using defaults\n", result);
            /* Create a default CPU device anyway */
            memset(&cpu_info, 0, sizeof(cpu_info));
            cpu_info.vendor = CPU_VENDOR_UNKNOWN;
            cpu_info.topology.physical_cores = 8;
            cpu_info.topology.logical_cores = 16;
            cpu_info.base_freq_mhz = 3000;
            strcpy(cpu_info.brand_string, "Unknown CPU");
            cpu_info_to_device(&cpu_info, &g_devices[g_device_count], g_device_count);
            g_device_count++;
        }
    }
    
    /* Detect GPUs */
    if (flags & FB_DISCOVERY_GPU) {
        fb_gpu_info_t gpus[MAX_DEVICES];
        uint32_t num_gpus = 0;
        int result = -1;
        
        /* Try CUDA GPUs */
#ifdef FB_ENABLE_CUDA
        memset(gpus, 0, sizeof(gpus));
        num_gpus = 0;
        result = fb_detect_cuda_gpus(gpus, MAX_DEVICES - g_device_count, &num_gpus);
        if (result == 0 && num_gpus > 0) {
            printf("  Found %u CUDA GPU(s)\n", num_gpus);
            for (uint32_t i = 0; i < num_gpus && g_device_count < MAX_DEVICES; i++) {
                gpu_info_to_device(&gpus[i], &g_devices[g_device_count], g_device_count);
                printf("    [%d] GPU: %s (%.0f GFLOPS, %zu MB)\n",
                       g_device_count,
                       g_devices[g_device_count].properties.name,
                       g_devices[g_device_count].properties.peak_gflops_fp32,
                       g_devices[g_device_count].properties.total_memory / (1024 * 1024));
                g_device_count++;
            }
        }
#endif
        
        /* Try ROCm GPUs */
        memset(gpus, 0, sizeof(gpus));
        num_gpus = 0;
        result = fb_detect_rocm_gpus(gpus, MAX_DEVICES - g_device_count, &num_gpus);
        if (result == 0 && num_gpus > 0) {
            printf("  Found %u ROCm GPU(s)\n", num_gpus);
            for (uint32_t i = 0; i < num_gpus && g_device_count < MAX_DEVICES; i++) {
                gpu_info_to_device(&gpus[i], &g_devices[g_device_count], g_device_count);
                printf("    [%d] GPU: %s (%.0f GFLOPS, %zu MB)\n",
                       g_device_count,
                       g_devices[g_device_count].properties.name,
                       g_devices[g_device_count].properties.peak_gflops_fp32,
                       g_devices[g_device_count].properties.total_memory / (1024 * 1024));
                g_device_count++;
            }
        }
        
        /* Try Intel GPUs */
        memset(gpus, 0, sizeof(gpus));
        num_gpus = 0;
        result = fb_detect_intel_gpus(gpus, MAX_DEVICES - g_device_count, &num_gpus);
        if (result == 0 && num_gpus > 0) {
            printf("  Found %u Intel GPU(s)\n", num_gpus);
            for (uint32_t i = 0; i < num_gpus && g_device_count < MAX_DEVICES; i++) {
                gpu_info_to_device(&gpus[i], &g_devices[g_device_count], g_device_count);
                printf("    [%d] GPU: %s (%.0f GFLOPS, %zu MB)\n",
                       g_device_count,
                       g_devices[g_device_count].properties.name,
                       g_devices[g_device_count].properties.peak_gflops_fp32,
                       g_devices[g_device_count].properties.total_memory / (1024 * 1024));
                g_device_count++;
            }
        }
        
        /* Try OpenCL GPUs */
#ifdef FB_ENABLE_OPENCL
        memset(gpus, 0, sizeof(gpus));
        num_gpus = 0;
        {
            int opencl_count_int = 0;
            result = fb_detect_opencl_gpus(gpus, MAX_DEVICES - g_device_count, &opencl_count_int);
            num_gpus = (uint32_t)opencl_count_int;
        }
        if (result == 0 && num_gpus > 0) {
            printf("  Found %u OpenCL GPU(s)\n", num_gpus);
            for (uint32_t i = 0; i < num_gpus && g_device_count < MAX_DEVICES; i++) {
                /* Check if this GPU is already in the registry (deduplication) */
                bool duplicate = false;
                for (int j = 0; j < g_device_count; j++) {
                    if (g_devices[j].properties.type == FB_DEVICE_TYPE_GPU) {
                        /* Primary: Compare by PCI bus ID (unique per physical PCIe slot) */
                        if (gpus[i].pci_bus_id[0] != '\0') {
                            /* Build PCI ID for comparison */
                            int domain, bus, dev, func;
                            if (sscanf(gpus[i].pci_bus_id, "%x:%x:%x.%x", &domain, &bus, &dev, &func) == 4) {
                                int opencl_pci = (domain << 16) | (bus << 8) | (dev << 3) | func;
                                if (g_devices[j].properties.gpu.pci_bus_id != 0 && 
                                    g_devices[j].properties.gpu.pci_bus_id == opencl_pci) {
                                    duplicate = true;
                                    printf("    Skipping duplicate GPU: %s (PCI %s already registered as device %d)\n",
                                           gpus[i].name, gpus[i].pci_bus_id, j);
                                    break;
                                }
                            }
                        }
                        
                        /* Fallback: Compare by name and memory (less reliable, won't work with multiple identical GPUs) */
                        if (!duplicate) {
                            bool same_name = (strcmp(g_devices[j].properties.name, gpus[i].name) == 0);
                            bool same_memory = (g_devices[j].properties.total_memory == gpus[i].total_memory_bytes);
                            
                            if (same_name && same_memory) {
                                duplicate = true;
                                printf("    Skipping duplicate GPU: %s (name+memory match, device %d - WARNING: may skip identical GPUs)\n",
                                       gpus[i].name, j);
                                break;
                            }
                        }
                    }
                }
                
                if (!duplicate) {
                    gpu_info_to_device(&gpus[i], &g_devices[g_device_count], g_device_count);
                    printf("    [%d] GPU: %s (%.0f GFLOPS, %zu MB)\n",
                           g_device_count,
                           g_devices[g_device_count].properties.name,
                           g_devices[g_device_count].properties.peak_gflops_fp32,
                           g_devices[g_device_count].properties.total_memory / (1024 * 1024));
                    g_device_count++;
                }
            }
        }
#endif
    }
    
    g_registry_initialized = true;
    printf("Device registry initialized: %d device(s) detected\n", g_device_count);
    
    return g_device_count;
}

void fb_registry_shutdown(void) {
    if (!g_registry_initialized) {
        return;
    }
    
    printf("Shutting down device registry...\n");
    
    g_device_count = 0;
    g_registry_initialized = false;
    memset(g_devices, 0, sizeof(g_devices));
    memset(g_callbacks, 0, sizeof(g_callbacks));
}

int fb_registry_rescan(fb_discovery_flags_t flags) {
    /* For now, just re-initialize */
    int old_count = g_device_count;
    fb_registry_shutdown();
    int new_count = fb_registry_init(flags);
    
    /* Trigger callbacks for new devices */
    if (new_count > old_count) {
        for (int i = old_count; i < new_count; i++) {
            trigger_callbacks(FB_DEVICE_EVENT_ADDED, i);
        }
        return new_count - old_count;
    }
    
    return 0;
}

int fb_registry_register_callback(fb_device_event_callback_t callback, void* user_data) {
    if (!callback) {
        return -1;
    }
    
    for (int i = 0; i < MAX_CALLBACKS; i++) {
        if (!g_callbacks[i].active) {
            g_callbacks[i].callback = callback;
            g_callbacks[i].user_data = user_data;
            g_callbacks[i].active = true;
            return g_next_callback_id++;
        }
    }
    
    return -1; // No slots available
}

void fb_registry_unregister_callback(int callback_id) {
    if (callback_id < 0 || callback_id >= MAX_CALLBACKS) {
        return;
    }
    
    g_callbacks[callback_id].active = false;
    g_callbacks[callback_id].callback = NULL;
    g_callbacks[callback_id].user_data = NULL;
}

/* ============================================================================
 * Device Querying
 * ========================================================================== */

int fb_registry_get_device_count(void) {
    return g_device_count;
}

fb_compute_device_t* fb_registry_get_device(int device_id) {
    if (device_id < 0 || device_id >= g_device_count) {
        return NULL;
    }
    return &g_devices[device_id];
}

fb_compute_device_t* fb_get_device(uint32_t index) {
    return fb_registry_get_device((int)index);
}

uint32_t fb_get_device_count(void) {
    return (uint32_t)g_device_count;
}

void fb_print_devices(void) {
    printf("\n=== Detected Compute Devices ===\n");
    for (int i = 0; i < g_device_count; i++) {
        fb_compute_device_t* dev = &g_devices[i];
        printf("[%d] %s: %s\n", i,
               dev->properties.type == FB_DEVICE_TYPE_CPU ? "CPU" : "GPU",
               dev->properties.name);
        printf("    Performance: %.0f GFLOPS (FP32), %.0f GFLOPS (FP64)\n",
               dev->properties.peak_gflops_fp32,
               dev->properties.peak_gflops_fp64);
        printf("    Memory: %.2f GB, Bandwidth: %.1f GB/s\n",
               dev->properties.total_memory / (1024.0 * 1024.0 * 1024.0),
               dev->properties.memory_bandwidth_gbps);
        
        /* Print capabilities */
        printf("    Capabilities:");
        if (dev->properties.capabilities & FB_DEVICE_CAP_FP64) printf(" FP64");
        if (dev->properties.capabilities & FB_DEVICE_CAP_FP16) printf(" FP16");
        if (dev->properties.capabilities & FB_DEVICE_CAP_BF16) printf(" BF16");
        if (dev->properties.capabilities & FB_DEVICE_CAP_TF32) printf(" TF32");
        if (dev->properties.capabilities & FB_DEVICE_CAP_TENSOR_CORES) printf(" TensorCores");
        if (dev->properties.capabilities & FB_DEVICE_CAP_AMX) printf(" AMX");
        if (dev->properties.capabilities & FB_DEVICE_CAP_SVE) printf(" SVE");
        printf("\n");
        
        if (dev->properties.type == FB_DEVICE_TYPE_CPU) {
            printf("    Cores: %d physical, %d logical\n",
                   dev->properties.cpu.num_physical_cores,
                   dev->properties.cpu.num_logical_cores);
            printf("    SIMD: %d-bit", dev->properties.cpu.simd_width);
            if (dev->properties.cpu.has_avx512) printf(" (AVX-512)");
            else if (dev->properties.cpu.has_avx2) printf(" (AVX2)");
            else if (dev->properties.cpu.has_avx) printf(" (AVX)");
            else if (dev->properties.cpu.has_neon) printf(" (NEON)");
            printf("\n");
        } else {
            printf("    Compute Capability: %d.%d\n",
                   dev->properties.gpu.compute_capability_major,
                   dev->properties.gpu.compute_capability_minor);
            printf("    Multiprocessors: %d\n",
                   dev->properties.gpu.multiprocessor_count);
        }
    }
    printf("================================\n\n");
}

int fb_registry_get_devices_by_type(fb_device_type_t type,
                                    fb_compute_device_t** devices,
                                    int max_devices) {
    int count = 0;
    for (int i = 0; i < g_device_count && count < max_devices; i++) {
        if (g_devices[i].properties.type == type) {
            devices[count++] = &g_devices[i];
        }
    }
    return count;
}

int fb_registry_find_devices(size_t min_memory,
                              double min_gflops,
                              uint32_t required_caps,
                              fb_compute_device_t** devices,
                              int max_devices) {
    int count = 0;
    for (int i = 0; i < g_device_count && count < max_devices; i++) {
        fb_compute_device_t* dev = &g_devices[i];
        
        /* Check memory */
        if (min_memory > 0 && dev->properties.total_memory < min_memory) {
            continue;
        }
        
        /* Check GFLOPS */
        if (min_gflops > 0 && dev->properties.peak_gflops_fp32 < min_gflops) {
            continue;
        }
        
        /* Check capabilities */
        if (required_caps != 0 && (dev->properties.capabilities & required_caps) != required_caps) {
            continue;
        }
        
        devices[count++] = dev;
    }
    return count;
}

/* ============================================================================
 * Public Device API Wrappers
 * ========================================================================== */

int fb_device_init(void) {
    return fb_registry_init(FB_DISCOVERY_CPU | FB_DISCOVERY_GPU);
}

void fb_device_shutdown(void) {
    fb_registry_shutdown();
}

int fb_device_get_count(void) {
    return fb_registry_get_device_count();
}

int fb_device_get_cpu_count(void) {
    int count = 0;
    for (int i = 0; i < g_device_count; i++) {
        if (g_devices[i].properties.type == FB_DEVICE_TYPE_CPU) {
            count++;
        }
    }
    return count;
}

int fb_device_get_gpu_count(void) {
    int count = 0;
    for (int i = 0; i < g_device_count; i++) {
        if (g_devices[i].properties.type == FB_DEVICE_TYPE_GPU) {
            count++;
        }
    }
    return count;
}

fb_compute_device_t* fb_device_get(int device_id) {
    return fb_registry_get_device(device_id);
}

fb_compute_device_t* fb_device_get_by_type(fb_device_type_t type, int index) {
    int count = 0;
    for (int i = 0; i < g_device_count; i++) {
        if (g_devices[i].properties.type == type) {
            if (count == index) {
                return &g_devices[i];
            }
            count++;
        }
    }
    return NULL;
}

const char* fb_device_get_name(const fb_compute_device_t* device) {
    return device ? device->properties.name : "Unknown";
}

const char* fb_device_get_type_string(const fb_compute_device_t* device) {
    if (!device) {
        return "Unknown";
    }
    switch (device->properties.type) {
        case FB_DEVICE_TYPE_CPU: return "CPU";
        case FB_DEVICE_TYPE_GPU: return "GPU";
        default: return "Unknown";
    }
}
