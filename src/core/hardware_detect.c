/**
 * @file hardware_detect.c
 * @brief Hardware detection implementation using cpuinfo and OpenCL
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "hardware_detect.h"
#include <cpuinfo.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

/* SHA-256 for fingerprinting (simple implementation) */
#include <stdint.h>

/* Global state */
static bool g_hw_initialized = false;
static fb_hardware_info_t g_hw_info;
static uint32_t g_cpu_features = 0;

/* Helper function to compute simple hash for fingerprinting */
static uint32_t hash_string(const char *str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash;
}

/* Generate hardware fingerprint from CPU and GPU info */
static void generate_fingerprint(fb_hardware_info_t *info) {
    uint32_t hash = 0;
    
    /* Hash CPU information */
    hash ^= hash_string(info->cpu_vendor);
    hash ^= hash_string(info->cpu_model);
    hash ^= (uint32_t)info->cpu_cores * 7919;
    hash ^= (uint32_t)(info->cpu_l3_cache >> 20) * 6151; /* L3 in MB */
    hash ^= g_cpu_features;
    
    /* Hash GPU information if present */
    if (info->has_gpu) {
        hash ^= hash_string(info->gpu_vendor);
        hash ^= hash_string(info->gpu_model);
        hash ^= (uint32_t)info->gpu_compute_units * 4561;
    }
    
    /* Convert hash to hex string */
    snprintf(info->fingerprint, sizeof(info->fingerprint), 
             "%08x%08x", hash, (hash ^ 0xDEADBEEF));
}

/* Detect CPU features and populate feature flags */
static void detect_cpu_features(void) {
    g_cpu_features = 0;
    
    #if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
        if (cpuinfo_has_x86_sse2()) g_cpu_features |= FB_CPU_FEATURE_SSE2;
        if (cpuinfo_has_x86_sse3()) g_cpu_features |= FB_CPU_FEATURE_SSE3;
        if (cpuinfo_has_x86_ssse3()) g_cpu_features |= FB_CPU_FEATURE_SSSE3;
        if (cpuinfo_has_x86_sse4_1()) g_cpu_features |= FB_CPU_FEATURE_SSE41;
        if (cpuinfo_has_x86_sse4_2()) g_cpu_features |= FB_CPU_FEATURE_SSE42;
        if (cpuinfo_has_x86_avx()) g_cpu_features |= FB_CPU_FEATURE_AVX;
        if (cpuinfo_has_x86_avx2()) g_cpu_features |= FB_CPU_FEATURE_AVX2;
        if (cpuinfo_has_x86_avx512f()) g_cpu_features |= FB_CPU_FEATURE_AVX512F;
        if (cpuinfo_has_x86_fma3()) g_cpu_features |= FB_CPU_FEATURE_FMA;
    #elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
        if (cpuinfo_has_arm_neon()) g_cpu_features |= FB_CPU_FEATURE_NEON;
        #if CPUINFO_ARCH_ARM64
            /* SVE detection if available in newer cpuinfo versions */
        #endif
    #endif
}

/* Detect GPU using OpenCL */
static int detect_gpu_opencl(fb_hardware_info_t *info) {
    cl_uint num_platforms = 0;
    cl_platform_id platform = NULL;
    cl_uint num_devices = 0;
    cl_device_id device = NULL;
    cl_int err;
    
    /* Get first platform */
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        return -1; /* No OpenCL platforms found */
    }
    
    /* Try to get a GPU device first */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        /* Try accelerator type (for Intel integrated graphics, etc.) */
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            return -1; /* No GPU devices found */
        }
    }
    
    /* Query GPU information */
    char device_name[256] = {0};
    char vendor_name[128] = {0};
    cl_uint compute_units = 0;
    cl_ulong global_mem_size = 0;
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    
    /* Populate GPU info */
    info->has_gpu = true;
    strncpy(info->gpu_model, device_name, sizeof(info->gpu_model) - 1);
    strncpy(info->gpu_vendor, vendor_name, sizeof(info->gpu_vendor) - 1);
    info->gpu_compute_units = (int)compute_units;
    info->gpu_memory_bytes = global_mem_size;
    
    return 0;
}

int fb_hw_init(void) {
    if (g_hw_initialized) {
        return 0;
    }
    
    /* Initialize cpuinfo */
    if (!cpuinfo_initialize()) {
        return -1;
    }
    
    g_hw_initialized = true;
    return 0;
}

int fb_hw_detect(fb_hardware_info_t *info) {
    if (!g_hw_initialized) {
        if (fb_hw_init() != 0) {
            return -1;
        }
    }
    
    memset(info, 0, sizeof(fb_hardware_info_t));
    
    /* Get CPU information from cpuinfo */
    const struct cpuinfo_processor *processor = cpuinfo_get_processor(0);
    const struct cpuinfo_package *package = cpuinfo_get_package(0);
    
    if (processor && package) {
        /* CPU vendor */
        #if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
            switch (cpuinfo_get_vendor()) {
                case cpuinfo_vendor_intel:
                    strncpy(info->cpu_vendor, "Intel", sizeof(info->cpu_vendor) - 1);
                    break;
                case cpuinfo_vendor_amd:
                    strncpy(info->cpu_vendor, "AMD", sizeof(info->cpu_vendor) - 1);
                    break;
                default:
                    strncpy(info->cpu_vendor, "Unknown", sizeof(info->cpu_vendor) - 1);
                    break;
            }
        #elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
            strncpy(info->cpu_vendor, "ARM", sizeof(info->cpu_vendor) - 1);
        #else
            strncpy(info->cpu_vendor, "Unknown", sizeof(info->cpu_vendor) - 1);
        #endif
        
        /* CPU model name */
        if (package->name) {
            strncpy(info->cpu_model, package->name, sizeof(info->cpu_model) - 1);
        } else {
            strncpy(info->cpu_model, "Unknown CPU", sizeof(info->cpu_model) - 1);
        }
        
        /* Core and thread count */
        info->cpu_cores = (int)cpuinfo_get_cores_count();
        info->cpu_threads = (int)cpuinfo_get_processors_count();
        
        /* Cache sizes */
        const struct cpuinfo_cache *l1 = cpuinfo_get_l1d_cache(0);
        const struct cpuinfo_cache *l2 = cpuinfo_get_l2_cache(0);
        const struct cpuinfo_cache *l3 = cpuinfo_get_l3_cache(0);
        
        if (l1) info->cpu_l1_cache = l1->size;
        if (l2) info->cpu_l2_cache = l2->size;
        if (l3) info->cpu_l3_cache = l3->size;
    }
    
    /* Detect CPU features */
    detect_cpu_features();
    
    /* Get total RAM - platform specific */
    #ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            info->ram_bytes = status.ullTotalPhys;
        }
    #elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
        #include <unistd.h>
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            info->ram_bytes = (uint)pages * (uint)page_size;
        }
    #endif
    
    /* Detect GPU via OpenCL */
    detect_gpu_opencl(info);
    
    /* Generate hardware fingerprint */
    generate_fingerprint(info);
    
    /* Cache the detected info */
    memcpy(&g_hw_info, info, sizeof(fb_hardware_info_t));
    
    return 0;
}

int fb_hw_get_fingerprint(char fingerprint[64]) {
    if (!g_hw_initialized) {
        if (fb_hw_detect(&g_hw_info) != 0) {
            return -1;
        }
    }
    
    strncpy(fingerprint, g_hw_info.fingerprint, 64);
    return 0;
}

bool fb_hw_fingerprints_compatible(const char *fp1, const char *fp2) {
    /* Exact match required for now */
    /* Future: Could allow minor variations (same CPU family, different stepping) */
    return strcmp(fp1, fp2) == 0;
}

int fb_hw_get_description(char *buffer, size_t buffer_size) {
    if (!g_hw_initialized) {
        if (fb_hw_detect(&g_hw_info) != 0) {
            return -1;
        }
    }
    
    int written = snprintf(buffer, buffer_size,
        "CPU: %s %s (%d cores, %d threads)\n"
        "L1: %zu KB, L2: %zu KB, L3: %zu KB\n"
        "RAM: %.2f GB\n",
        g_hw_info.cpu_vendor,
        g_hw_info.cpu_model,
        g_hw_info.cpu_cores,
        g_hw_info.cpu_threads,
        g_hw_info.cpu_l1_cache / 1024,
        g_hw_info.cpu_l2_cache / 1024,
        g_hw_info.cpu_l3_cache / 1024,
        g_hw_info.ram_bytes / (1024.0 * 1024.0 * 1024.0)
    );
    
    if (g_hw_info.has_gpu && written < (int)buffer_size) {
        written += snprintf(buffer + written, buffer_size - written,
            "GPU: %s %s (%d CUs, %.2f GB VRAM)\n",
            g_hw_info.gpu_vendor,
            g_hw_info.gpu_model,
            g_hw_info.gpu_compute_units,
            g_hw_info.gpu_memory_bytes / (1024.0 * 1024.0 * 1024.0)
        );
    }
    
    if (written < (int)buffer_size) {
        written += snprintf(buffer + written, buffer_size - written,
            "Fingerprint: %s", g_hw_info.fingerprint
        );
    }
    
    return written;
}

uint32_t fb_hw_get_cpu_features(void) {
    if (!g_hw_initialized) {
        fb_hw_detect(&g_hw_info);
    }
    return g_cpu_features;
}

bool fb_hw_has_cpu_feature(fb_cpu_feature_t feature) {
    return (fb_hw_get_cpu_features() & feature) != 0;
}

fb_gpu_vendor_t fb_hw_get_gpu_vendor(void) {
    if (!g_hw_initialized) {
        fb_hw_detect(&g_hw_info);
    }
    
    if (!g_hw_info.has_gpu) {
        return FB_GPU_VENDOR_NONE;
    }
    
    /* Parse vendor string */
    if (strstr(g_hw_info.gpu_vendor, "NVIDIA") || strstr(g_hw_info.gpu_vendor, "nvidia")) {
        return FB_GPU_VENDOR_NVIDIA;
    } else if (strstr(g_hw_info.gpu_vendor, "AMD") || strstr(g_hw_info.gpu_vendor, "Advanced Micro Devices")) {
        return FB_GPU_VENDOR_AMD;
    } else if (strstr(g_hw_info.gpu_vendor, "Intel")) {
        return FB_GPU_VENDOR_INTEL;
    } else if (strstr(g_hw_info.gpu_vendor, "Apple")) {
        return FB_GPU_VENDOR_APPLE;
    }
    
    return FB_GPU_VENDOR_UNKNOWN;
}

int fb_hw_get_gpu_compute_capability(void) {
    if (!g_hw_initialized) {
        fb_hw_detect(&g_hw_info);
    }
    
    if (!g_hw_info.has_gpu) {
        return 0;
    }
    
    /* For now, return compute units count */
    /* Future: Query actual compute capability from CUDA/ROCm if available */
    return g_hw_info.gpu_compute_units;
}

void fb_hw_cleanup(void) {
    if (g_hw_initialized) {
        cpuinfo_deinitialize();
        g_hw_initialized = false;
    }
}
