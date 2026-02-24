#include "benchmark_system.h"
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <sys/utsname.h>
#include <cpuid.h>
#endif

// FNV-1a hash function for string hashing
static uint32_t hash_string(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint8_t)(*str++);
        hash *= 16777619u;
    }
    return hash;
}

// Get CPU vendor and model string
static void get_cpu_info(uint32_t* vendor_hash, uint16_t* family, uint16_t* model) {
    char vendor_str[64] = {0};
    
#ifdef _WIN32
    int cpuinfo[4];
    __cpuid(cpuinfo, 0);
    memcpy(vendor_str, &cpuinfo[1], 4);
    memcpy(vendor_str + 4, &cpuinfo[3], 4);
    memcpy(vendor_str + 8, &cpuinfo[2], 4);
    
    __cpuid(cpuinfo, 1);
    *family = (cpuinfo[0] >> 8) & 0xF;
    *model = (cpuinfo[0] >> 4) & 0xF;
    
    // Extended family/model
    if (*family == 0xF) {
        *family += (cpuinfo[0] >> 20) & 0xFF;
    }
    if (*family == 0xF || *family == 0x6) {
        *model += ((cpuinfo[0] >> 16) & 0xF) << 4;
    }
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        memcpy(vendor_str, &ebx, 4);
        memcpy(vendor_str + 4, &edx, 4);
        memcpy(vendor_str + 8, &ecx, 4);
    }
    
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        *family = (eax >> 8) & 0xF;
        *model = (eax >> 4) & 0xF;
        
        if (*family == 0xF) {
            *family += (eax >> 20) & 0xFF;
        }
        if (*family == 0xF || *family == 0x6) {
            *model += ((eax >> 16) & 0xF) << 4;
        }
    }
#endif
    
    *vendor_hash = hash_string(vendor_str);
}

// Get GPU device IDs (simplified - actual implementation would query CUDA/ROCm/etc.)
static void get_gpu_device_ids(uint32_t device_ids[4]) {
    // TODO: Query actual GPU devices via CUDA/ROCm/Level Zero APIs
    // For now, zero out (will be implemented in backend integration)
    memset(device_ids, 0, 4 * sizeof(uint32_t));
}

// Get system memory size
static uint32_t get_memory_size_mb(void) {
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        return (uint32_t)(statex.ullTotalPhys / (1024 * 1024));
    }
#else
    // Linux/macOS: read from /proc/meminfo or sysconf
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return (uint32_t)((pages * page_size) / (1024 * 1024));
    }
#endif
    return 0;
}

// Get backend version hash
static uint32_t get_backend_version_hash(void) {
    // TODO: Query actual backend versions
    // For now, return a placeholder
    // Will be implemented when backends provide version info
    return hash_string("backends:v1.0");
}

int fb_generate_hardware_fingerprint(fb_hardware_fingerprint_t* fingerprint) {
    if (!fingerprint) {
        return -1;
    }
    
    memset(fingerprint, 0, sizeof(fb_hardware_fingerprint_t));
    
    // CPU information
    get_cpu_info(&fingerprint->cpu_vendor_hash, 
                 &fingerprint->cpu_family, 
                 &fingerprint->cpu_model);
    
    // GPU device IDs
    get_gpu_device_ids(fingerprint->gpu_device_ids);
    
    // System memory
    fingerprint->memory_size_mb = get_memory_size_mb();
    
    // Backend versions
    fingerprint->backend_version_hash = get_backend_version_hash();
    
    // Schema version (increment when data format changes)
    fingerprint->schema_version = 1;
    
    return 0;
}

bool fb_fingerprints_match(const fb_hardware_fingerprint_t* a,
                           const fb_hardware_fingerprint_t* b) {
    if (!a || !b) {
        return false;
    }
    
    // Schema version must match exactly
    if (a->schema_version != b->schema_version) {
        return false;
    }
    
    // Backend versions must match exactly
    if (a->backend_version_hash != b->backend_version_hash) {
        return false;
    }
    
    // CPU must match
    if (a->cpu_vendor_hash != b->cpu_vendor_hash ||
        a->cpu_family != b->cpu_family ||
        a->cpu_model != b->cpu_model) {
        return false;
    }
    
    // GPU devices must match
    if (memcmp(a->gpu_device_ids, b->gpu_device_ids, 
               sizeof(a->gpu_device_ids)) != 0) {
        return false;
    }
    
    // Memory size can differ slightly (within 5%)
    uint32_t mem_diff = (a->memory_size_mb > b->memory_size_mb) ?
                        (a->memory_size_mb - b->memory_size_mb) :
                        (b->memory_size_mb - a->memory_size_mb);
    uint32_t mem_avg = (a->memory_size_mb + b->memory_size_mb) / 2;
    if (mem_avg > 0 && (mem_diff * 100 / mem_avg) > 5) {
        return false;
    }
    
    return true;
}
