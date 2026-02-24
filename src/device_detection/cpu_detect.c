/**
 * @file cpu_detect.c
 * @brief CPU Detection Implementation
 */

#include "device_detection/cpu_detect.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sched.h>
#endif

// ============================================================================
// x86_64 CPUID Intrinsics
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define HAS_X86_CPUID

static void cpuid(uint32_t function, uint32_t subfunction, 
                  uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, function, subfunction);
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#else
    __asm__ __volatile__(
        "cpuid"
        : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
        : "a"(function), "c"(subfunction)
    );
#endif
}

static void cpuid_simple(uint32_t function, uint32_t* eax, uint32_t* ebx, 
                         uint32_t* ecx, uint32_t* edx) {
    cpuid(function, 0, eax, ebx, ecx, edx);
}

#endif // HAS_X86_CPUID

// ============================================================================
// CPU Vendor Detection
// ============================================================================

fb_cpu_vendor_t fb_get_cpu_vendor(void) {
#ifdef HAS_X86_CPUID
    uint32_t eax, ebx, ecx, edx;
    char vendor[13] = {0};
    
    cpuid_simple(0, &eax, &ebx, &ecx, &edx);
    
    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    
    if (strcmp(vendor, "GenuineIntel") == 0) {
        return CPU_VENDOR_INTEL;
    } else if (strcmp(vendor, "AuthenticAMD") == 0) {
        return CPU_VENDOR_AMD;
    } else {
        return CPU_VENDOR_OTHER;
    }
    
#elif defined(__aarch64__) || defined(__arm__)
    // ARM vendor detection via /proc/cpuinfo or sysctl
#ifdef __APPLE__
    return CPU_VENDOR_APPLE;
#elif defined(__linux__)
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "CPU implementer")) {
                unsigned int implementer;
                if (sscanf(line, "CPU implementer\t: 0x%x", &implementer) == 1) {
                    fclose(fp);
                    switch (implementer) {
                        case 0x41: return CPU_VENDOR_ARM;
                        case 0x61: return CPU_VENDOR_APPLE;
                        case 0xC0: return CPU_VENDOR_AMPERE;
                        case 0x4E: return CPU_VENDOR_NVIDIA;
                        case 0x51: return CPU_VENDOR_QUALCOMM;
                        case 0x43: return CPU_VENDOR_CAVIUM;
                        default: return CPU_VENDOR_OTHER;
                    }
                }
            }
        }
        fclose(fp);
    }
#endif
    return CPU_VENDOR_ARM;
    
#else
    return CPU_VENDOR_UNKNOWN;
#endif
}

fb_cpu_arch_t fb_get_cpu_arch(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return CPU_ARCH_X86_64;
#elif defined(__aarch64__) || defined(__arm64__)
    return CPU_ARCH_ARM64;
#elif defined(__arm__)
    return CPU_ARCH_ARMV7;
#elif defined(__riscv)
    return CPU_ARCH_RISCV;
#elif defined(__powerpc64__)
    return CPU_ARCH_POWER;
#else
    return CPU_ARCH_UNKNOWN;
#endif
}

// ============================================================================
// x86_64 Detection
// ============================================================================

#ifdef HAS_X86_CPUID

int fb_detect_x86_cpu(fb_cpu_info_t* cpu_info) {
    uint32_t eax, ebx, ecx, edx;
    
    // Vendor string
    cpuid_simple(0, &eax, &ebx, &ecx, &edx);
    memcpy(cpu_info->vendor_string + 0, &ebx, 4);
    memcpy(cpu_info->vendor_string + 4, &edx, 4);
    memcpy(cpu_info->vendor_string + 8, &ecx, 4);
    cpu_info->vendor_string[12] = '\0';
    
    // Brand string (requires CPUID function 0x80000002-0x80000004)
    uint32_t max_extended;
    cpuid_simple(0x80000000, &max_extended, &ebx, &ecx, &edx);
    
    if (max_extended >= 0x80000004) {
        uint32_t* brand = (uint32_t*)cpu_info->brand_string;
        cpuid_simple(0x80000002, &brand[0], &brand[1], &brand[2], &brand[3]);
        cpuid_simple(0x80000003, &brand[4], &brand[5], &brand[6], &brand[7]);
        cpuid_simple(0x80000004, &brand[8], &brand[9], &brand[10], &brand[11]);
        cpu_info->brand_string[48] = '\0';
        
        // Trim leading spaces
        char* src = cpu_info->brand_string;
        while (*src == ' ') src++;
        if (src != cpu_info->brand_string) {
            memmove(cpu_info->brand_string, src, strlen(src) + 1);
        }
    }
    
    // Family, Model, Stepping
    cpuid_simple(1, &eax, &ebx, &ecx, &edx);
    cpu_info->stepping = eax & 0xF;
    cpu_info->model = (eax >> 4) & 0xF;
    cpu_info->family = (eax >> 8) & 0xF;
    
    // Extended model and family
    if (cpu_info->family == 0xF) {
        cpu_info->family += (eax >> 20) & 0xFF;
    }
    if (cpu_info->family == 0xF || cpu_info->family == 0x6) {
        cpu_info->model += ((eax >> 16) & 0xF) << 4;
    }
    
    // SIMD capabilities (CPUID function 1, ECX and EDX)
    cpu_info->simd.x86.mmx = (edx >> 23) & 1;
    cpu_info->simd.x86.sse = (edx >> 25) & 1;
    cpu_info->simd.x86.sse2 = (edx >> 26) & 1;
    cpu_info->simd.x86.sse3 = (ecx >> 0) & 1;
    cpu_info->simd.x86.ssse3 = (ecx >> 9) & 1;
    cpu_info->simd.x86.sse4_1 = (ecx >> 19) & 1;
    cpu_info->simd.x86.sse4_2 = (ecx >> 20) & 1;
    cpu_info->simd.x86.avx = (ecx >> 28) & 1;
    cpu_info->simd.x86.fma = (ecx >> 12) & 1;
    
    // Extended features (CPUID function 7, subfunction 0)
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    cpu_info->simd.x86.avx2 = (ebx >> 5) & 1;
    cpu_info->simd.x86.avx512f = (ebx >> 16) & 1;
    cpu_info->simd.x86.avx512dq = (ebx >> 17) & 1;
    cpu_info->simd.x86.avx512ifma = (ebx >> 21) & 1;
    cpu_info->simd.x86.avx512pf = (ebx >> 26) & 1;
    cpu_info->simd.x86.avx512er = (ebx >> 27) & 1;
    cpu_info->simd.x86.avx512cd = (ebx >> 28) & 1;
    cpu_info->simd.x86.avx512bw = (ebx >> 30) & 1;
    cpu_info->simd.x86.avx512vl = (ebx >> 31) & 1;
    cpu_info->simd.x86.avx512_vnni = (ecx >> 11) & 1;
    cpu_info->simd.x86.avx512_bf16 = (eax >> 5) & 1; // From subfunction 1
    cpu_info->simd.x86.amx_tile = (edx >> 24) & 1;
    cpu_info->simd.x86.amx_int8 = (edx >> 25) & 1;
    cpu_info->simd.x86.amx_bf16 = (edx >> 22) & 1;
    
    // AMD-specific features
    if (cpu_info->vendor == CPU_VENDOR_AMD) {
        cpuid_simple(0x80000001, &eax, &ebx, &ecx, &edx);
        cpu_info->simd.x86.fma4 = (ecx >> 16) & 1;
    }
    
    // Hyperthreading
    cpuid_simple(1, &eax, &ebx, &ecx, &edx);
    cpu_info->has_hyperthreading = (edx >> 28) & 1;
    
    return 0;
}

#endif // HAS_X86_CPUID

// ============================================================================
// ARM Detection
// ============================================================================

#if defined(__aarch64__) || defined(__arm__)

int fb_detect_arm_cpu(fb_cpu_info_t* cpu_info) {
#ifdef __APPLE__
    // Use sysctl for Apple Silicon
    size_t len;
    char brand[128];
    
    len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, NULL, 0) == 0) {
        strncpy(cpu_info->brand_string, brand, sizeof(cpu_info->brand_string) - 1);
    }
    
    // NEON is standard on all modern ARM
    cpu_info->simd.arm.neon = true;
    
    // Apple Silicon specific features
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    cpu_info->simd.arm.fp16 = true;
#endif
    
#elif defined(__linux__)
    // Parse /proc/cpuinfo
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "model name")) {
                char* colon = strchr(line, ':');
                if (colon) {
                    char* name = colon + 2; // Skip ": "
                    strncpy(cpu_info->brand_string, name, sizeof(cpu_info->brand_string) - 1);
                    // Remove newline
                    cpu_info->brand_string[strcspn(cpu_info->brand_string, "\n")] = 0;
                }
            }
            
            if (strstr(line, "Features")) {
                if (strstr(line, "neon")) cpu_info->simd.arm.neon = true;
                if (strstr(line, "sve")) cpu_info->simd.arm.sve = true;
                if (strstr(line, "sve2")) cpu_info->simd.arm.sve2 = true;
                if (strstr(line, "fp16")) cpu_info->simd.arm.fp16 = true;
                if (strstr(line, "bf16")) cpu_info->simd.arm.bf16 = true;
                if (strstr(line, "i8mm")) cpu_info->simd.arm.i8mm = true;
                if (strstr(line, "dotprod")) cpu_info->simd.arm.dotprod = true;
            }
        }
        fclose(fp);
    }
    
    // Compiler-time feature detection as fallback
#ifdef __ARM_NEON
    cpu_info->simd.arm.neon = true;
#endif
#ifdef __ARM_FEATURE_SVE
    cpu_info->simd.arm.sve = true;
#endif
#ifdef __ARM_FEATURE_SVE2
    cpu_info->simd.arm.sve2 = true;
#endif
    
#endif // __linux__
    
    return 0;
}

#endif // ARM

// ============================================================================
// CPU Topology Detection
// ============================================================================

int fb_detect_cpu_topology(fb_cpu_topology_t* topology) {
    memset(topology, 0, sizeof(*topology));
    
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    topology->logical_cores = sysinfo.dwNumberOfProcessors;
    
    // Get physical cores via GetLogicalProcessorInformation
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (len > 0) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = 
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(len);
        if (buffer && GetLogicalProcessorInformation(buffer, &len)) {
            DWORD count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            uint32_t physical = 0;
            for (DWORD i = 0; i < count; i++) {
                if (buffer[i].Relationship == RelationProcessorCore) {
                    physical++;
                }
            }
            topology->physical_cores = physical;
        }
        free(buffer);
    }
    
#elif defined(__APPLE__)
    int32_t physical, logical;
    size_t len;
    
    len = sizeof(physical);
    sysctlbyname("hw.physicalcpu", &physical, &len, NULL, 0);
    topology->physical_cores = physical;
    
    len = sizeof(logical);
    sysctlbyname("hw.logicalcpu", &logical, &len, NULL, 0);
    topology->logical_cores = logical;
    
#elif defined(__linux__)
    // Use sysconf
    topology->logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    
    // Count physical cores from /sys/devices/system/cpu/cpu*/topology/core_id
    uint32_t max_core_id = 0;
    for (uint32_t cpu = 0; cpu < topology->logical_cores; cpu++) {
        char path[256];
        snprintf(path, sizeof(path), 
                 "/sys/devices/system/cpu/cpu%u/topology/core_id", cpu);
        FILE* fp = fopen(path, "r");
        if (fp) {
            uint32_t core_id;
            if (fscanf(fp, "%u", &core_id) == 1) {
                if (core_id > max_core_id) max_core_id = core_id;
            }
            fclose(fp);
        }
    }
    topology->physical_cores = max_core_id + 1;
    
#else
    // Fallback: assume no hyperthreading
    topology->logical_cores = 1;
    topology->physical_cores = 1;
#endif
    
    // Calculate derived values
    if (topology->physical_cores > 0) {
        topology->threads_per_core = topology->logical_cores / topology->physical_cores;
    }
    topology->sockets = 1; // Default, can be refined
    topology->numa_nodes = 1; // Default
    
    return 0;
}

// ============================================================================
// CPU Frequency Detection
// ============================================================================

int fb_detect_cpu_frequency(uint32_t* base_freq_mhz, uint32_t* max_freq_mhz) {
    *base_freq_mhz = 0;
    *max_freq_mhz = 0;
    
#ifdef _WIN32
    // On Windows, query registry for nominal frequency
    HKEY hKey;
    if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD freq_mhz = 0;
        DWORD size = sizeof(freq_mhz);
        if (RegQueryValueEx(hKey, "~MHz", NULL, NULL, (LPBYTE)&freq_mhz, &size) == ERROR_SUCCESS) {
            *base_freq_mhz = freq_mhz;
            *max_freq_mhz = freq_mhz;  // Use base as max estimate
        }
        RegCloseKey(hKey);
    }
    
#elif defined(__APPLE__)
    // On macOS, use sysctl
    uint64_t freq_hz = 0;
    size_t size = sizeof(freq_hz);
    if (sysctlbyname("hw.cpufrequency", &freq_hz, &size, NULL, 0) == 0) {
        *base_freq_mhz = (uint32_t)(freq_hz / 1000000);
        *max_freq_mhz = *base_freq_mhz;
    }
    
    // Try max frequency
    if (sysctlbyname("hw.cpufrequency_max", &freq_hz, &size, NULL, 0) == 0) {
        *max_freq_mhz = (uint32_t)(freq_hz / 1000000);
    }
    
#elif defined(__linux__)
    // On Linux, read from /sys/devices/system/cpu/cpu0/cpufreq/
    FILE* fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency", "r");
    if (fp) {
        uint32_t freq_khz;
        if (fscanf(fp, "%u", &freq_khz) == 1) {
            *base_freq_mhz = freq_khz / 1000;
        }
        fclose(fp);
    }
    
    fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
    if (fp) {
        uint32_t freq_khz;
        if (fscanf(fp, "%u", &freq_khz) == 1) {
            *max_freq_mhz = freq_khz / 1000;
        }
        fclose(fp);
    }
    
    // Fallback: try scaling_max_freq
    if (*max_freq_mhz == 0) {
        fp = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", "r");
        if (fp) {
            uint32_t freq_khz;
            if (fscanf(fp, "%u", &freq_khz) == 1) {
                *max_freq_mhz = freq_khz / 1000;
                if (*base_freq_mhz == 0) *base_freq_mhz = *max_freq_mhz;
            }
            fclose(fp);
        }
    }
#endif
    
    return (*base_freq_mhz > 0 || *max_freq_mhz > 0) ? 0 : -1;
}

// ============================================================================
// Cache Detection
// ============================================================================

int fb_detect_cache_hierarchy(fb_cache_hierarchy_t* cache) {
    memset(cache, 0, sizeof(*cache));
    
#ifdef HAS_X86_CPUID
    uint32_t eax, ebx, ecx, edx;
    
    // Intel cache detection (CPUID function 4)
    cpuid_simple(0, &eax, &ebx, &ecx, &edx);
    if (eax >= 4) {
        for (uint32_t i = 0; ; i++) {
            cpuid(4, i, &eax, &ebx, &ecx, &edx);
            
            uint32_t cache_type = eax & 0x1F;
            if (cache_type == 0) break; // No more caches
            
            uint32_t level = (eax >> 5) & 0x7;
            uint32_t line_size = (ebx & 0xFFF) + 1;
            uint32_t partitions = ((ebx >> 12) & 0x3FF) + 1;
            uint32_t ways = ((ebx >> 22) & 0x3FF) + 1;
            uint32_t sets = ecx + 1;
            uint32_t size_kb = (ways * partitions * line_size * sets) / 1024;
            
            fb_cache_level_t* target = NULL;
            switch (level) {
                case 1:
                    if (cache_type == 1) target = &cache->l1_data;
                    else if (cache_type == 2) target = &cache->l1_inst;
                    break;
                case 2: target = &cache->l2; break;
                case 3: target = &cache->l3; break;
                case 4: target = &cache->l4; break;
            }
            
            if (target) {
                target->size_kb = size_kb;
                target->line_size = line_size;
                target->ways = ways;
            }
        }
    }
    
#elif defined(__APPLE__)
    size_t len;
    uint64_t value;
    
    len = sizeof(value);
    if (sysctlbyname("hw.l1dcachesize", &value, &len, NULL, 0) == 0) {
        cache->l1_data.size_kb = value / 1024;
    }
    if (sysctlbyname("hw.l1icachesize", &value, &len, NULL, 0) == 0) {
        cache->l1_inst.size_kb = value / 1024;
    }
    if (sysctlbyname("hw.l2cachesize", &value, &len, NULL, 0) == 0) {
        cache->l2.size_kb = value / 1024;
    }
    if (sysctlbyname("hw.l3cachesize", &value, &len, NULL, 0) == 0) {
        cache->l3.size_kb = value / 1024;
    }
    
    // Cache line size
    if (sysctlbyname("hw.cachelinesize", &value, &len, NULL, 0) == 0) {
        cache->l1_data.line_size = value;
        cache->l1_inst.line_size = value;
        cache->l2.line_size = value;
        cache->l3.line_size = value;
    }
    
#elif defined(__linux__)
    // Read from /sys/devices/system/cpu/cpu0/cache/index*
    for (int idx = 0; idx < 10; idx++) {
        char path[256];
        FILE* fp;
        
        // Level
        snprintf(path, sizeof(path), 
                 "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
        fp = fopen(path, "r");
        if (!fp) break;
        
        uint32_t level;
        fscanf(fp, "%u", &level);
        fclose(fp);
        
        // Type
        snprintf(path, sizeof(path), 
                 "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
        fp = fopen(path, "r");
        if (!fp) continue;
        
        char type[32];
        fscanf(fp, "%s", type);
        fclose(fp);
        
        // Size
        snprintf(path, sizeof(path), 
                 "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
        fp = fopen(path, "r");
        if (!fp) continue;
        
        uint32_t size_kb;
        char unit[4];
        if (fscanf(fp, "%u%s", &size_kb, unit) == 2) {
            if (strcmp(unit, "M") == 0) size_kb *= 1024;
        }
        fclose(fp);
        
        // Select target cache level
        fb_cache_level_t* target = NULL;
        if (level == 1 && strcmp(type, "Data") == 0) {
            target = &cache->l1_data;
        } else if (level == 1 && strcmp(type, "Instruction") == 0) {
            target = &cache->l1_inst;
        } else if (level == 2) {
            target = &cache->l2;
        } else if (level == 3) {
            target = &cache->l3;
        }
        
        if (target) {
            target->size_kb = size_kb;
        }
    }
#endif
    
    return 0;
}

// ============================================================================
// Main Detection Function
// ============================================================================

int fb_detect_cpu(fb_cpu_info_t* cpu_info) {
    memset(cpu_info, 0, sizeof(*cpu_info));
    
    cpu_info->vendor = fb_get_cpu_vendor();
    cpu_info->architecture = fb_get_cpu_arch();
    
    // Architecture-specific detection
    switch (cpu_info->architecture) {
#ifdef HAS_X86_CPUID
        case CPU_ARCH_X86_64:
            fb_detect_x86_cpu(cpu_info);
            break;
#endif
            
#if defined(__aarch64__) || defined(__arm__)
        case CPU_ARCH_ARM64:
        case CPU_ARCH_ARMV7:
            fb_detect_arm_cpu(cpu_info);
            break;
#endif
            
        default:
            break;
    }
    
    // Common detection (works on all platforms)
    fb_detect_cpu_topology(&cpu_info->topology);
    fb_detect_cache_hierarchy(&cpu_info->cache);
    fb_detect_cpu_frequency(&cpu_info->base_freq_mhz, &cpu_info->max_freq_mhz);
    
    // Estimate GFLOPS
    fb_estimate_cpu_gflops(cpu_info, 
                          &cpu_info->estimated_gflops_sp,
                          &cpu_info->estimated_gflops_dp);
    
    return 0;
}

// ============================================================================
// GFLOPS Estimation
// ============================================================================

void fb_estimate_cpu_gflops(const fb_cpu_info_t* cpu_info, 
                            double* sp_gflops, double* dp_gflops) {
    // Estimate based on core count, frequency, and SIMD width
    
    double base_freq_ghz = cpu_info->max_freq_mhz / 1000.0;
    if (base_freq_ghz == 0) base_freq_ghz = 3.0; // Fallback estimate
    
    uint32_t cores = cpu_info->topology.physical_cores;
    if (cores == 0) cores = 1;
    
    double flops_per_cycle_sp = 2; // Base FMA (2 ops)
    double flops_per_cycle_dp = 2;
    
    if (cpu_info->architecture == CPU_ARCH_X86_64) {
        // x86_64 SIMD width multiplier
        if (cpu_info->simd.x86.avx512f) {
            flops_per_cycle_sp *= 32; // 512-bit / 16-bit SP = 16 lanes, 2 ops/cycle FMA
            flops_per_cycle_dp *= 16;
        } else if (cpu_info->simd.x86.avx2) {
            flops_per_cycle_sp *= 16; // 256-bit / 16-bit = 8 lanes
            flops_per_cycle_dp *= 8;
        } else if (cpu_info->simd.x86.avx) {
            flops_per_cycle_sp *= 16;
            flops_per_cycle_dp *= 8;
        } else if (cpu_info->simd.x86.sse2) {
            flops_per_cycle_sp *= 8;
            flops_per_cycle_dp *= 4;
        }
    } else if (cpu_info->architecture == CPU_ARCH_ARM64) {
        // ARM NEON/SVE
        if (cpu_info->simd.arm.sve2) {
            // SVE2: variable width, assume 256-bit
            flops_per_cycle_sp *= 16;
            flops_per_cycle_dp *= 8;
        } else if (cpu_info->simd.arm.neon) {
            // NEON: 128-bit
            flops_per_cycle_sp *= 8;
            flops_per_cycle_dp *= 4;
        }
    }
    
    *sp_gflops = cores * base_freq_ghz * flops_per_cycle_sp;
    *dp_gflops = cores * base_freq_ghz * flops_per_cycle_dp;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool fb_has_x86_instruction_set(const char* instruction_set) {
    fb_cpu_info_t cpu_info;
    fb_detect_cpu(&cpu_info);
    
    if (cpu_info.architecture != CPU_ARCH_X86_64) return false;
    
    if (strcmp(instruction_set, "AVX512F") == 0) return cpu_info.simd.x86.avx512f;
    if (strcmp(instruction_set, "AVX2") == 0) return cpu_info.simd.x86.avx2;
    if (strcmp(instruction_set, "AVX") == 0) return cpu_info.simd.x86.avx;
    if (strcmp(instruction_set, "SSE4.2") == 0) return cpu_info.simd.x86.sse4_2;
    if (strcmp(instruction_set, "FMA") == 0) return cpu_info.simd.x86.fma;
    
    return false;
}

uint32_t fb_get_recommended_thread_count(const fb_cpu_info_t* cpu_info) {
    // For BLAS, use physical cores (avoid hyperthreading overhead)
    uint32_t count = cpu_info->topology.physical_cores;
    if (count == 0) count = cpu_info->topology.logical_cores;
    if (count == 0) count = 1;
    
    return count;
}

void fb_print_cpu_info(const fb_cpu_info_t* cpu_info) {
    printf("CPU Information:\n");
    printf("  Vendor: %s\n", cpu_info->vendor_string);
    printf("  Model: %s\n", cpu_info->brand_string);
    printf("  Family: %u, Model: %u, Stepping: %u\n", 
           cpu_info->family, cpu_info->model, cpu_info->stepping);
    
    printf("\nTopology:\n");
    printf("  Physical Cores: %u\n", cpu_info->topology.physical_cores);
    printf("  Logical Cores: %u\n", cpu_info->topology.logical_cores);
    printf("  Threads per Core: %u\n", cpu_info->topology.threads_per_core);
    
    printf("\nCache:\n");
    printf("  L1 Data: %u KB\n", cpu_info->cache.l1_data.size_kb);
    printf("  L1 Instruction: %u KB\n", cpu_info->cache.l1_inst.size_kb);
    printf("  L2: %u KB\n", cpu_info->cache.l2.size_kb);
    printf("  L3: %u KB\n", cpu_info->cache.l3.size_kb);
    
    if (cpu_info->architecture == CPU_ARCH_X86_64) {
        printf("\nSIMD Capabilities (x86):\n");
        printf("  SSE: %s\n", cpu_info->simd.x86.sse ? "Yes" : "No");
        printf("  AVX: %s\n", cpu_info->simd.x86.avx ? "Yes" : "No");
        printf("  AVX2: %s\n", cpu_info->simd.x86.avx2 ? "Yes" : "No");
        printf("  AVX-512F: %s\n", cpu_info->simd.x86.avx512f ? "Yes" : "No");
        printf("  FMA: %s\n", cpu_info->simd.x86.fma ? "Yes" : "No");
        printf("  AMX: %s\n", cpu_info->simd.x86.amx_tile ? "Yes" : "No");
    } else if (cpu_info->architecture == CPU_ARCH_ARM64) {
        printf("\nSIMD Capabilities (ARM):\n");
        printf("  NEON: %s\n", cpu_info->simd.arm.neon ? "Yes" : "No");
        printf("  SVE: %s\n", cpu_info->simd.arm.sve ? "Yes" : "No");
        printf("  SVE2: %s\n", cpu_info->simd.arm.sve2 ? "Yes" : "No");
        printf("  FP16: %s\n", cpu_info->simd.arm.fp16 ? "Yes" : "No");
    }
    
    printf("\nEstimated Performance:\n");
    printf("  Single Precision: %.1f GFLOPS\n", cpu_info->estimated_gflops_sp);
    printf("  Double Precision: %.1f GFLOPS\n", cpu_info->estimated_gflops_dp);
}
