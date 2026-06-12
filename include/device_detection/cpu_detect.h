/**
 * @file cpu_detect.h
 * @brief CPU Detection and Capability Discovery
 * 
 * Detects CPU vendor, model, core topology, cache hierarchy, and SIMD capabilities
 * across x86_64, ARM, and other architectures.
 */

#ifndef FASTER_BLASTER_CPU_DETECT_H
#define FASTER_BLASTER_CPU_DETECT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// CPU Vendor IDs
typedef enum {
    CPU_VENDOR_UNKNOWN = 0,
    CPU_VENDOR_INTEL,
    CPU_VENDOR_AMD,
    CPU_VENDOR_APPLE,
    CPU_VENDOR_ARM,
    CPU_VENDOR_AMPERE,
    CPU_VENDOR_QUALCOMM,
    CPU_VENDOR_NVIDIA,     // Grace CPU
    CPU_VENDOR_CAVIUM,
    CPU_VENDOR_OTHER
} fb_cpu_vendor_t;

// CPU Architecture
typedef enum {
    CPU_ARCH_UNKNOWN = 0,
    CPU_ARCH_X86_64,
    CPU_ARCH_ARM64,
    CPU_ARCH_ARMV7,
    CPU_ARCH_RISCV,
    CPU_ARCH_POWER,
    CPU_ARCH_OTHER
} fb_cpu_arch_t;

// SIMD Instruction Sets (x86_64)
typedef struct {
    bool mmx;
    bool sse;
    bool sse2;
    bool sse3;
    bool ssse3;
    bool sse4_1;
    bool sse4_2;
    bool avx;
    bool avx2;
    bool avx512f;         // AVX-512 Foundation
    bool avx512dq;        // Doubleword and Quadword
    bool avx512ifma;      // Integer Fused Multiply-Add
    bool avx512pf;        // Prefetch
    bool avx512er;        // Exponential and Reciprocal
    bool avx512cd;        // Conflict Detection
    bool avx512bw;        // Byte and Word
    bool avx512vl;        // Vector Length Extensions
    bool avx512_vnni;     // Vector Neural Network Instructions
    bool avx512_bf16;     // Brain Float16
    bool avx512_fp16;     // Float16
    bool fma;             // Fused Multiply-Add
    bool fma4;            // AMD FMA4
    bool amx_tile;        // Advanced Matrix Extensions
    bool amx_int8;
    bool amx_bf16;
} fb_x86_simd_caps_t;

// ARM SIMD Capabilities
typedef struct {
    bool neon;            // ARM NEON
    bool sve;             // Scalable Vector Extension
    bool sve2;
    uint32_t sve_vl;      // SVE vector length in bits
    bool fp16;            // Half precision
    bool bf16;            // BrainFloat16
    bool i8mm;            // Integer Matrix Multiply
    bool dotprod;         // Dot Product
} fb_arm_simd_caps_t;

// Cache Level Information
typedef struct {
    uint32_t size_kb;     // Cache size in KB
    uint32_t line_size;   // Cache line size in bytes
    uint32_t ways;        // Associativity
    bool is_inclusive;    // Inclusive of lower levels
} fb_cache_level_t;

// Cache Hierarchy
typedef struct {
    fb_cache_level_t l1_data;
    fb_cache_level_t l1_inst;
    fb_cache_level_t l2;
    fb_cache_level_t l3;
    fb_cache_level_t l4;  // If available (e.g., Intel Xeon)
} fb_cache_hierarchy_t;

// CPU Topology
typedef struct {
    uint32_t physical_cores;    // Physical cores
    uint32_t logical_cores;     // Logical cores (with SMT/HT)
    uint32_t sockets;           // Number of CPU sockets
    uint32_t numa_nodes;        // NUMA nodes
    uint32_t cores_per_socket;
    uint32_t threads_per_core;
} fb_cpu_topology_t;

// Complete CPU Information
typedef struct {
    // Identification
    fb_cpu_vendor_t vendor;
    fb_cpu_arch_t architecture;
    char vendor_string[64];     // e.g., "GenuineIntel"
    char brand_string[128];     // e.g., "Intel(R) Core(TM) i9-13900K"
    uint32_t family;
    uint32_t model;
    uint32_t stepping;
    
    // Topology
    fb_cpu_topology_t topology;
    
    // Cache
    fb_cache_hierarchy_t cache;
    
    // SIMD Capabilities
    union {
        fb_x86_simd_caps_t x86;
        fb_arm_simd_caps_t arm;
    } simd;
    
    // Frequency
    uint32_t base_freq_mhz;
    uint32_t max_freq_mhz;
    uint32_t current_freq_mhz;
    
    // Performance
    double estimated_gflops_sp;  // Single precision GFLOPS
    double estimated_gflops_dp;  // Double precision GFLOPS
    
    // Features
    bool has_hyperthreading;
    bool has_turbo_boost;
    bool is_big_little;          // ARM big.LITTLE
    
} fb_cpu_info_t;

/**
 * @brief Detect CPU and populate information structure
 * 
 * @param cpu_info Pointer to structure to fill with CPU information
 * @return 0 on success, negative error code on failure
 */
int fb_detect_cpu(fb_cpu_info_t* cpu_info);

/**
 * @brief Get CPU vendor
 * 
 * @return CPU vendor enum
 */
fb_cpu_vendor_t fb_get_cpu_vendor(void);

/**
 * @brief Get CPU architecture
 * 
 * @return CPU architecture enum
 */
fb_cpu_arch_t fb_get_cpu_arch(void);

/**
 * @brief Detect x86_64 CPUID information
 * 
 * @param cpu_info CPU information structure to populate
 * @return 0 on success
 */
int fb_detect_x86_cpu(fb_cpu_info_t* cpu_info);

/**
 * @brief Detect ARM CPU information
 * 
 * @param cpu_info CPU information structure to populate
 * @return 0 on success
 */
int fb_detect_arm_cpu(fb_cpu_info_t* cpu_info);

/**
 * @brief Detect cache hierarchy
 * 
 * @param cache Cache hierarchy structure to populate
 * @return 0 on success
 */
int fb_detect_cache_hierarchy(fb_cache_hierarchy_t* cache);

/**
 * @brief Detect CPU topology (cores, threads, NUMA)
 * 
 * @param topology Topology structure to populate
 * @return 0 on success
 */
int fb_detect_cpu_topology(fb_cpu_topology_t* topology);

/**
 * @brief Estimate CPU GFLOPS based on capabilities
 * 
 * @param cpu_info CPU information
 * @param sp_gflops Output: Single precision GFLOPS
 * @param dp_gflops Output: Double precision GFLOPS
 */
void fb_estimate_cpu_gflops(const fb_cpu_info_t* cpu_info, 
                            double* sp_gflops, double* dp_gflops);

/**
 * @brief Print CPU information (for debugging)
 * 
 * @param cpu_info CPU information to print
 */
void fb_print_cpu_info(const fb_cpu_info_t* cpu_info);

/**
 * @brief Check if specific SIMD instruction set is available (x86)
 * 
 * @param instruction_set Name of instruction set ("AVX2", "AVX512F", etc.)
 * @return true if available
 */
bool fb_has_x86_instruction_set(const char* instruction_set);

/**
 * @brief Get recommended thread count for BLAS operations
 * 
 * @param cpu_info CPU information
 * @return Recommended number of threads
 */
uint32_t fb_get_recommended_thread_count(const fb_cpu_info_t* cpu_info);

#ifdef __cplusplus
}
#endif

#endif // FASTER_BLASTER_CPU_DETECT_H
