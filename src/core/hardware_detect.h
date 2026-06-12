/**
 * @file hardware_detect.h
 * @brief Hardware detection and fingerprinting
 * 
 * This module wraps the Rust hardware-query crate to provide hardware
 * detection capabilities to the C codebase. It generates unique fingerprints
 * for hardware configurations to enable hardware-specific calibration caching.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_HARDWARE_DETECT_H
#define FB_HARDWARE_DETECT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration from public API */
typedef struct {
    char cpu_vendor[64];
    char cpu_model[128];
    int cpu_cores;
    int cpu_threads;
    uint64_t cpu_l1_cache;
    uint64_t cpu_l2_cache;
    uint64_t cpu_l3_cache;
    uint64_t ram_bytes;
    
    bool has_gpu;
    char gpu_vendor[64];
    char gpu_model[128];
    int gpu_compute_units;
    uint64_t gpu_memory_bytes;
    
    char fingerprint[64];
} fb_hardware_info_t;

/**
 * @brief Initialize hardware detection system
 * 
 * Must be called before any other hardware detection functions.
 * 
 * @return 0 on success, negative error code on failure
 */
int fb_hw_init(void);

/**
 * @brief Detect and populate hardware information
 * 
 * Queries the system for CPU, GPU, and memory information using the
 * hardware-query Rust crate via FFI.
 * 
 * @param info Output parameter for hardware information
 * @return 0 on success, negative error code on failure
 */
int fb_hw_detect(fb_hardware_info_t *info);

/**
 * @brief Generate a unique fingerprint for the current hardware
 * 
 * Creates a hash of relevant hardware characteristics that affect BLAS
 * performance. Machines with identical fingerprints can share calibration data.
 * 
 * The fingerprint includes:
 * - CPU model and microarchitecture
 * - Number of cores and cache sizes
 * - GPU model and compute capability (if present)
 * - SIMD/vector instruction support (AVX2, AVX-512, NEON, etc.)
 * 
 * @param fingerprint Output buffer (must be at least 64 bytes)
 * @return 0 on success, negative error code on failure
 */
int fb_hw_get_fingerprint(char fingerprint[64]);

/**
 * @brief Check if two hardware configurations are compatible
 * 
 * Determines if calibration data from one hardware configuration can be
 * safely used on another. Generally true if fingerprints match.
 * 
 * @param fp1 First hardware fingerprint
 * @param fp2 Second hardware fingerprint
 * @return true if compatible, false otherwise
 */
bool fb_hw_fingerprints_compatible(const char *fp1, const char *fp2);

/**
 * @brief Get human-readable hardware description
 * 
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 * @return Number of bytes written (excluding null terminator)
 */
int fb_hw_get_description(char *buffer, size_t buffer_size);

/**
 * @brief Check for specific CPU features
 */
typedef enum {
    FB_CPU_FEATURE_SSE2     = 1 << 0,
    FB_CPU_FEATURE_SSE3     = 1 << 1,
    FB_CPU_FEATURE_SSSE3    = 1 << 2,
    FB_CPU_FEATURE_SSE41    = 1 << 3,
    FB_CPU_FEATURE_SSE42    = 1 << 4,
    FB_CPU_FEATURE_AVX      = 1 << 5,
    FB_CPU_FEATURE_AVX2     = 1 << 6,
    FB_CPU_FEATURE_AVX512F  = 1 << 7,
    FB_CPU_FEATURE_FMA      = 1 << 8,
    FB_CPU_FEATURE_NEON     = 1 << 9,
    FB_CPU_FEATURE_SVE      = 1 << 10,
    FB_CPU_FEATURE_AMX      = 1 << 11
} fb_cpu_feature_t;

/**
 * @brief Get CPU feature bitmask
 * 
 * @return Bitmask of supported CPU features
 */
uint32_t fb_hw_get_cpu_features(void);

/**
 * @brief Check if a specific CPU feature is supported
 * 
 * @param feature Feature to check
 * @return true if supported, false otherwise
 */
bool fb_hw_has_cpu_feature(fb_cpu_feature_t feature);

/**
 * @brief GPU vendor enumeration
 */
typedef enum {
    FB_GPU_VENDOR_NONE = 0,
    FB_GPU_VENDOR_NVIDIA,
    FB_GPU_VENDOR_AMD,
    FB_GPU_VENDOR_INTEL,
    FB_GPU_VENDOR_APPLE,
    FB_GPU_VENDOR_UNKNOWN
} fb_gpu_vendor_t;

/**
 * @brief Get GPU vendor
 * 
 * @return GPU vendor enumeration value
 */
fb_gpu_vendor_t fb_hw_get_gpu_vendor(void);

/**
 * @brief Get GPU compute capability (NVIDIA) or equivalent
 * 
 * For NVIDIA: Returns compute capability as major*10 + minor (e.g., 86 for 8.6)
 * For AMD: Returns gfx generation number (e.g., 1030 for gfx1030)
 * For others: Returns 0
 * 
 * @return GPU compute capability or 0
 */
int fb_hw_get_gpu_compute_capability(void);

/**
 * @brief Cleanup hardware detection system
 */
void fb_hw_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* FB_HARDWARE_DETECT_H */
