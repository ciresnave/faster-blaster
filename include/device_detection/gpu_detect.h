/**
 * @file gpu_detect.h
 * @brief GPU Detection and Capability Discovery
 * 
 * Detects GPUs from NVIDIA (CUDA), AMD (ROCm), Intel (oneAPI), and Apple (Metal)
 * with comprehensive capability and performance information.
 */

#ifndef FASTER_BLASTER_GPU_DETECT_H
#define FASTER_BLASTER_GPU_DETECT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU Vendor
typedef enum {
    GPU_VENDOR_UNKNOWN = 0,
    GPU_VENDOR_NVIDIA,
    GPU_VENDOR_AMD,
    GPU_VENDOR_INTEL,
    GPU_VENDOR_APPLE,
    GPU_VENDOR_ARM,      // Mali, Immortalis
    GPU_VENDOR_QUALCOMM, // Adreno
    GPU_VENDOR_IMG,      // PowerVR
    GPU_VENDOR_OTHER
} fb_gpu_vendor_t;

// GPU Architecture Type
typedef enum {
    GPU_ARCH_UNKNOWN = 0,
    // NVIDIA
    GPU_ARCH_NVIDIA_FERMI,
    GPU_ARCH_NVIDIA_KEPLER,
    GPU_ARCH_NVIDIA_MAXWELL,
    GPU_ARCH_NVIDIA_PASCAL,
    GPU_ARCH_NVIDIA_VOLTA,
    GPU_ARCH_NVIDIA_TURING,
    GPU_ARCH_NVIDIA_AMPERE,
    GPU_ARCH_NVIDIA_ADA_LOVELACE,
    GPU_ARCH_NVIDIA_HOPPER,
    GPU_ARCH_NVIDIA_BLACKWELL,
    // AMD
    GPU_ARCH_AMD_GCN,
    GPU_ARCH_AMD_RDNA,
    GPU_ARCH_AMD_RDNA2,
    GPU_ARCH_AMD_RDNA3,
    GPU_ARCH_AMD_CDNA,
    GPU_ARCH_AMD_CDNA2,
    GPU_ARCH_AMD_CDNA3,
    // Intel
    GPU_ARCH_INTEL_GEN9,
    GPU_ARCH_INTEL_GEN11,
    GPU_ARCH_INTEL_GEN12,
    GPU_ARCH_INTEL_XE,
    GPU_ARCH_INTEL_XE_HPG,  // Arc
    GPU_ARCH_INTEL_XE_HPC,  // Ponte Vecchio
    // Apple
    GPU_ARCH_APPLE_M1,
    GPU_ARCH_APPLE_M2,
    GPU_ARCH_APPLE_M3,
    GPU_ARCH_APPLE_M4,
    GPU_ARCH_OTHER
} fb_gpu_arch_t;

// Memory Types
typedef enum {
    GPU_MEMORY_GDDR5 = 0,
    GPU_MEMORY_GDDR6,
    GPU_MEMORY_GDDR6X,
    GPU_MEMORY_HBM,
    GPU_MEMORY_HBM2,
    GPU_MEMORY_HBM2E,
    GPU_MEMORY_HBM3,
    GPU_MEMORY_UNIFIED,  // Apple UMA
    GPU_MEMORY_UNKNOWN
} fb_gpu_memory_type_t;

// Compute Capabilities
typedef struct {
    // CUDA compute capability (NVIDIA)
    int cuda_major;
    int cuda_minor;
    
    // HIP architecture (AMD)
    char hip_arch[32]; // e.g., "gfx90a"
    
    // OpenCL version
    int opencl_major;
    int opencl_minor;
    
    // Metal feature set (Apple)
    char metal_family[32]; // e.g., "Apple8"
    
    // Supported precision
    bool fp64;           // Double precision
    bool fp32;           // Single precision
    bool fp16;           // Half precision
    bool bf16;           // BrainFloat16
    bool int8;           // INT8
    bool int4;           // INT4
    
    // Matrix/Tensor cores
    bool has_tensor_cores;     // NVIDIA Tensor Cores
    bool has_matrix_cores;     // AMD Matrix Cores
    bool has_xmx;              // Intel XMX
    
    // Memory features
    bool has_unified_memory;   // CUDA Unified Memory
    bool has_peer_access;      // Multi-GPU peer access
    bool has_ecc;              // Error-correcting memory
    
} fb_gpu_compute_caps_t;

// GPU Information Structure
typedef struct {
    // Identification
    fb_gpu_vendor_t vendor;
    fb_gpu_arch_t architecture;
    int device_id;                  // Device index (0, 1, 2...)
    char name[256];                 // e.g., "NVIDIA GeForce RTX 4090"
    char pci_bus_id[32];            // PCIe bus ID
    
    // Compute
    fb_gpu_compute_caps_t compute;
    uint32_t compute_units;         // NVIDIA: SMs, AMD: CUs, Intel: EUs
    uint32_t cores_per_cu;          // Cores per compute unit
    uint32_t total_cores;           // Total shader cores
    
    // Clocks
    uint32_t base_clock_mhz;
    uint32_t boost_clock_mhz;
    uint32_t memory_clock_mhz;
    
    // Memory
    uint64_t total_memory_bytes;
    uint64_t free_memory_bytes;
    fb_gpu_memory_type_t memory_type;
    uint32_t memory_bus_width;      // bits
    uint32_t l2_cache_bytes;
    
    // Bandwidth
    double memory_bandwidth_gbps;
    double pcie_bandwidth_gbps;
    
    // Performance
    double fp32_tflops;
    double fp64_tflops;
    double fp16_tflops;
    double tensor_tflops;           // With tensor cores
    
    // Power
    uint32_t tdp_watts;
    uint32_t current_power_watts;
    uint32_t power_limit_watts;
    
    // Thermal
    uint32_t current_temp_c;
    uint32_t max_temp_c;
    
    // Utilization
    uint32_t gpu_utilization_percent;
    uint32_t memory_utilization_percent;
    
    // Runtime availability
    bool cuda_available;
    bool hip_available;
    bool opencl_available;
    bool metal_available;
    bool sycl_available;
    
} fb_gpu_info_t;

/**
 * @brief Detect all GPUs in the system
 * 
 * @param gpus Array to fill with GPU information
 * @param max_gpus Maximum number of GPUs to detect
 * @param num_gpus Output: Actual number of GPUs detected
 * @return 0 on success, negative error code on failure
 */
int fb_detect_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus);

/**
 * @brief Detect NVIDIA CUDA GPUs
 * 
 * @param gpus Array to fill with GPU information
 * @param max_gpus Maximum number of GPUs
 * @param num_gpus Output: Number detected
 * @return 0 on success
 */
int fb_detect_cuda_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus);

/**
 * @brief Detect AMD ROCm/HIP GPUs
 * 
 * @param gpus Array to fill with GPU information
 * @param max_gpus Maximum number of GPUs
 * @param num_gpus Output: Number detected
 * @return 0 on success
 */
int fb_detect_rocm_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus);

/**
 * @brief Detect Intel oneAPI GPUs
 * 
 * @param gpus Array to fill with GPU information
 * @param max_gpus Maximum number of GPUs
 * @param num_gpus Output: Number detected
 * @return 0 on success
 */
int fb_detect_intel_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus);

/**
 * @brief Detect Apple Metal GPUs
 * 
 * @param gpus Array to fill with GPU information
 * @param max_gpus Maximum number of GPUs
 * @param num_gpus Output: Number detected
 * @return 0 on success
 */
int fb_detect_metal_gpus(fb_gpu_info_t* gpus, uint32_t max_gpus, uint32_t* num_gpus);

/**
 * @brief Get current GPU utilization
 * 
 * @param gpu_info GPU to query
 * @return 0 on success
 */
int fb_update_gpu_utilization(fb_gpu_info_t* gpu_info);

/**
 * @brief Get current GPU temperature
 * 
 * @param gpu_info GPU to query
 * @return 0 on success
 */
int fb_update_gpu_temperature(fb_gpu_info_t* gpu_info);

/**
 * @brief Get current GPU power consumption
 * 
 * @param gpu_info GPU to query
 * @return 0 on success
 */
int fb_update_gpu_power(fb_gpu_info_t* gpu_info);

/**
 * @brief Get current GPU memory usage
 * 
 * @param gpu_info GPU to query
 * @return 0 on success
 */
int fb_update_gpu_memory(fb_gpu_info_t* gpu_info);

/**
 * @brief Estimate GPU TFLOPS based on specifications
 * 
 * @param gpu_info GPU information
 */
void fb_estimate_gpu_tflops(fb_gpu_info_t* gpu_info);

/**
 * @brief Print GPU information (for debugging)
 * 
 * @param gpu_info GPU information to print
 */
void fb_print_gpu_info(const fb_gpu_info_t* gpu_info);

/**
 * @brief Check if GPU supports specific compute capability
 * 
 * @param gpu_info GPU information
 * @param major Major version
 * @param minor Minor version
 * @return true if supported
 */
bool fb_gpu_supports_compute_capability(const fb_gpu_info_t* gpu_info, 
                                        int major, int minor);

#ifdef __cplusplus
}
#endif

#endif // FASTER_BLASTER_GPU_DETECT_H
