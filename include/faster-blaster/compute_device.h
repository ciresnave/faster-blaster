/**
 * @file compute_device.h
 * @brief Unified compute device abstraction for heterogeneous CPU/GPU computing
 * 
 * This module provides a unified interface for all compute devices (CPUs and GPUs),
 * treating them as interchangeable resources for linear algebra operations.
 * 
 * Key capabilities:
 * - Device enumeration and discovery
 * - Unified device properties (cores, memory, performance)
 * - Real-time load tracking
 * - Performance cost models per device
 * - Power and thermal state monitoring
 */

#ifndef FASTER_BLASTER_COMPUTE_DEVICE_H
#define FASTER_BLASTER_COMPUTE_DEVICE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Device Types and Capabilities
 * ========================================================================== */

/**
 * @brief Compute device type
 */
typedef enum {
    FB_DEVICE_TYPE_CPU = 0,     /**< CPU device */
    FB_DEVICE_TYPE_GPU = 1,     /**< GPU device (CUDA, ROCm, oneAPI, etc.) */
    FB_DEVICE_TYPE_ACCELERATOR  /**< Other accelerators (TPU, IPU, etc.) */
} fb_device_type_t;

/**
 * @brief GPU vendor type
 */
typedef enum {
    FB_GPU_VENDOR_UNKNOWN = 0,
    FB_GPU_VENDOR_NVIDIA,       /**< NVIDIA (CUDA) */
    FB_GPU_VENDOR_AMD,          /**< AMD (ROCm/HIP) */
    FB_GPU_VENDOR_INTEL,        /**< Intel (oneAPI) */
    FB_GPU_VENDOR_APPLE,        /**< Apple (Metal) */
    FB_GPU_VENDOR_ARM           /**< ARM Mali */
} fb_gpu_vendor_t;

/**
 * @brief CPU vendor type
 */
typedef enum {
    FB_CPU_VENDOR_UNKNOWN = 0,
    FB_CPU_VENDOR_INTEL,        /**< Intel x86_64 */
    FB_CPU_VENDOR_AMD,          /**< AMD x86_64 */
    FB_CPU_VENDOR_ARM,          /**< ARM (Apple Silicon, AWS Graviton, etc.) */
    FB_CPU_VENDOR_IBM,          /**< IBM Power */
    FB_CPU_VENDOR_RISCV         /**< RISC-V */
} fb_cpu_vendor_t;

/**
 * @brief Device capability flags
 */
typedef enum {
    FB_DEVICE_CAP_NONE             = 0,
    FB_DEVICE_CAP_FP64             = 1 << 0,  /**< Double precision support */
    FB_DEVICE_CAP_FP16             = 1 << 1,  /**< Half precision support */
    FB_DEVICE_CAP_BF16             = 1 << 2,  /**< BFloat16 support */
    FB_DEVICE_CAP_TF32             = 1 << 3,  /**< TensorFloat32 support */
    FB_DEVICE_CAP_INT8             = 1 << 4,  /**< INT8 operations */
    FB_DEVICE_CAP_TENSOR_CORES     = 1 << 5,  /**< Tensor cores (NVIDIA) */
    FB_DEVICE_CAP_MATRIX_CORES     = 1 << 6,  /**< Matrix cores (AMD) */
    FB_DEVICE_CAP_AMX              = 1 << 7,  /**< Advanced Matrix Extensions (Intel) */
    FB_DEVICE_CAP_SVE              = 1 << 8,  /**< Scalable Vector Extension (ARM) */
    FB_DEVICE_CAP_UNIFIED_MEMORY   = 1 << 9,  /**< Unified CPU/GPU memory */
    FB_DEVICE_CAP_ASYNC_COPY       = 1 << 10, /**< Async CPU↔GPU transfers */
    FB_DEVICE_CAP_PEER_ACCESS      = 1 << 11, /**< GPU-to-GPU direct access */
    FB_DEVICE_CAP_ECC_MEMORY       = 1 << 12  /**< Error-correcting memory */
} fb_device_capability_t;

/**
 * @brief Device power state
 */
typedef enum {
    FB_POWER_STATE_UNKNOWN = 0,
    FB_POWER_STATE_PERFORMANCE,     /**< Maximum performance (AC power) */
    FB_POWER_STATE_BALANCED,        /**< Balanced (may throttle) */
    FB_POWER_STATE_POWER_SAVER,     /**< Low power (battery mode) */
    FB_POWER_STATE_THERMAL_LIMIT    /**< Thermal throttling active */
} fb_power_state_t;

/* ============================================================================
 * Device Properties
 * ========================================================================== */

/**
 * @brief CPU-specific properties
 */
typedef struct {
    fb_cpu_vendor_t vendor;         /**< CPU vendor */
    int num_physical_cores;         /**< Physical core count */
    int num_logical_cores;          /**< Logical core count (with SMT) */
    int cache_line_size;            /**< Cache line size (bytes) */
    size_t l1_cache_size;           /**< L1 cache per core (bytes) */
    size_t l2_cache_size;           /**< L2 cache per core (bytes) */
    size_t l3_cache_size;           /**< L3 cache total (bytes) */
    int simd_width;                 /**< SIMD register width (bits: 128/256/512) */
    bool has_avx;                   /**< AVX support */
    bool has_avx2;                  /**< AVX2 support */
    bool has_avx512;                /**< AVX-512 support */
    bool has_fma;                   /**< FMA support */
    bool has_neon;                  /**< ARM NEON support */
} fb_cpu_properties_t;

/**
 * @brief GPU-specific properties
 */
typedef struct {
    fb_gpu_vendor_t vendor;         /**< GPU vendor */
    int compute_capability_major;   /**< Compute capability (major version) */
    int compute_capability_minor;   /**< Compute capability (minor version) */
    int multiprocessor_count;       /**< Number of SMs/CUs */
    int max_threads_per_block;      /**< Max threads per block */
    int max_threads_per_multiprocessor; /**< Max threads per SM/CU */
    size_t total_global_memory;     /**< Total GPU memory (bytes) */
    size_t shared_memory_per_block; /**< Shared memory per block (bytes) */
    int warp_size;                  /**< Warp/wavefront size (32/64) */
    int max_grid_size[3];           /**< Max grid dimensions */
    int max_block_size[3];          /**< Max block dimensions */
    int pci_bus_id;                 /**< PCI bus ID */
    int pci_device_id;              /**< PCI device ID */
} fb_gpu_properties_t;

/**
 * @brief Unified compute device properties
 */
typedef struct {
    char name[256];                 /**< Device name (e.g., "NVIDIA RTX 4090") */
    fb_device_type_t type;          /**< Device type (CPU/GPU) */
    int device_id;                  /**< Unique device ID in system */
    
    /* Performance metrics */
    double peak_gflops_fp32;        /**< Peak GFLOPS (single precision) */
    double peak_gflops_fp64;        /**< Peak GFLOPS (double precision) */
    double memory_bandwidth_gbps;   /**< Memory bandwidth (GB/s) */
    size_t total_memory;            /**< Total memory (bytes) */
    
    /* Capabilities */
    uint32_t capabilities;          /**< Capability flags (fb_device_capability_t) */
    
    /* Type-specific properties */
    union {
        fb_cpu_properties_t cpu;    /**< CPU-specific properties */
        fb_gpu_properties_t gpu;    /**< GPU-specific properties */
    };
} fb_device_properties_t;

/* ============================================================================
 * Device State and Load Tracking
 * ========================================================================== */

/**
 * @brief Real-time device load metrics
 */
typedef struct {
    double utilization_percent;     /**< Current utilization (0-100%) */
    size_t memory_used;             /**< Memory currently used (bytes) */
    size_t memory_available;        /**< Memory available (bytes) */
    int active_operations;          /**< Number of operations in flight */
    double temperature_celsius;     /**< Device temperature (if available) */
    fb_power_state_t power_state;   /**< Current power state */
    uint64_t timestamp_us;          /**< Timestamp of measurement (microseconds) */
} fb_device_load_t;

/**
 * @brief Compute device handle
 */
typedef struct fb_compute_device fb_compute_device_t;

struct fb_compute_device {
    /* Immutable properties */
    fb_device_properties_t properties;
    
    /* Runtime state */
    fb_device_load_t load;
    bool is_available;              /**< Device currently available */
    bool is_initialized;            /**< Device successfully initialized */
    
    /* Backend-specific handle */
    void* backend_handle;           /**< Backend context (CPU/GPU specific) */
    
    /* Cost model for operation estimation */
    void* cost_model;               /**< Performance cost model (opaque) */
    
    /* Calibration data */
    bool is_calibrated;             /**< Has been benchmarked */
    uint64_t calibration_timestamp; /**< When last calibrated */
};

/* ============================================================================
 * Device Management Functions
 * ========================================================================== */

/**
 * @brief Initialize device subsystem and discover all devices
 * @return Number of devices discovered, or -1 on error
 */
int fb_device_init(void);

/**
 * @brief Shutdown device subsystem and release resources
 */
void fb_device_shutdown(void);

/**
 * @brief Get number of available compute devices
 * @return Total device count (CPU + GPU)
 */
int fb_device_get_count(void);

/**
 * @brief Get number of CPU devices
 * @return CPU device count
 */
int fb_device_get_cpu_count(void);

/**
 * @brief Get number of GPU devices
 * @return GPU device count
 */
int fb_device_get_gpu_count(void);

/**
 * @brief Get device by ID
 * @param device_id Device ID (0 to count-1)
 * @return Device handle, or NULL if invalid
 */
fb_compute_device_t* fb_device_get(int device_id);

/**
 * @brief Get device by type and index
 * @param type Device type (CPU/GPU)
 * @param index Index within type (0 to type_count-1)
 * @return Device handle, or NULL if invalid
 */
fb_compute_device_t* fb_device_get_by_type(fb_device_type_t type, int index);

/**
 * @brief Find best device for a specific operation type
 * @param op_type Operation type hint (GEMM, GETRF, etc.)
 * @param data_size Data size in bytes
 * @param prefer_type Preferred device type (or -1 for any)
 * @return Best device ID, or -1 if none available
 */
int fb_device_find_best(const char* op_type, size_t data_size, fb_device_type_t prefer_type);

/**
 * @brief Update device load metrics
 * @param device Device to update
 * @return 0 on success, non-zero on error
 */
int fb_device_update_load(fb_compute_device_t* device);

/**
 * @brief Check if device has specific capability
 * @param device Device to check
 * @param capability Capability flag to test
 * @return true if device has capability
 */
bool fb_device_has_capability(const fb_compute_device_t* device, fb_device_capability_t capability);

/**
 * @brief Estimate operation cost on device
 * @param device Device to estimate for
 * @param op_name Operation name (e.g., "sgemm")
 * @param m Matrix dimension M
 * @param n Matrix dimension N
 * @param k Matrix dimension K
 * @return Estimated time in microseconds
 */
double fb_device_estimate_cost(const fb_compute_device_t* device,
                               const char* op_name,
                               int m, int n, int k);

/**
 * @brief Reserve device for operation (increase load counter)
 * @param device Device to reserve
 */
void fb_device_reserve(fb_compute_device_t* device);

/**
 * @brief Release device after operation (decrease load counter)
 * @param device Device to release
 */
void fb_device_release(fb_compute_device_t* device);

/* ============================================================================
 * Device Enumeration and Query
 * ========================================================================== */

/**
 * @brief Print device information
 * @param device Device to print
 */
void fb_device_print_info(const fb_compute_device_t* device);

/**
 * @brief Print all devices
 */
void fb_device_print_all(void);

/**
 * @brief Get device name string
 * @param device Device to query
 * @return Device name
 */
const char* fb_device_get_name(const fb_compute_device_t* device);

/**
 * @brief Get device type string
 * @param device Device to query
 * @return Device type string ("CPU", "GPU", etc.)
 */
const char* fb_device_get_type_string(const fb_compute_device_t* device);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_COMPUTE_DEVICE_H */
