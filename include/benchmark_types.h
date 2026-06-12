#ifndef FASTER_BLASTER_BENCHMARK_TYPES_H
#define FASTER_BLASTER_BENCHMARK_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Size classes for benchmark stratification
 */
typedef enum {
    FB_SIZE_TINY = 0,    // ≤32 elements or 32×32 matrices
    FB_SIZE_SMALL,       // 64×64 to 256×256
    FB_SIZE_MEDIUM,      // 512×512 to 2048×2048
    FB_SIZE_LARGE,       // 4096×4096 to 8192×8192
    FB_SIZE_HUGE,        // >8192×8192
    FB_SIZE_CLASS_COUNT  // Total: 5 classes
} fb_size_class_t;

/**
 * Shape classes for matrix operations
 */
typedef enum {
    FB_SHAPE_SQUARE = 0,        // m ≈ n (within 10%)
    FB_SHAPE_TALL_SKINNY,       // m >> n (m > 4n)
    FB_SHAPE_SHORT_WIDE,        // n >> m (n > 4m)
    FB_SHAPE_SKINNY_MIDDLE,     // m > n but not extreme (2n < m ≤ 4n)
    FB_SHAPE_FAT_MIDDLE,        // n > m but not extreme (2m < n ≤ 4m)
    FB_SHAPE_CLASS_COUNT        // Total: 5 classes
} fb_shape_class_t;

/**
 * Benchmark statistics for a single operation configuration
 * Stored per: [operation_id][backend_device_combo][size_class][shape_class]
 */
typedef struct {
    // Accuracy metrics (0-255, where 255 = perfect)
    uint8_t accuracy_mean;      // Average relative error (scaled)
    uint8_t accuracy_worst;     // Worst case relative error
    uint8_t precision_mean;     // Average precision loss
    uint8_t precision_worst;    // Worst precision loss
    
    // Timing metrics (nanoseconds)
    uint32_t time_mean_ns;      // Mean execution time
    uint32_t time_stddev_ns;    // Standard deviation
    uint32_t time_p99_ns;       // 99th percentile (outliers)
    
    // Metadata
    uint16_t sample_count;      // Number of successful runs
    uint8_t flags;              // Status flags (see below)
    uint8_t reserved;           // Alignment padding
} fb_benchmark_stats_t;

// Benchmark status flags
#define FB_BENCH_FLAG_VALID        0x01  // Benchmark completed successfully
#define FB_BENCH_FLAG_CRASHED      0x02  // Operation crashed during benchmark
#define FB_BENCH_FLAG_TIMEOUT      0x04  // Operation timed out
#define FB_BENCH_FLAG_NAN_INF      0x08  // Output contained NaN/Inf
#define FB_BENCH_FLAG_INACCURATE   0x10  // Failed accuracy threshold
#define FB_BENCH_FLAG_SLOW         0x20  // Slower than reference by large margin
#define FB_BENCH_FLAG_CORRUPTED    0x40  // Memory corruption detected
#define FB_BENCH_FLAG_NOT_IMPL     0x80  // Operation not implemented (vtable NULL)

/**
 * Hardware fingerprint for cache validation
 * 32-byte structure containing hardware identifiers
 */
typedef struct {
    uint32_t cpu_vendor_hash;      // Hash of CPU vendor + model string
    uint16_t cpu_family;            // CPU family ID
    uint16_t cpu_model;             // CPU model ID
    uint32_t gpu_device_ids[4];    // Up to 4 GPU device IDs (0 if absent)
    uint32_t memory_size_mb;       // System RAM in MB
    uint32_t backend_version_hash; // Hash of all backend version strings
    uint32_t schema_version;       // Benchmark data format version
    uint32_t reserved;             // Future use
} fb_hardware_fingerprint_t;

/**
 * Benchmark cache file header
 */
typedef struct {
    char magic[8];                      // "FBBENCH\0"
    uint32_t version;                   // Format version (currently 1)
    uint32_t operation_count;           // Total operations (should be 1248)
    uint32_t backend_device_count;      // Number of backend+device combos
    uint32_t size_class_count;          // Number of size classes (5)
    uint32_t shape_class_count;         // Number of shape classes (5)
    fb_hardware_fingerprint_t hw_print; // Hardware fingerprint
    uint64_t timestamp_created;         // Unix timestamp
    uint64_t timestamp_modified;        // Last modification timestamp
    uint32_t compression_type;          // 0=none, 1=zstd
    uint32_t checksum;                  // CRC32 of data section
} fb_benchmark_cache_header_t;

/**
 * Benchmark configuration for a single test run
 */
typedef struct {
    size_t m;                    // Matrix dimension M
    size_t n;                    // Matrix dimension N
    size_t k;                    // Matrix dimension K (for GEMM-like ops)
    fb_size_class_t size_class;  // Classified size
    fb_shape_class_t shape_class; // Classified shape
    uint32_t warmup_iters;       // Warmup iterations (default: 3)
    uint32_t sample_iters;       // Sample iterations (default: 10)
    uint32_t timeout_ms;         // Timeout per iteration (default: 5000)
    bool validate_output;        // Check output against reference
    bool detect_corruption;      // Enable memory corruption detection
} fb_benchmark_config_t;

/**
 * Result from a single benchmark run
 */
typedef struct {
    fb_benchmark_stats_t stats;  // Statistical results
    bool success;                // Overall success flag
    const char* error_message;   // Error description if failed (NULL if success)
} fb_benchmark_result_t;

#ifdef __cplusplus
}
#endif

#endif // FASTER_BLASTER_BENCHMARK_TYPES_H
