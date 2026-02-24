#include "benchmark_system.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

// Benchmark cache structure (internal)
typedef struct {
    fb_benchmark_cache_header_t header;
    fb_benchmark_stats_t* data;  // 4D array: [ops][backends][sizes][shapes]
    size_t data_size;            // Total size in bytes
    bool modified;               // Needs saving
} fb_benchmark_cache_internal_t;

// Cache dimensions
#define MAX_OPERATIONS      1248
#define MAX_BACKENDS        64   // Maximum backend+device combos
#define SIZE_CLASSES        5
#define SHAPE_CLASSES       5

// Calculate linear index into 4D array
static inline size_t calc_index(uint32_t op_id, uint32_t backend_id,
                                fb_size_class_t size_class,
                                fb_shape_class_t shape_class) {
    return ((op_id * MAX_BACKENDS + backend_id) * SIZE_CLASSES + size_class) 
           * SHAPE_CLASSES + shape_class;
}

int fb_get_default_cache_path(char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return -1;
    }
    
#ifdef _WIN32
    // Windows: %LOCALAPPDATA%\faster-blaster\benchmarks.bin
    char appdata[MAX_PATH];
    if (SHGetFolderPathA(NULL, CSIDL_LOCAL_APPDATA, NULL, 0, appdata) != S_OK) {
        return -1;
    }
    snprintf(buffer, buffer_size, "%s\\faster-blaster\\benchmarks.bin", appdata);
#else
    // Linux/macOS: ~/.cache/faster-blaster/benchmarks.bin
    const char* home = getenv("HOME");
    if (!home) {
        return -1;
    }
    snprintf(buffer, buffer_size, "%s/.cache/faster-blaster/benchmarks.bin", home);
#endif
    
    return 0;
}

// Create directory recursively
static int create_directory_recursive(const char* path) {
#ifdef _WIN32
    char dir[MAX_PATH];
    strncpy(dir, path, MAX_PATH - 1);
    dir[MAX_PATH - 1] = '\0';
    
    // Find last backslash
    char* last_slash = strrchr(dir, '\\');
    if (last_slash) {
        *last_slash = '\0';
        CreateDirectoryA(dir, NULL); // Ignore errors (may already exist)
    }
#else
    char dir[1024];
    strncpy(dir, path, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';
    
    char* last_slash = strrchr(dir, '/');
    if (last_slash) {
        *last_slash = '\0';
        mkdir(dir, 0755); // Ignore errors
    }
#endif
    return 0;
}

void* fb_benchmark_cache_load(const char* cache_path) {
    char default_path[512];
    if (!cache_path) {
        if (fb_get_default_cache_path(default_path, sizeof(default_path)) != 0) {
            return NULL;
        }
        cache_path = default_path;
    }
    
    FILE* fp = fopen(cache_path, "rb");
    if (!fp) {
        return NULL; // File doesn't exist (not an error)
    }
    
    // Allocate cache structure
    fb_benchmark_cache_internal_t* cache = 
        (fb_benchmark_cache_internal_t*)calloc(1, sizeof(fb_benchmark_cache_internal_t));
    if (!cache) {
        fclose(fp);
        return NULL;
    }
    
    // Read header
    if (fread(&cache->header, sizeof(fb_benchmark_cache_header_t), 1, fp) != 1) {
        free(cache);
        fclose(fp);
        return NULL;
    }
    
    // Validate magic
    if (memcmp(cache->header.magic, "FBBENCH\0", 8) != 0) {
        free(cache);
        fclose(fp);
        return NULL;
    }
    
    // Validate hardware fingerprint
    fb_hardware_fingerprint_t current_hw;
    if (fb_generate_hardware_fingerprint(&current_hw) != 0 ||
        !fb_fingerprints_match(&current_hw, &cache->header.hw_print)) {
        free(cache);
        fclose(fp);
        return NULL; // Hardware mismatch
    }
    
    // Allocate data array
    cache->data_size = MAX_OPERATIONS * MAX_BACKENDS * SIZE_CLASSES * SHAPE_CLASSES 
                      * sizeof(fb_benchmark_stats_t);
    cache->data = (fb_benchmark_stats_t*)calloc(1, cache->data_size);
    if (!cache->data) {
        free(cache);
        fclose(fp);
        return NULL;
    }
    
    // Read data
    if (fread(cache->data, cache->data_size, 1, fp) != 1) {
        free(cache->data);
        free(cache);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    cache->modified = false;
    return cache;
}

int fb_benchmark_cache_save(void* cache, const char* cache_path) {
    if (!cache) {
        return -1;
    }
    
    fb_benchmark_cache_internal_t* internal = 
        (fb_benchmark_cache_internal_t*)cache;
    
    char default_path[512];
    if (!cache_path) {
        if (fb_get_default_cache_path(default_path, sizeof(default_path)) != 0) {
            return -1;
        }
        cache_path = default_path;
    }
    
    // Create directory if needed
    create_directory_recursive(cache_path);
    
    FILE* fp = fopen(cache_path, "wb");
    if (!fp) {
        return -1;
    }
    
    // Update header
    memcpy(internal->header.magic, "FBBENCH\0", 8);
    internal->header.version = 1;
    internal->header.operation_count = MAX_OPERATIONS;
    internal->header.backend_device_count = MAX_BACKENDS;
    internal->header.size_class_count = SIZE_CLASSES;
    internal->header.shape_class_count = SHAPE_CLASSES;
    internal->header.timestamp_modified = (uint64_t)time(NULL);
    internal->header.compression_type = 0; // No compression yet
    internal->header.checksum = 0; // TODO: Calculate CRC32
    
    // Update hardware fingerprint
    fb_generate_hardware_fingerprint(&internal->header.hw_print);
    
    // Write header
    if (fwrite(&internal->header, sizeof(fb_benchmark_cache_header_t), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    
    // Write data
    if (fwrite(internal->data, internal->data_size, 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    internal->modified = false;
    return 0;
}

void fb_benchmark_cache_free(void* cache) {
    if (cache) {
        fb_benchmark_cache_internal_t* internal = 
            (fb_benchmark_cache_internal_t*)cache;
        free(internal->data);
        free(internal);
    }
}

bool fb_benchmark_cache_query(void* cache,
                              uint32_t operation_id,
                              uint32_t backend_device_id,
                              fb_size_class_t size_class,
                              fb_shape_class_t shape_class,
                              fb_benchmark_stats_t* stats_out) {
    if (!cache || !stats_out) {
        return false;
    }
    
    if (operation_id >= MAX_OPERATIONS || backend_device_id >= MAX_BACKENDS ||
        size_class >= SIZE_CLASSES || shape_class >= SHAPE_CLASSES) {
        return false;
    }
    
    fb_benchmark_cache_internal_t* internal = 
        (fb_benchmark_cache_internal_t*)cache;
    
    size_t idx = calc_index(operation_id, backend_device_id, size_class, shape_class);
    fb_benchmark_stats_t* stats = &internal->data[idx];
    
    // Check if valid entry
    if (!(stats->flags & FB_BENCH_FLAG_VALID)) {
        return false;
    }
    
    memcpy(stats_out, stats, sizeof(fb_benchmark_stats_t));
    return true;
}

int fb_benchmark_cache_store(void* cache,
                             uint32_t operation_id,
                             uint32_t backend_device_id,
                             fb_size_class_t size_class,
                             fb_shape_class_t shape_class,
                             const fb_benchmark_stats_t* stats) {
    if (!cache || !stats) {
        return -1;
    }
    
    if (operation_id >= MAX_OPERATIONS || backend_device_id >= MAX_BACKENDS ||
        size_class >= SIZE_CLASSES || shape_class >= SHAPE_CLASSES) {
        return -1;
    }
    
    fb_benchmark_cache_internal_t* internal = 
        (fb_benchmark_cache_internal_t*)cache;
    
    size_t idx = calc_index(operation_id, backend_device_id, size_class, shape_class);
    memcpy(&internal->data[idx], stats, sizeof(fb_benchmark_stats_t));
    
    internal->modified = true;
    return 0;
}

bool fb_benchmark_operation_usable(const fb_benchmark_stats_t* stats) {
    if (!stats) {
        return false;
    }
    
    // Must be valid and not have critical errors
    if (!(stats->flags & FB_BENCH_FLAG_VALID)) {
        return false;
    }
    
    // Check for fatal errors
    if (stats->flags & (FB_BENCH_FLAG_CRASHED | 
                       FB_BENCH_FLAG_TIMEOUT |
                       FB_BENCH_FLAG_CORRUPTED |
                       FB_BENCH_FLAG_NOT_IMPL)) {
        return false;
    }
    
    return true;
}

void fb_print_benchmark_stats(const fb_benchmark_stats_t* stats) {
    if (!stats) {
        printf("NULL stats\n");
        return;
    }
    
    printf("Benchmark Statistics:\n");
    printf("  Accuracy (mean/worst): %u / %u\n", 
           stats->accuracy_mean, stats->accuracy_worst);
    printf("  Precision (mean/worst): %u / %u\n", 
           stats->precision_mean, stats->precision_worst);
    printf("  Time (mean): %.3f ms\n", stats->time_mean_ns / 1000000.0);
    printf("  Time (stddev): %.3f ms\n", stats->time_stddev_ns / 1000000.0);
    printf("  Time (p99): %.3f ms\n", stats->time_p99_ns / 1000000.0);
    printf("  Samples: %u\n", stats->sample_count);
    printf("  Flags: 0x%02X ", stats->flags);
    
    if (stats->flags & FB_BENCH_FLAG_VALID) printf("[VALID] ");
    if (stats->flags & FB_BENCH_FLAG_CRASHED) printf("[CRASHED] ");
    if (stats->flags & FB_BENCH_FLAG_TIMEOUT) printf("[TIMEOUT] ");
    if (stats->flags & FB_BENCH_FLAG_NAN_INF) printf("[NAN/INF] ");
    if (stats->flags & FB_BENCH_FLAG_INACCURATE) printf("[INACCURATE] ");
    if (stats->flags & FB_BENCH_FLAG_SLOW) printf("[SLOW] ");
    if (stats->flags & FB_BENCH_FLAG_CORRUPTED) printf("[CORRUPTED] ");
    if (stats->flags & FB_BENCH_FLAG_NOT_IMPL) printf("[NOT_IMPL] ");
    
    printf("\n");
}
