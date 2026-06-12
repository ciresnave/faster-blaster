#include "benchmark_system.h"
#include <stdio.h>
#include <stdlib.h>

// Constants
#define SIZE_CLASSES 5
#define SHAPE_CLASSES 5

// Test the benchmark infrastructure
int main(int argc, char** argv) {
    printf("=== Faster-Blaster Benchmark System Test ===\n\n");
    
    // Test 1: Size classification
    printf("Test 1: Size Classification\n");
    size_t test_sizes[][3] = {
        {16, 16, 0},      // TINY
        {128, 128, 0},    // SMALL
        {1024, 1024, 0},  // MEDIUM
        {5000, 5000, 0},  // LARGE
        {16384, 16384, 0} // HUGE
    };
    
    const char* size_names[] = {"TINY", "SMALL", "MEDIUM", "LARGE", "HUGE"};
    
    for (size_t i = 0; i < 5; i++) {
        fb_size_class_t cls = fb_classify_size(test_sizes[i][0], 
                                              test_sizes[i][1], 
                                              test_sizes[i][2]);
        printf("  %zu x %zu -> %s ", test_sizes[i][0], test_sizes[i][1], size_names[cls]);
        printf("%s\n", (cls == (int)i) ? "✓" : "✗ FAIL");
    }
    
    // Test 2: Shape classification
    printf("\nTest 2: Shape Classification\n");
    size_t test_shapes[][2] = {
        {1024, 1024},  // SQUARE
        {5120, 1024},  // TALL_SKINNY
        {1024, 5120},  // SHORT_WIDE
        {3072, 1024},  // SKINNY_MIDDLE
        {1024, 3072}   // FAT_MIDDLE
    };
    
    const char* shape_names[] = {"SQUARE", "TALL_SKINNY", "SHORT_WIDE", 
                                "SKINNY_MIDDLE", "FAT_MIDDLE"};
    
    for (size_t i = 0; i < 5; i++) {
        fb_shape_class_t cls = fb_classify_shape(test_shapes[i][0], 
                                                 test_shapes[i][1]);
        printf("  %zu x %zu -> %s ", test_shapes[i][0], test_shapes[i][1], shape_names[cls]);
        printf("%s\n", (cls == (int)i) ? "✓" : "✗ FAIL");
    }
    
    // Test 3: Hardware fingerprint
    printf("\nTest 3: Hardware Fingerprint\n");
    fb_hardware_fingerprint_t fp;
    if (fb_generate_hardware_fingerprint(&fp) == 0) {
        printf("  CPU Vendor Hash: 0x%08X\n", fp.cpu_vendor_hash);
        printf("  CPU Family/Model: 0x%X / 0x%X\n", fp.cpu_family, fp.cpu_model);
        printf("  Memory: %u MB\n", fp.memory_size_mb);
        printf("  Schema Version: %u\n", fp.schema_version);
        printf("  ✓ Generated successfully\n");
    } else {
        printf("  ✗ FAIL: Could not generate fingerprint\n");
    }
    
    // Test 4: Cache operations
    printf("\nTest 4: Benchmark Cache\n");
    
    // Create cache path
    char cache_path[512];
    if (fb_get_default_cache_path(cache_path, sizeof(cache_path)) == 0) {
        printf("  Default cache path: %s\n", cache_path);
    }
    
    // Try to load cache (may not exist)
    void* cache = fb_benchmark_cache_load(NULL);
    if (cache) {
        printf("  ✓ Loaded existing cache\n");
    } else {
        printf("  Cache doesn't exist (expected for first run)\n");
        // Create new empty cache
        cache = fb_benchmark_cache_create();
        if (cache) {
            printf("  ✓ Created new cache structure\n");
        }
    }
    
    if (cache) {
        // Test storing and querying
        fb_benchmark_stats_t test_stats = {
            .accuracy_mean = 255,
            .accuracy_worst = 255,
            .precision_mean = 255,
            .precision_worst = 255,
            .time_mean_ns = 1000000,
            .time_stddev_ns = 50000,
            .time_p99_ns = 1200000,
            .sample_count = 10,
            .flags = FB_BENCH_FLAG_VALID
        };
        
        // Store test data
        if (fb_benchmark_cache_store(cache, 0, 0, FB_SIZE_MEDIUM, 
                                    FB_SHAPE_SQUARE, &test_stats) == 0) {
            printf("  ✓ Stored test benchmark data\n");
        }
        
        // Query back
        fb_benchmark_stats_t retrieved;
        if (fb_benchmark_cache_query(cache, 0, 0, FB_SIZE_MEDIUM, 
                                    FB_SHAPE_SQUARE, &retrieved)) {
            printf("  ✓ Retrieved benchmark data\n");
            printf("    Time: %.3f ms (mean)\n", retrieved.time_mean_ns / 1000000.0);
        } else {
            printf("  ✗ FAIL: Could not retrieve data\n");
        }
        
        fb_benchmark_cache_free(cache);
    }
    
    // Test 5: Representative sizes/shapes
    printf("\nTest 5: Representative Dimensions\n");
    for (fb_size_class_t sc = 0; sc < SIZE_CLASSES; sc++) {
        size_t m, n, k;
        fb_get_representative_size(sc, &m, &n, &k);
        printf("  %s: %zu x %zu x %zu\n", size_names[sc], m, n, k);
    }
    
    printf("\nTest 6: Shape Representatives (base=1024)\n");
    for (fb_shape_class_t sh = 0; sh < SHAPE_CLASSES; sh++) {
        size_t m, n;
        fb_get_representative_shape(sh, 1024, &m, &n);
        printf("  %s: %zu x %zu (ratio: %.2f)\n", 
               shape_names[sh], m, n, (double)m / (double)n);
    }
    
    // Test 7: Operation usability check
    printf("\nTest 7: Operation Usability Check\n");
    fb_benchmark_stats_t good_op = {.flags = FB_BENCH_FLAG_VALID};
    fb_benchmark_stats_t crashed_op = {.flags = FB_BENCH_FLAG_CRASHED};
    fb_benchmark_stats_t timeout_op = {.flags = FB_BENCH_FLAG_TIMEOUT};
    
    printf("  Valid operation: %s\n", 
           fb_benchmark_operation_usable(&good_op) ? "✓ Usable" : "✗ Blocked");
    printf("  Crashed operation: %s\n", 
           fb_benchmark_operation_usable(&crashed_op) ? "✗ FAIL" : "✓ Blocked");
    printf("  Timeout operation: %s\n", 
           fb_benchmark_operation_usable(&timeout_op) ? "✗ FAIL" : "✓ Blocked");
    
    printf("\n=== All Infrastructure Tests Complete ===\n");
    printf("Note: Safe execution tests require backend integration\n");
    
    return 0;
}
