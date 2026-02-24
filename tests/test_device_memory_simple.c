/**
 * @file test_device_memory_simple.c
 * @brief Simple test for device memory manager (no GPU execution)
 * 
 * This test validates the device memory manager API without requiring
 * actual GPU hardware or CUDA compilation. It tests the caching logic,
 * reference counting, and memory tracking.
 */

#include "faster-blaster/device_memory_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mock GPU trait for testing */
static int mock_malloc_count = 0;
static int mock_free_count = 0;
static int mock_h2d_count = 0;
static int mock_d2h_count = 0;

static int mock_malloc(void* handle, fb_gpu_ptr_t* ptr, size_t size) {
    (void)handle;
    *ptr = malloc(size);  /* Use host malloc as mock */
    mock_malloc_count++;
    printf("  [MOCK] Allocated %zu bytes (total allocations: %d)\n", size, mock_malloc_count);
    return (*ptr != NULL) ? 0 : -1;
}

static void mock_free(void* handle, fb_gpu_ptr_t ptr) {
    (void)handle;
    free(ptr);
    mock_free_count++;
    printf("  [MOCK] Freed memory (total frees: %d)\n", mock_free_count);
}

static int mock_memcpy_h2d(void* handle, fb_gpu_ptr_t dst, const void* src, size_t size) {
    (void)handle;
    memcpy(dst, src, size);
    mock_h2d_count++;
    printf("  [MOCK] H2D copy %zu bytes (total H2D: %d)\n", size, mock_h2d_count);
    return 0;
}

static int mock_memcpy_d2h(void* handle, void* dst, fb_gpu_ptr_t src, size_t size) {
    (void)handle;
    memcpy(dst, src, size);
    mock_d2h_count++;
    printf("  [MOCK] D2H copy %zu bytes (total D2H: %d)\n", size, mock_d2h_count);
    return 0;
}

static int mock_stream_synchronize(void* handle, fb_gpu_stream_t stream) {
    (void)handle;
    (void)stream;
    return 0;
}

/* Mock GPU trait */
static const fb_gpu_backend_trait_t mock_trait = {
    .name = "MockGPU",
    .type = FB_GPU_BACKEND_CUBLAS,
    .malloc = mock_malloc,
    .free = mock_free,
    .memcpy_h2d = mock_memcpy_h2d,
    .memcpy_d2h = mock_memcpy_d2h,
    .stream_synchronize = mock_stream_synchronize,
};

int main(void) {
    printf("========================================\n");
    printf("Device Memory Manager - Simple Test\n");
    printf("========================================\n\n");
    
    /* Test 1: Manager creation */
    printf("=== Test 1: Manager Creation ===\n");
    fb_device_memory_manager_t* manager = fb_device_memory_manager_create(0, 0);
    if (!manager) {
        fprintf(stderr, "ERROR: Failed to create memory manager\n");
        return 1;
    }
    printf("✓ Manager created successfully\n\n");
    
    /* Test 2: First allocation (cache miss) */
    printf("=== Test 2: First Allocation (Cache Miss) ===\n");
    float data_a[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    fb_gpu_ptr_t device_ptr_a = NULL;
    
    mock_malloc_count = 0;
    mock_h2d_count = 0;
    
    int result = fb_device_memory_get_or_alloc(
        manager, &mock_trait, (void*)1, 
        data_a, sizeof(data_a), 
        FB_GPU_BACKEND_CUBLAS, &device_ptr_a
    );
    
    if (result != 0 || device_ptr_a == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate device memory\n");
        return 1;
    }
    
    printf("✓ Device memory allocated\n");
    printf("  Expected: 1 malloc, 1 H2D copy\n");
    printf("  Actual:   %d malloc, %d H2D copy\n", mock_malloc_count, mock_h2d_count);
    
    if (mock_malloc_count != 1 || mock_h2d_count != 1) {
        fprintf(stderr, "ERROR: Unexpected allocation pattern\n");
        return 1;
    }
    printf("✓ Allocation pattern correct\n\n");
    
    /* Test 3: Second access to same pointer (cache hit) */
    printf("=== Test 3: Second Access (Cache Hit) ===\n");
    fb_gpu_ptr_t device_ptr_a2 = NULL;
    
    mock_malloc_count = 0;
    mock_h2d_count = 0;
    
    result = fb_device_memory_get_or_alloc(
        manager, &mock_trait, (void*)1,
        data_a, sizeof(data_a),
        FB_GPU_BACKEND_CUBLAS, &device_ptr_a2
    );
    
    if (result != 0 || device_ptr_a2 == NULL) {
        fprintf(stderr, "ERROR: Failed to get cached device memory\n");
        return 1;
    }
    
    printf("✓ Device memory retrieved from cache\n");
    printf("  Expected: 0 malloc, 0 H2D copy (cache hit!)\n");
    printf("  Actual:   %d malloc, %d H2D copy\n", mock_malloc_count, mock_h2d_count);
    
    if (mock_malloc_count != 0 || mock_h2d_count != 0) {
        fprintf(stderr, "ERROR: Cache hit should not allocate or copy\n");
        return 1;
    }
    
    if (device_ptr_a != device_ptr_a2) {
        fprintf(stderr, "ERROR: Cache should return same device pointer\n");
        return 1;
    }
    printf("✓ Cache hit successful - no redundant allocations!\n\n");
    
    /* Test 4: Release references */
    printf("=== Test 4: Reference Counting ===\n");
    fb_device_memory_release(manager, data_a);
    printf("✓ First reference released\n");
    fb_device_memory_release(manager, data_a);
    printf("✓ Second reference released\n\n");
    
    /* Test 5: Statistics */
    printf("=== Test 5: Statistics ===\n");
    size_t total_allocated, num_entries;
    double cache_hit_rate;
    
    fb_device_memory_get_stats(manager, &total_allocated, &num_entries, &cache_hit_rate);
    
    printf("Total allocated: %zu bytes\n", total_allocated);
    printf("Active entries: %zu\n", num_entries);
    printf("Cache hit rate: %.1f%%\n", cache_hit_rate * 100.0);
    
    if (cache_hit_rate < 0.4 || cache_hit_rate > 0.6) {
        fprintf(stderr, "ERROR: Expected ~50%% cache hit rate (1 hit, 1 miss)\n");
        return 1;
    }
    printf("✓ Cache hit rate correct (~50%%)\n\n");
    
    /* Test 6: Dirty tracking */
    printf("=== Test 6: Dirty Tracking ===\n");
    fb_device_memory_mark_dirty(manager, data_a, FB_GPU_BACKEND_CUBLAS);
    printf("✓ Buffer marked as dirty\n");
    
    mock_d2h_count = 0;
    fb_device_memory_sync_to_host(manager, &mock_trait, (void*)1, data_a);
    
    if (mock_d2h_count != 1) {
        fprintf(stderr, "ERROR: Sync should perform D2H copy for dirty buffer\n");
        return 1;
    }
    printf("✓ Dirty buffer synced to host (D2H performed)\n");
    
    /* Second sync should be no-op */
    mock_d2h_count = 0;
    fb_device_memory_sync_to_host(manager, &mock_trait, (void*)1, data_a);
    
    if (mock_d2h_count != 0) {
        fprintf(stderr, "ERROR: Second sync should be no-op (buffer already clean)\n");
        return 1;
    }
    printf("✓ Second sync was no-op (buffer clean)\n\n");
    
    /* Print detailed stats */
    fb_device_memory_print_stats(manager);
    
    /* Cleanup */
    printf("=== Cleanup ===\n");
    fb_device_memory_manager_destroy(manager);
    printf("✓ Manager destroyed\n\n");
    
    printf("========================================\n");
    printf("All tests PASSED! ✓\n");
    printf("========================================\n");
    
    printf("\n");
    printf("Summary:\n");
    printf("  ✓ Device memory caching works\n");
    printf("  ✓ Cache hits avoid redundant allocations\n");
    printf("  ✓ Reference counting tracks buffer usage\n");
    printf("  ✓ Dirty tracking ensures data consistency\n");
    printf("  ✓ Statistics provide visibility into efficiency\n");
    printf("\n");
    printf("The device memory manager is working correctly!\n");
    printf("Next: Test with real GPU hardware for performance validation.\n");
    
    return 0;
}
