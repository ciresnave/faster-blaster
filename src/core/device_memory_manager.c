/**
 * @file device_memory_manager.c
 * @brief Device-level memory manager implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/device_memory_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

/* Maximum number of physical devices supported */
#define MAX_DEVICES 8

/* Initial capacity for entries array */
#define INITIAL_CAPACITY 64

/* Global array of device managers (one per device) */
static fb_device_memory_manager_t* g_managers[MAX_DEVICES] = {NULL};

/* Helper to get current timestamp (for LRU) */
static uint64_t get_timestamp_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000) / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
#endif
}

/* ============================================================================
 * Manager Lifecycle
 * ========================================================================== */

fb_device_memory_manager_t* fb_device_memory_manager_create(
    int device_id,
    size_t max_memory)
{
    if (device_id < 0 || device_id >= MAX_DEVICES) {
        fprintf(stderr, "Invalid device_id: %d (must be 0-%d)\n", 
                device_id, MAX_DEVICES - 1);
        return NULL;
    }
    
    fb_device_memory_manager_t* manager = 
        (fb_device_memory_manager_t*)calloc(1, sizeof(fb_device_memory_manager_t));
    
    if (!manager) {
        fprintf(stderr, "Failed to allocate device memory manager\n");
        return NULL;
    }
    
    manager->device_id = device_id;
    manager->max_memory = max_memory;
    manager->capacity = INITIAL_CAPACITY;
    
    manager->entries = (fb_device_memory_entry_t*)calloc(
        manager->capacity, sizeof(fb_device_memory_entry_t));
    
    if (!manager->entries) {
        fprintf(stderr, "Failed to allocate entries array\n");
        free(manager);
        return NULL;
    }
    
    return manager;
}

void fb_device_memory_manager_destroy(fb_device_memory_manager_t* manager) {
    if (!manager) {
        return;
    }
    
    /* Note: We can't free device memory here without backend handle and trait.
     * Caller should call fb_device_memory_clear() with sync first.
     * For now, just free host-side structures. */
    
    if (manager->entries) {
        free(manager->entries);
    }
    
    free(manager);
}

void fb_device_memory_manager_clear(
    fb_device_memory_manager_t* manager,
    int sync_dirty)
{
    if (!manager) {
        return;
    }
    
    /* TODO: If sync_dirty, need to copy dirty buffers back
     * This requires a trait/handle which we don't have here.
     * Consider adding these as parameters or storing in manager. */
    
    if (sync_dirty) {
        fprintf(stderr, "Warning: sync_dirty requested but not implemented yet\n");
    }
    
    /* Clear all entries */
    manager->num_entries = 0;
    manager->total_allocated = 0;
    memset(manager->entries, 0, manager->capacity * sizeof(fb_device_memory_entry_t));
}

/* ============================================================================
 * Internal Helpers
 * ========================================================================== */

/**
 * @brief Find entry by host pointer
 * @return Index of entry, or -1 if not found
 */
static int find_entry_by_host_ptr(
    const fb_device_memory_manager_t* manager,
    const void* host_ptr)
{
    for (size_t i = 0; i < manager->num_entries; i++) {
        if (manager->entries[i].host_ptr == host_ptr) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief Grow entries array if needed
 */
static int grow_entries_if_needed(fb_device_memory_manager_t* manager) {
    if (manager->num_entries < manager->capacity) {
        return 0; /* No growth needed */
    }
    
    size_t new_capacity = manager->capacity * 2;
    fb_device_memory_entry_t* new_entries = (fb_device_memory_entry_t*)realloc(
        manager->entries, new_capacity * sizeof(fb_device_memory_entry_t));
    
    if (!new_entries) {
        fprintf(stderr, "Failed to grow entries array\n");
        return -1;
    }
    
    /* Zero out new entries */
    memset(&new_entries[manager->capacity], 0,
           (new_capacity - manager->capacity) * sizeof(fb_device_memory_entry_t));
    
    manager->entries = new_entries;
    manager->capacity = new_capacity;
    
    return 0;
}

/* ============================================================================
 * Memory Operations
 * ========================================================================== */

int fb_device_memory_get_or_alloc(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle,
    const void* host_ptr,
    size_t size,
    fb_gpu_backend_type_t backend_type,
    fb_gpu_ptr_t* device_ptr_out)
{
    if (!manager || !trait || !backend_handle || !host_ptr || !device_ptr_out) {
        return -1;
    }
    
    /* Check if already allocated (cache lookup) */
    int idx = find_entry_by_host_ptr(manager, host_ptr);
    
    if (idx >= 0) {
        /* Cache hit! */
        fb_device_memory_entry_t* entry = &manager->entries[idx];
        
        /* Verify size matches */
        if (entry->size != size) {
            fprintf(stderr, "Warning: Host ptr %p size mismatch (cached: %zu, requested: %zu)\n",
                    host_ptr, entry->size, size);
            /* Could reallocate here, but for now treat as error */
            return -1;
        }
        
        /* Update metadata */
        entry->ref_count++;
        entry->last_access_time = get_timestamp_ms();
        
        /* Return cached device pointer */
        *device_ptr_out = entry->device_ptr;
        
        manager->stats.num_cache_hits++;
        
        return 0;
    }
    
    /* Cache miss - need to allocate and transfer */
    manager->stats.num_cache_misses++;
    
    /* Check if we need to grow entries array */
    if (grow_entries_if_needed(manager) != 0) {
        return -1;
    }
    
    /* Check memory limit */
    if (manager->max_memory > 0 && 
        manager->total_allocated + size > manager->max_memory) {
        fprintf(stderr, "Device memory limit exceeded (limit: %zu, allocated: %zu, requested: %zu)\n",
                manager->max_memory, manager->total_allocated, size);
        /* TODO: Implement LRU eviction here */
        return -1;
    }
    
    /* Allocate device memory */
    fb_gpu_ptr_t device_ptr = NULL;
    if (!trait->malloc) {
        fprintf(stderr, "Backend trait missing malloc function\n");
        return -1;
    }
    
    int result = trait->malloc(backend_handle, &device_ptr, size);
    if (result != 0 || device_ptr == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes on device %d\n", 
                size, manager->device_id);
        return -1;
    }
    
    /* Copy host → device */
    if (trait->memcpy_h2d) {
        result = trait->memcpy_h2d(backend_handle, device_ptr, host_ptr, size);
        if (result != 0) {
            fprintf(stderr, "Failed to copy %zu bytes H2D\n", size);
            /* Clean up: free device memory */
            if (trait->free) {
                trait->free(backend_handle, device_ptr);
            }
            return -1;
        }
        
        manager->stats.num_h2d_copies++;
        manager->stats.bytes_h2d += size;
    }
    
    /* Register new entry */
    fb_device_memory_entry_t* entry = &manager->entries[manager->num_entries];
    entry->host_ptr = (void*)host_ptr;
    entry->device_ptr = device_ptr;
    entry->size = size;
    entry->device_id = manager->device_id;
    entry->dirty = 0;  /* Just copied H2D, so host and device are in sync */
    entry->ref_count = 1;
    entry->last_writer_backend = backend_type;
    entry->last_access_time = get_timestamp_ms();
    
    manager->num_entries++;
    manager->total_allocated += size;
    manager->stats.num_allocations++;
    
    *device_ptr_out = device_ptr;
    
    return 0;
}

int fb_device_memory_mark_dirty(
    fb_device_memory_manager_t* manager,
    const void* host_ptr,
    fb_gpu_backend_type_t backend_type)
{
    if (!manager || !host_ptr) {
        return -1;
    }
    
    int idx = find_entry_by_host_ptr(manager, host_ptr);
    if (idx < 0) {
        fprintf(stderr, "Cannot mark dirty: host ptr %p not found\n", host_ptr);
        return -1;
    }
    
    fb_device_memory_entry_t* entry = &manager->entries[idx];
    entry->dirty = 1;
    entry->last_writer_backend = backend_type;
    entry->last_access_time = get_timestamp_ms();
    
    return 0;
}

int fb_device_memory_release(
    fb_device_memory_manager_t* manager,
    const void* host_ptr)
{
    if (!manager || !host_ptr) {
        return -1;
    }
    
    int idx = find_entry_by_host_ptr(manager, host_ptr);
    if (idx < 0) {
        fprintf(stderr, "Cannot release: host ptr %p not found\n", host_ptr);
        return -1;
    }
    
    fb_device_memory_entry_t* entry = &manager->entries[idx];
    
    if (entry->ref_count > 0) {
        entry->ref_count--;
    }
    
    /* Note: We keep the entry even when ref_count reaches 0 (cache behavior).
     * Memory will be reused on next get_or_alloc for same host_ptr.
     * Actual deallocation happens in fb_device_memory_free() or clear(). */
    
    return 0;
}

int fb_device_memory_sync_to_host(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle,
    void* host_ptr)
{
    if (!manager || !trait || !backend_handle || !host_ptr) {
        return -1;
    }
    
    int idx = find_entry_by_host_ptr(manager, host_ptr);
    if (idx < 0) {
        fprintf(stderr, "Cannot sync: host ptr %p not found\n", host_ptr);
        return -1;
    }
    
    fb_device_memory_entry_t* entry = &manager->entries[idx];
    
    /* Only sync if dirty */
    if (!entry->dirty) {
        return 0; /* No-op: already in sync */
    }
    
    /* Copy device → host */
    if (!trait->memcpy_d2h) {
        fprintf(stderr, "Backend trait missing memcpy_d2h function\n");
        return -1;
    }
    
    int result = trait->memcpy_d2h(backend_handle, host_ptr, 
                                    entry->device_ptr, entry->size);
    if (result != 0) {
        fprintf(stderr, "Failed to copy %zu bytes D2H\n", entry->size);
        return -1;
    }
    
    /* Mark as clean */
    entry->dirty = 0;
    
    manager->stats.num_d2h_copies++;
    manager->stats.bytes_d2h += entry->size;
    
    return 0;
}

int fb_device_memory_sync_all_to_host(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle)
{
    if (!manager || !trait || !backend_handle) {
        return -1;
    }
    
    int num_failures = 0;
    
    for (size_t i = 0; i < manager->num_entries; i++) {
        fb_device_memory_entry_t* entry = &manager->entries[i];
        
        if (entry->dirty && entry->host_ptr != NULL) {
            int result = fb_device_memory_sync_to_host(
                manager, trait, backend_handle, entry->host_ptr);
            
            if (result != 0) {
                num_failures++;
            }
        }
    }
    
    return num_failures;
}

int fb_device_memory_free(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle,
    const void* host_ptr)
{
    if (!manager || !trait || !backend_handle || !host_ptr) {
        return -1;
    }
    
    int idx = find_entry_by_host_ptr(manager, host_ptr);
    if (idx < 0) {
        fprintf(stderr, "Cannot free: host ptr %p not found\n", host_ptr);
        return -1;
    }
    
    fb_device_memory_entry_t* entry = &manager->entries[idx];
    
    /* Sync to host if dirty */
    if (entry->dirty) {
        fb_device_memory_sync_to_host(manager, trait, backend_handle, 
                                       (void*)host_ptr);
    }
    
    /* Free device memory */
    if (trait->free && entry->device_ptr != NULL) {
        trait->free(backend_handle, entry->device_ptr);
        manager->stats.num_deallocations++;
        manager->total_allocated -= entry->size;
    }
    
    /* Remove entry by swapping with last entry */
    if (idx < (int)manager->num_entries - 1) {
        manager->entries[idx] = manager->entries[manager->num_entries - 1];
    }
    
    /* Zero out last entry */
    memset(&manager->entries[manager->num_entries - 1], 0, 
           sizeof(fb_device_memory_entry_t));
    
    manager->num_entries--;
    
    return 0;
}

/* ============================================================================
 * Backend Synchronization
 * ========================================================================== */

int fb_device_memory_sync_backend_switch(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle,
    const void* host_ptr,
    fb_gpu_backend_type_t current_backend)
{
    if (!manager || !trait || !backend_handle || !host_ptr) {
        return -1;
    }
    
    int idx = find_entry_by_host_ptr(manager, host_ptr);
    if (idx < 0) {
        /* Not found - no sync needed (will be allocated fresh) */
        return 0;
    }
    
    fb_device_memory_entry_t* entry = &manager->entries[idx];
    
    /* Check if switching between different backends */
    if (entry->last_writer_backend != current_backend &&
        entry->last_writer_backend != FB_GPU_BACKEND_NONE) {
        
        /* Synchronize previous backend's stream */
        if (trait->stream_synchronize && backend_handle) {
            int result = trait->stream_synchronize(backend_handle, NULL);
            if (result != 0) {
                fprintf(stderr, "Warning: Failed to sync stream when switching backends\n");
                return -1;
            }
        }
    }
    
    return 0;
}

/* ============================================================================
 * Diagnostics and Statistics
 * ========================================================================== */

void fb_device_memory_get_stats(
    const fb_device_memory_manager_t* manager,
    size_t* total_allocated,
    size_t* num_entries,
    double* cache_hit_rate)
{
    if (!manager) {
        return;
    }
    
    if (total_allocated) {
        *total_allocated = manager->total_allocated;
    }
    
    if (num_entries) {
        *num_entries = manager->num_entries;
    }
    
    if (cache_hit_rate) {
        uint64_t total_accesses = manager->stats.num_cache_hits + 
                                   manager->stats.num_cache_misses;
        
        *cache_hit_rate = (total_accesses > 0) ? 
            (double)manager->stats.num_cache_hits / total_accesses : 0.0;
    }
}

void fb_device_memory_print_stats(
    const fb_device_memory_manager_t* manager)
{
    if (!manager) {
        return;
    }
    
    printf("\n=== Device Memory Manager Stats (Device %d) ===\n", 
           manager->device_id);
    printf("Total allocated: %zu bytes (%.2f MB)\n",
           manager->total_allocated,
           manager->total_allocated / (1024.0 * 1024.0));
    printf("Active entries: %zu / %zu capacity\n",
           manager->num_entries, manager->capacity);
    printf("Allocations: %llu\n", 
           (unsigned long long)manager->stats.num_allocations);
    printf("Deallocations: %llu\n",
           (unsigned long long)manager->stats.num_deallocations);
    printf("Cache hits: %llu\n",
           (unsigned long long)manager->stats.num_cache_hits);
    printf("Cache misses: %llu\n",
           (unsigned long long)manager->stats.num_cache_misses);
    
    uint64_t total_accesses = manager->stats.num_cache_hits + 
                               manager->stats.num_cache_misses;
    if (total_accesses > 0) {
        double hit_rate = (double)manager->stats.num_cache_hits / total_accesses;
        printf("Cache hit rate: %.2f%%\n", hit_rate * 100.0);
    }
    
    printf("H2D copies: %llu (%.2f MB)\n",
           (unsigned long long)manager->stats.num_h2d_copies,
           manager->stats.bytes_h2d / (1024.0 * 1024.0));
    printf("D2H copies: %llu (%.2f MB)\n",
           (unsigned long long)manager->stats.num_d2h_copies,
           manager->stats.bytes_d2h / (1024.0 * 1024.0));
    printf("==========================================\n\n");
}

void fb_device_memory_dump_entries(
    const fb_device_memory_manager_t* manager)
{
    if (!manager) {
        return;
    }
    
    printf("\n=== Device Memory Entries (Device %d) ===\n", 
           manager->device_id);
    printf("%-4s %-18s %-18s %-10s %-5s %-8s %-15s\n",
           "Idx", "Host Ptr", "Device Ptr", "Size", "Refs", "Dirty", "Last Writer");
    printf("------------------------------------------------------------------------------------\n");
    
    for (size_t i = 0; i < manager->num_entries; i++) {
        const fb_device_memory_entry_t* entry = &manager->entries[i];
        
        const char* backend_name = "UNKNOWN";
        switch (entry->last_writer_backend) {
            case FB_GPU_BACKEND_CUBLAS: backend_name = "cuBLAS"; break;
            case FB_GPU_BACKEND_ROCBLAS: backend_name = "rocBLAS"; break;
            case FB_GPU_BACKEND_CLBLAST: backend_name = "CLBlast"; break;
            case FB_GPU_BACKEND_ONEMKL: backend_name = "oneMKL"; break;
            default: backend_name = "NONE"; break;
        }
        
        printf("%-4zu %p %p %-10zu %-5d %-8s %-15s\n",
               i, entry->host_ptr, entry->device_ptr,
               entry->size, entry->ref_count,
               entry->dirty ? "YES" : "NO",
               backend_name);
    }
    
    printf("==========================================\n\n");
}

/* ============================================================================
 * Global Manager Access
 * ========================================================================== */

fb_device_memory_manager_t* fb_device_memory_get_manager(int device_id) {
    if (device_id < 0 || device_id >= MAX_DEVICES) {
        fprintf(stderr, "Invalid device_id: %d\n", device_id);
        return NULL;
    }
    
    /* Create on-demand if doesn't exist */
    if (g_managers[device_id] == NULL) {
        g_managers[device_id] = fb_device_memory_manager_create(device_id, 0);
    }
    
    return g_managers[device_id];
}

void fb_device_memory_shutdown_all(void) {
    for (int i = 0; i < MAX_DEVICES; i++) {
        if (g_managers[i] != NULL) {
            fb_device_memory_manager_destroy(g_managers[i]);
            g_managers[i] = NULL;
        }
    }
}
