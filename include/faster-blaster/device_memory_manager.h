/**
 * @file device_memory_manager.h
 * @brief Device-level memory manager for GPU operations
 * 
 * This manager tracks host-to-device memory mappings across multiple backends
 * on the same physical device. It enables efficient operation chaining by:
 * - Keeping data on device between operations
 * - Sharing device memory across different backends (e.g., rocBLAS → CLBlast)
 * - Automatic device synchronization when switching backends
 * - Lazy D2H copies (only when needed)
 * 
 * Key design principles:
 * - Device-level scope (one manager per GPU, not per backend)
 * - Backend-agnostic (any backend can use any device memory)
 * - Reference counting for safe memory management
 * - Transparent to callers (no API changes needed)
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_DEVICE_MEMORY_MANAGER_H
#define FASTER_BLASTER_DEVICE_MEMORY_MANAGER_H

#include <stddef.h>
#include <stdint.h>
#include "gpu_backend_trait.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Types and Structures
 * ========================================================================== */

/**
 * @brief Device memory entry - tracks a single host↔device mapping
 */
typedef struct fb_device_memory_entry {
    void* host_ptr;              /**< Host memory address */
    fb_gpu_ptr_t device_ptr;     /**< Device memory address */
    size_t size;                 /**< Buffer size in bytes */
    int device_id;               /**< Physical device (GPU) ID */
    
    /** Dirty flag: true if device data was modified and needs D2H copy */
    int dirty;
    
    /** Reference count: how many operations are using this buffer */
    int ref_count;
    
    /** Last backend that wrote to this buffer (for synchronization) */
    fb_gpu_backend_type_t last_writer_backend;
    
    /** Timestamp of last access (for LRU eviction) */
    uint64_t last_access_time;
    
} fb_device_memory_entry_t;

/**
 * @brief Device memory manager - one per physical GPU
 */
typedef struct fb_device_memory_manager {
    /** Device ID */
    int device_id;
    
    /** Array of memory mappings */
    fb_device_memory_entry_t* entries;
    size_t num_entries;
    size_t capacity;
    
    /** Total device memory allocated (bytes) */
    size_t total_allocated;
    
    /** Maximum memory to use (bytes, 0 = no limit) */
    size_t max_memory;
    
    /** Statistics */
    struct {
        uint64_t num_allocations;
        uint64_t num_deallocations;
        uint64_t num_h2d_copies;
        uint64_t num_d2h_copies;
        uint64_t num_cache_hits;    /**< Host ptr already on device */
        uint64_t num_cache_misses;  /**< Host ptr not on device */
        uint64_t bytes_h2d;
        uint64_t bytes_d2h;
    } stats;
    
} fb_device_memory_manager_t;

/* ============================================================================
 * Manager Lifecycle
 * ========================================================================== */

/**
 * @brief Initialize device memory manager for a specific device
 * 
 * @param device_id Physical device ID (0 = first GPU, 1 = second GPU, etc.)
 * @param max_memory Maximum device memory to use (0 = no limit)
 * @return Pointer to manager, or NULL on error
 */
fb_device_memory_manager_t* fb_device_memory_manager_create(
    int device_id,
    size_t max_memory
);

/**
 * @brief Destroy device memory manager and free all tracked memory
 * 
 * This will:
 * - Copy all dirty buffers back to host (D2H)
 * - Free all device memory
 * - Destroy the manager
 * 
 * @param manager Manager to destroy
 */
void fb_device_memory_manager_destroy(fb_device_memory_manager_t* manager);

/**
 * @brief Clear all entries but keep manager alive
 * 
 * @param manager Manager to clear
 * @param sync_dirty If true, copy dirty buffers to host before clearing
 */
void fb_device_memory_manager_clear(
    fb_device_memory_manager_t* manager,
    int sync_dirty
);

/* ============================================================================
 * Memory Operations
 * ========================================================================== */

/**
 * @brief Get or allocate device memory for a host pointer
 * 
 * This is the main entry point for GPU wrappers. It:
 * 1. Checks if host_ptr is already mapped to device memory
 * 2. If yes (cache hit): returns existing device pointer, increments ref count
 * 3. If no (cache miss): allocates device memory, performs H2D copy, registers mapping
 * 
 * @param manager Device memory manager
 * @param trait GPU backend trait for memory allocation
 * @param backend_handle Backend-specific handle (e.g., cuBLAS handle)
 * @param host_ptr Host memory address
 * @param size Buffer size in bytes
 * @param backend_type Type of backend making this call (for tracking)
 * @param device_ptr_out Output: device memory address
 * @return 0 on success, -1 on error
 */
int fb_device_memory_get_or_alloc(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle,
    const void* host_ptr,
    size_t size,
    fb_gpu_backend_type_t backend_type,
    fb_gpu_ptr_t* device_ptr_out
);

/**
 * @brief Mark device memory as dirty (needs D2H sync before host can read)
 * 
 * Call this after an operation that writes to device memory.
 * 
 * @param manager Device memory manager
 * @param host_ptr Host memory address (key to lookup device buffer)
 * @param backend_type Backend that wrote to the buffer
 * @return 0 on success, -1 if host_ptr not found
 */
int fb_device_memory_mark_dirty(
    fb_device_memory_manager_t* manager,
    const void* host_ptr,
    fb_gpu_backend_type_t backend_type
);

/**
 * @brief Release reference to device memory
 * 
 * Decrements reference count. When ref count reaches 0, memory becomes
 * eligible for eviction (but is not immediately freed - kept as cache).
 * 
 * @param manager Device memory manager
 * @param host_ptr Host memory address
 * @return 0 on success, -1 if host_ptr not found
 */
int fb_device_memory_release(
    fb_device_memory_manager_t* manager,
    const void* host_ptr
);

/**
 * @brief Synchronize device memory to host (D2H copy if dirty)
 * 
 * If the buffer is dirty (device has newer data than host), performs D2H copy.
 * Otherwise, this is a no-op.
 * 
 * @param manager Device memory manager
 * @param trait GPU backend trait for memory copy
 * @param backend_handle Backend-specific handle
 * @param host_ptr Host memory address
 * @return 0 on success, -1 if host_ptr not found or copy failed
 */
int fb_device_memory_sync_to_host(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle,
    void* host_ptr
);

/**
 * @brief Synchronize all dirty buffers to host
 * 
 * Useful before returning control to user or when switching devices.
 * 
 * @param manager Device memory manager
 * @param trait GPU backend trait for memory copy
 * @param backend_handle Backend-specific handle
 * @return 0 on success, number of failures on error
 */
int fb_device_memory_sync_all_to_host(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle
);

/**
 * @brief Free device memory for a specific host pointer
 * 
 * Performs D2H copy if dirty, then frees device memory.
 * 
 * @param manager Device memory manager
 * @param trait GPU backend trait for memory operations
 * @param backend_handle Backend-specific handle
 * @param host_ptr Host memory address
 * @return 0 on success, -1 if host_ptr not found
 */
int fb_device_memory_free(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle,
    const void* host_ptr
);

/* ============================================================================
 * Backend Synchronization
 * ========================================================================== */

/**
 * @brief Synchronize device when switching between backends
 * 
 * When operation chains involve multiple backends (e.g., rocBLAS → CLBlast),
 * we need to ensure that:
 * 1. Previous backend's stream has completed
 * 2. Memory is visible to the next backend
 * 
 * This function checks if the last writer was a different backend and
 * performs necessary synchronization.
 * 
 * @param manager Device memory manager
 * @param trait Current backend's trait
 * @param backend_handle Current backend's handle
 * @param host_ptr Host memory address of buffer being accessed
 * @param current_backend Type of current backend
 * @return 0 if no sync needed or sync succeeded, -1 on error
 */
int fb_device_memory_sync_backend_switch(
    fb_device_memory_manager_t* manager,
    const fb_gpu_backend_trait_t* trait,
    void* backend_handle,
    const void* host_ptr,
    fb_gpu_backend_type_t current_backend
);

/* ============================================================================
 * Diagnostics and Statistics
 * ========================================================================== */

/**
 * @brief Get statistics about device memory usage
 * 
 * @param manager Device memory manager
 * @param total_allocated Output: total bytes allocated
 * @param num_entries Output: number of tracked entries
 * @param cache_hit_rate Output: cache hit rate (0.0-1.0)
 */
void fb_device_memory_get_stats(
    const fb_device_memory_manager_t* manager,
    size_t* total_allocated,
    size_t* num_entries,
    double* cache_hit_rate
);

/**
 * @brief Print device memory manager statistics to stdout
 * 
 * @param manager Device memory manager
 */
void fb_device_memory_print_stats(
    const fb_device_memory_manager_t* manager
);

/**
 * @brief List all tracked memory entries (for debugging)
 * 
 * @param manager Device memory manager
 */
void fb_device_memory_dump_entries(
    const fb_device_memory_manager_t* manager
);

/* ============================================================================
 * Global Manager Access
 * ========================================================================== */

/**
 * @brief Get device memory manager for a specific device
 * 
 * Managers are created on-demand and cached globally.
 * 
 * @param device_id Physical device ID
 * @return Pointer to manager, or NULL on error
 */
fb_device_memory_manager_t* fb_device_memory_get_manager(int device_id);

/**
 * @brief Shutdown all device memory managers
 * 
 * Call this at program exit to free all device memory and managers.
 */
void fb_device_memory_shutdown_all(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_DEVICE_MEMORY_MANAGER_H */
