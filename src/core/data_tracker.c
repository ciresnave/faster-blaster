/**
 * @file data_tracker.c
 * @brief Data Location Tracking Implementation
 */

#include "core/data_tracker.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Simple hash table for pointer tracking
#define TRACKER_HASH_SIZE 4096

typedef struct data_entry {
    const void* ptr;
    fb_data_location_t location;
    size_t size_bytes;
    uint64_t last_access_time;
    uint32_t access_count;
    struct data_entry* next;
} data_entry_t;

typedef struct {
    data_entry_t* buckets[TRACKER_HASH_SIZE];
    uint64_t time_counter;
    bool initialized;
} data_tracker_state_t;

static data_tracker_state_t g_tracker = {0};

// ============================================================================
// Hash Function
// ============================================================================

static uint32_t hash_ptr(const void* ptr) {
    uintptr_t p = (uintptr_t)ptr;
    // Simple hash: mix bits
    p = ((p >> 16) ^ p) * 0x45d9f3b;
    p = ((p >> 16) ^ p) * 0x45d9f3b;
    p = (p >> 16) ^ p;
    return p % TRACKER_HASH_SIZE;
}

// ============================================================================
// Initialization
// ============================================================================

int fb_data_tracker_init(void) {
    if (g_tracker.initialized) {
        return 0;
    }
    
    memset(&g_tracker, 0, sizeof(g_tracker));
    g_tracker.time_counter = 0;
    g_tracker.initialized = true;
    
    return 0;
}

void fb_data_tracker_shutdown(void) {
    if (!g_tracker.initialized) return;
    
    // Free all entries
    for (uint32_t i = 0; i < TRACKER_HASH_SIZE; i++) {
        data_entry_t* entry = g_tracker.buckets[i];
        while (entry) {
            data_entry_t* next = entry->next;
            free(entry);
            entry = next;
        }
    }
    
    memset(&g_tracker, 0, sizeof(g_tracker));
}

// ============================================================================
// Registration
// ============================================================================

int fb_data_register(const void* ptr, fb_data_location_t location, size_t size_bytes) {
    if (!g_tracker.initialized) {
        fb_data_tracker_init();
    }
    
    uint32_t bucket = hash_ptr(ptr);
    
    // Check if already registered
    data_entry_t* entry = g_tracker.buckets[bucket];
    while (entry) {
        if (entry->ptr == ptr) {
            // Update existing entry
            entry->location = location;
            entry->size_bytes = size_bytes;
            entry->last_access_time = g_tracker.time_counter++;
            return 0;
        }
        entry = entry->next;
    }
    
    // Create new entry
    entry = (data_entry_t*)malloc(sizeof(data_entry_t));
    if (!entry) return -1;
    
    entry->ptr = ptr;
    entry->location = location;
    entry->size_bytes = size_bytes;
    entry->last_access_time = g_tracker.time_counter++;
    entry->access_count = 0;
    entry->next = g_tracker.buckets[bucket];
    
    g_tracker.buckets[bucket] = entry;
    
    return 0;
}

int fb_data_unregister(const void* ptr) {
    if (!g_tracker.initialized) return -1;
    
    uint32_t bucket = hash_ptr(ptr);
    data_entry_t** prev = &g_tracker.buckets[bucket];
    data_entry_t* entry = *prev;
    
    while (entry) {
        if (entry->ptr == ptr) {
            *prev = entry->next;
            free(entry);
            return 0;
        }
        prev = &entry->next;
        entry = entry->next;
    }
    
    return -1; // Not found
}

// ============================================================================
// Queries
// ============================================================================

fb_data_location_t fb_data_get_location(const void* ptr) {
    if (!g_tracker.initialized) {
        fb_data_tracker_init();
    }
    
    uint32_t bucket = hash_ptr(ptr);
    data_entry_t* entry = g_tracker.buckets[bucket];
    
    while (entry) {
        if (entry->ptr == ptr) {
            entry->access_count++;
            entry->last_access_time = g_tracker.time_counter++;
            return entry->location;
        }
        entry = entry->next;
    }
    
    // Unknown location - assume host
    return FB_DATA_HOST;
}

bool fb_data_is_on_device(const void* ptr, fb_compute_device_t* device) {
    fb_data_location_t location = fb_data_get_location(ptr);
    
    if (location.type == FB_DATA_HOST && device->type == FB_DEVICE_CPU) {
        return true;
    }
    
    if (location.type == FB_DATA_DEVICE && device->type == FB_DEVICE_GPU) {
        return location.device_id == device->device_id;
    }
    
    if (location.type == FB_DATA_UNIFIED) {
        return true; // Accessible from all devices
    }
    
    return false;
}

size_t fb_data_get_size(const void* ptr) {
    if (!g_tracker.initialized) return 0;
    
    uint32_t bucket = hash_ptr(ptr);
    data_entry_t* entry = g_tracker.buckets[bucket];
    
    while (entry) {
        if (entry->ptr == ptr) {
            return entry->size_bytes;
        }
        entry = entry->next;
    }
    
    return 0;
}

// ============================================================================
// Transfer Cost Estimation
// ============================================================================

double fb_estimate_transfer_cost(const void* ptr, fb_compute_device_t* target_device) {
    fb_data_location_t current_loc = fb_data_get_location(ptr);
    size_t size = fb_data_get_size(ptr);
    
    if (size == 0) {
        size = 1024 * 1024; // Default estimate: 1 MB
    }
    
    // Already on target device?
    if (fb_data_is_on_device(ptr, target_device)) {
        return 0.0; // No transfer needed
    }
    
    // Unified memory - small cost
    if (current_loc.type == FB_DATA_UNIFIED) {
        return size / (100.0 * 1024.0 * 1024.0 * 1024.0); // 100 GB/s
    }
    
    // Estimate PCIe transfer time
    double bandwidth_gbps;
    
    if (current_loc.type == FB_DATA_HOST && target_device->type == FB_DEVICE_GPU) {
        // Host -> GPU (PCIe)
        bandwidth_gbps = 16.0; // PCIe 4.0 x16
    } else if (current_loc.type == FB_DATA_DEVICE && target_device->type == FB_DEVICE_CPU) {
        // GPU -> Host (PCIe)
        bandwidth_gbps = 16.0;
    } else if (current_loc.type == FB_DATA_DEVICE && target_device->type == FB_DEVICE_GPU) {
        // GPU -> GPU (P2P or via host)
        // TODO: Check if P2P is available
        bandwidth_gbps = 16.0; // Conservative estimate
    } else {
        bandwidth_gbps = 10.0; // Generic estimate
    }
    
    // Transfer time in seconds
    double size_gb = size / (1024.0 * 1024.0 * 1024.0);
    double transfer_time_s = size_gb / bandwidth_gbps;
    
    return transfer_time_s;
}

double fb_estimate_prefetch_benefit(const void* ptr, fb_compute_device_t* device) {
    // Estimate benefit of prefetching data to device
    // Returns expected time saved (in seconds)
    
    double transfer_cost = fb_estimate_transfer_cost(ptr, device);
    
    // If data is already on device, no benefit
    if (transfer_cost == 0.0) return 0.0;
    
    // Benefit is avoiding blocking transfer later
    // Assume 80% of transfer cost can be hidden by async prefetch
    return transfer_cost * 0.8;
}

// ============================================================================
// Access Pattern Tracking
// ============================================================================

uint32_t fb_data_get_access_count(const void* ptr) {
    if (!g_tracker.initialized) return 0;
    
    uint32_t bucket = hash_ptr(ptr);
    data_entry_t* entry = g_tracker.buckets[bucket];
    
    while (entry) {
        if (entry->ptr == ptr) {
            return entry->access_count;
        }
        entry = entry->next;
    }
    
    return 0;
}

void fb_data_record_access(const void* ptr) {
    if (!g_tracker.initialized) {
        fb_data_tracker_init();
    }
    
    uint32_t bucket = hash_ptr(ptr);
    data_entry_t* entry = g_tracker.buckets[bucket];
    
    while (entry) {
        if (entry->ptr == ptr) {
            entry->access_count++;
            entry->last_access_time = g_tracker.time_counter++;
            return;
        }
        entry = entry->next;
    }
    
    // Not registered - register as host memory
    fb_data_location_t loc = { FB_DATA_HOST, 0 };
    fb_data_register(ptr, loc, 0);
}

// ============================================================================
// Bulk Operations
// ============================================================================

int fb_data_mark_transfer(const void* ptr, fb_compute_device_t* new_device) {
    fb_data_location_t new_loc;
    
    if (new_device->type == FB_DEVICE_CPU) {
        new_loc.type = FB_DATA_HOST;
        new_loc.device_id = 0;
    } else {
        new_loc.type = FB_DATA_DEVICE;
        new_loc.device_id = new_device->device_id;
    }
    
    size_t size = fb_data_get_size(ptr);
    return fb_data_register(ptr, new_loc, size);
}

void fb_data_clear_all(void) {
    fb_data_tracker_shutdown();
    fb_data_tracker_init();
}

// ============================================================================
// Debug/Stats
// ============================================================================

void fb_data_print_stats(void) {
    if (!g_tracker.initialized) {
        printf("Data tracker not initialized\n");
        return;
    }
    
    uint32_t total_entries = 0;
    uint32_t host_entries = 0;
    uint32_t device_entries = 0;
    uint32_t unified_entries = 0;
    size_t total_bytes = 0;
    
    for (uint32_t i = 0; i < TRACKER_HASH_SIZE; i++) {
        data_entry_t* entry = g_tracker.buckets[i];
        while (entry) {
            total_entries++;
            total_bytes += entry->size_bytes;
            
            switch (entry->location.type) {
                case FB_DATA_HOST: host_entries++; break;
                case FB_DATA_DEVICE: device_entries++; break;
                case FB_DATA_UNIFIED: unified_entries++; break;
                default: break;
            }
            
            entry = entry->next;
        }
    }
    
    printf("=== Data Tracker Statistics ===\n");
    printf("Total tracked allocations: %u\n", total_entries);
    printf("  Host: %u\n", host_entries);
    printf("  Device: %u\n", device_entries);
    printf("  Unified: %u\n", unified_entries);
    printf("Total tracked memory: %.2f MB\n", total_bytes / (1024.0 * 1024.0));
}
