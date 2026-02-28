/**
 * @file data_tracker.c
 * @brief Data Location Tracking (matches data_tracker.h API)
 */

#include "faster-blaster/data_tracker.h"
#include <stdlib.h>
#include <string.h>

#define FB_DEVICE_ID_HOST (-1)
#define TRACKER_HASH_SIZE 4096u

typedef struct data_entry {
    const void       *ptr;
    fb_data_info_t    info;
    struct data_entry *next;
} data_entry_t;

typedef struct {
    data_entry_t *buckets[TRACKER_HASH_SIZE];
    bool          initialized;
} data_tracker_state_t;

static data_tracker_state_t g_tracker = {0};

static uint32_t hash_ptr(const void *ptr) {
    uintptr_t p = (uintptr_t)ptr;
    p = ((p >> 16) ^ p) * 0x45d9f3bu;
    p = ((p >> 16) ^ p) * 0x45d9f3bu;
    return (uint32_t)((p >> 16) ^ p) % TRACKER_HASH_SIZE;
}

int fb_data_tracker_init(void) {
    if (g_tracker.initialized) return 0;
    memset(&g_tracker, 0, sizeof(g_tracker));
    g_tracker.initialized = true;
    return 0;
}

void fb_data_tracker_shutdown(void) {
    if (!g_tracker.initialized) return;
    for (uint32_t i = 0; i < TRACKER_HASH_SIZE; i++) {
        data_entry_t *e = g_tracker.buckets[i];
        while (e) { data_entry_t *n = e->next; free(e); e = n; }
        g_tracker.buckets[i] = NULL;
    }
    g_tracker.initialized = false;
}

void fb_data_register(const void *ptr, size_t size, int device_id,
                      bool is_pinned, bool is_managed) {
    if (!ptr) return;
    if (!g_tracker.initialized) fb_data_tracker_init();
    uint32_t bucket = hash_ptr(ptr);
    for (data_entry_t *e = g_tracker.buckets[bucket]; e; e = e->next) {
        if (e->ptr != ptr) continue;
        e->info.size = size; e->info.device_id = device_id;
        e->info.is_pinned = is_pinned; e->info.is_managed = is_managed;
        return;
    }
    data_entry_t *entry = (data_entry_t *)malloc(sizeof(data_entry_t));
    if (!entry) return;
    entry->ptr = ptr;
    entry->info.ptr = ptr; entry->info.size = size;
    entry->info.device_id = device_id; entry->info.is_pinned = is_pinned;
    entry->info.is_managed = is_managed; entry->info.last_access_time = 0;
    entry->info.access_count = 0; entry->info.is_temporary = false;
    entry->next = g_tracker.buckets[bucket];
    g_tracker.buckets[bucket] = entry;
}

void fb_data_unregister(const void *ptr) {
    if (!ptr || !g_tracker.initialized) return;
    uint32_t bucket = hash_ptr(ptr);
    data_entry_t **prev = &g_tracker.buckets[bucket], *e = *prev;
    while (e) {
        if (e->ptr == ptr) { *prev = e->next; free(e); return; }
        prev = &e->next; e = e->next;
    }
}

int fb_data_query_location(const void *ptr, fb_data_info_t *info) {
    if (!ptr || !g_tracker.initialized) return -1;
    uint32_t bucket = hash_ptr(ptr);
    for (data_entry_t *e = g_tracker.buckets[bucket]; e; e = e->next) {
        if (e->ptr == ptr) {
            e->info.access_count++;
            if (info) *info = e->info;
            return e->info.device_id;
        }
    }
    return -1;
}

void fb_data_record_access(const void *ptr, int device_id) {
    if (!ptr || !g_tracker.initialized) return;
    uint32_t bucket = hash_ptr(ptr);
    for (data_entry_t *e = g_tracker.buckets[bucket]; e; e = e->next) {
        if (e->ptr == ptr) { e->info.access_count++; e->info.device_id = device_id; return; }
    }
}

double fb_data_estimate_transfer_cost(const void *ptr, int target_device) {
    if (!ptr) return 0.0;
    fb_data_info_t info = {0};
    int current = fb_data_query_location(ptr, &info);
    if (current == target_device) return 0.0;
    size_t sz = info.size > 0 ? info.size : (1u << 20);
    /* PCIe 4.0 x16 ~16 GB/s, return microseconds */
    return ((double)sz / (16.0 * 1024.0 * 1024.0 * 1024.0)) * 1.0e6;
}

double fb_data_estimate_operation_cost(const void **input_ptrs, int input_count,
                                       const void **output_ptrs, int output_count,
                                       int target_device) {
    double total = 0.0;
    if (input_ptrs)
        for (int i = 0; i < input_count; i++)
            if (input_ptrs[i])
                total += fb_data_estimate_transfer_cost(input_ptrs[i], target_device);
    if (output_ptrs)
        for (int i = 0; i < output_count; i++)
            if (output_ptrs[i])
                total += fb_data_estimate_transfer_cost(output_ptrs[i], target_device);
    return total;
}

int fb_data_recommend_device(const void **input_ptrs, int input_count,
                              const void **output_ptrs, int output_count) {
#define MAX_DEVS 32
    int ids[MAX_DEVS] = {0}; size_t szs[MAX_DEVS] = {0}; int n = 0;
    const void **ptrs[2] = {input_ptrs, output_ptrs};
    int cnts[2] = {input_count, output_count};
    for (int g = 0; g < 2; g++) {
        if (!ptrs[g]) continue;
        for (int i = 0; i < cnts[g]; i++) {
            if (!ptrs[g][i]) continue;
            fb_data_info_t inf = {0};
            int dev = fb_data_query_location(ptrs[g][i], &inf);
            if (dev < 0) dev = FB_DEVICE_ID_HOST;
            bool found = false;
            for (int d = 0; d < n; d++)
                if (ids[d] == dev) { szs[d] += inf.size; found = true; break; }
            if (!found && n < MAX_DEVS) { ids[n] = dev; szs[n++] = inf.size; }
        }
    }
    int best = FB_DEVICE_ID_HOST; size_t best_sz = 0;
    for (int d = 0; d < n; d++) if (szs[d] > best_sz) { best_sz = szs[d]; best = ids[d]; }
    return best;
#undef MAX_DEVS
}

double fb_data_locality_score(const void **input_ptrs, int input_count,
                               const void **output_ptrs, int output_count,
                               int device_id) {
    size_t local = 0, total = 0;
    const void **ptrs[2] = {input_ptrs, output_ptrs};
    int cnts[2] = {input_count, output_count};
    for (int g = 0; g < 2; g++) {
        if (!ptrs[g]) continue;
        for (int i = 0; i < cnts[g]; i++) {
            if (!ptrs[g][i]) continue;
            fb_data_info_t inf = {0};
            int dev = fb_data_query_location(ptrs[g][i], &inf);
            total += inf.size;
            if (dev == device_id || inf.is_managed) local += inf.size;
        }
    }
    return total > 0 ? ((double)local / (double)total) : 0.5;
}

int fb_data_prefetch(const void *ptr, int target_device) {
    if (!ptr) return -1;
    fb_data_info_t inf = {0};
    fb_data_query_location(ptr, &inf);
    fb_data_register(ptr, inf.size, target_device, inf.is_pinned, inf.is_managed);
    return 0;
}
