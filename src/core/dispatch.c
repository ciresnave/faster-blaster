/**
 * @file dispatch.c
 * @brief Dispatch system implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "dispatch.h"
#include "hardware_detect.h"
#include "calibration.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global dispatch table (initialized on fb_init) */
static fb_dispatch_table_t g_dispatch = {0};

fb_dispatch_table_t *fb_dispatch_table_create(void) {
    fb_dispatch_table_t *table = calloc(1, sizeof(fb_dispatch_table_t));
    if (!table) return NULL;
    
    table->policy = FB_DISPATCH_AUTO;
    table->active_backend_idx = 0;
    
    return table;
}

void fb_dispatch_table_destroy(fb_dispatch_table_t *table) {
    if (!table) return;
    
    /* Finalize all backends - commented out, no finalize in vtable yet */
    /* for (size_t i = 0; i < table->num_backends; i++) {
        if (table->backends[i] && table->backends[i]->finalize) {
            table->backends[i]->finalize();
        }
    } */
    
    free(table);
}

int fb_dispatch_register_backend(
    fb_dispatch_table_t *table,
    const char *name,
    const fb_backend_vtable_t *backend)
{
    if (!table || !name || !backend) return -1;
    if (table->num_backends >= MAX_BACKENDS) return -1;
    
    /* Initialize backend */
    // COMMENTED OUT: vtable doesn't have init member yet
    // if (backend->init && backend->init() != 0) {
    //     return -1;
    // }
    
    size_t idx = table->num_backends;
    table->backends[idx] = backend;
    table->backend_names[idx] = name;
    table->num_backends++;
    
    return 0;
}

int fb_dispatch_set_policy(
    fb_dispatch_table_t *table,
    fb_dispatch_policy_t policy)
{
    if (!table) return -1;
    table->policy = policy;
    return 0;
}

int fb_dispatch_set_active_backend(
    fb_dispatch_table_t *table,
    const char *name)
{
    if (!table || !name) return -1;
    
    for (size_t i = 0; i < table->num_backends; i++) {
        if (strcmp(table->backend_names[i], name) == 0) {
            table->active_backend_idx = i;
            return 0;
        }
    }
    
    return -1;  /* Backend not found */
}

const fb_backend_vtable_t *fb_dispatch_get_backend(
    const fb_dispatch_table_t *table)
{
    if (!table || table->num_backends == 0) return NULL;
    return table->backends[table->active_backend_idx];
}

const char *fb_dispatch_get_active_backend_name(
    const fb_dispatch_table_t *table)
{
    if (!table || table->num_backends == 0) return NULL;
    return table->backend_names[table->active_backend_idx];
}

int fb_dispatch_list_backends(
    const fb_dispatch_table_t *table,
    const char **names,
    size_t max_names)
{
    if (!table || !names) return -1;
    
    size_t count = table->num_backends < max_names ? table->num_backends : max_names;
    for (size_t i = 0; i < count; i++) {
        names[i] = table->backend_names[i];
    }
    
    return (int)count;
}

// COMMENTED OUT: Uses hardware_info field that was removed
// static int select_best_backend_auto(fb_dispatch_table_t *table) {
//     ... hardware detection and backend selection logic ...
// }

int fb_dispatch_auto_select(fb_dispatch_table_t *table) {
    if (!table) return -1;
    
    /* Simplified: just use first backend if not manually set */
    if (table->policy == FB_DISPATCH_MANUAL) {
        return 0;
    }
    
    /* Default to first available backend */
    if (table->num_backends > 0) {
        table->active_backend_idx = 0;
        return 0;
    }
    
    return -1;
}

// COMMENTED OUT: fb_calibration_config_t doesn't match new calibration.h API
// int fb_dispatch_calibrate_all(
//     fb_dispatch_table_t *table,
//     const fb_calibration_config_t *config,
//     const char *output_dir)
// {
//     ... calibration implementation ...
// }

/* ============================================================================
 * Global Dispatch Table Accessors
 * ========================================================================= */

/* Global dispatch table accessors */
fb_dispatch_table_t *fb_dispatch_global(void) {
    return &g_dispatch;
}

int fb_dispatch_init_global(void) {
    if (g_dispatch.initialized) return 0;
    
    g_dispatch.policy = FB_DISPATCH_AUTO;
    g_dispatch.num_backends = 0;
    g_dispatch.active_backend_idx = 0;
    g_dispatch.initialized = true;
    
    return 0;
}

void fb_dispatch_finalize_global(void) {
    if (!g_dispatch.initialized) return;
    
    // COMMENTED OUT: vtable doesn't have finalize member yet
    // for (size_t i = 0; i < g_dispatch.num_backends; i++) {
    //     if (g_dispatch.backends[i] && g_dispatch.backends[i]->finalize) {
    //         g_dispatch.backends[i]->finalize();
    //     }
    // }
    
    memset(&g_dispatch, 0, sizeof(fb_dispatch_table_t));
}

