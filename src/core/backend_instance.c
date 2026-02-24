/**
 * @file backend_instance.c
 * @brief Implementation of backend instance management
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_instance.h"
#include "faster-blaster/backend_plugin.h"
#include "faster-blaster/device_registry.h"
#include "backends/backend_interface.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global backend instance manager */
static fb_backend_instance_manager_t g_instance_manager = {0};

/* Reference backend (always available as fallback) */
extern const fb_backend_vtable_t* fb_reference_get_vtable(void);

/* Forward declarations for device-to-plugin mapping */
static const char** select_plugins_for_device(const fb_compute_device_t* device, int* count);
static int initialize_backend_instance(fb_backend_instance_t* instance,
                                       fb_compute_device_t* device,
                                       const char* plugin_name);

/* ============================================================================
 * Initialization
 * ========================================================================== */

int fb_backend_instance_manager_init(void) {
    if (g_instance_manager.initialized) {
        return 0; /* Already initialized */
    }
    
    /* Start with capacity for 16 devices (typical multi-GPU + CPU cores) */
    g_instance_manager.capacity = 16;
    g_instance_manager.instances = (fb_backend_instance_t*)calloc(
        g_instance_manager.capacity, sizeof(fb_backend_instance_t));
    
    if (!g_instance_manager.instances) {
        fprintf(stderr, "[ERROR] Failed to allocate backend instance array\n");
        return -1;
    }
    
    g_instance_manager.num_instances = 0;
    g_instance_manager.initialized = true;
    
    printf("[INFO] Backend instance manager initialized\n");
    return 0;
}

void fb_backend_instance_manager_shutdown(void) {
    if (!g_instance_manager.initialized) {
        return;
    }
    
    /* Shutdown all loaded backends */
    for (uint32_t i = 0; i < g_instance_manager.num_instances; i++) {
        fb_backend_instance_t* instance = &g_instance_manager.instances[i];
        if (instance->initialized && instance->plugin && instance->plugin->shutdown) {
            printf("[INFO] Shutting down backend for device %u\n", instance->device->properties.device_id);
            instance->plugin->shutdown(instance->plugin_ctx);
        }
    }
    
    free(g_instance_manager.instances);
    memset(&g_instance_manager, 0, sizeof(g_instance_manager));
    
    printf("[INFO] Backend instance manager shutdown complete\n");
}

/* ============================================================================
 * Device-to-Plugin Mapping - Returns ALL compatible backends
 * ========================================================================== */

/* Select compatible plugins for device - returns list of plugin names */
static const char** select_plugins_for_device(const fb_compute_device_t* device, int* count) {
    static const char* plugins[8]; /* Max 8 backends per device */
    int idx = 0;
    
    if (!device || !count) {
        if (count) *count = 0;
        return NULL;
    }
    
    /* GPU devices - try all compatible backends */
    if (device->properties.type == FB_DEVICE_TYPE_GPU) {
        switch (device->properties.gpu.vendor) {
            case FB_GPU_VENDOR_NVIDIA:
                plugins[idx++] = "cublas";   /* Primary: cuBLAS */
                plugins[idx++] = "clblast";  /* Fallback: OpenCL */
                break;
            case FB_GPU_VENDOR_AMD:
                plugins[idx++] = "rocblas";  /* For discrete AMD GPUs with ROCm */
                plugins[idx++] = "clblast";  /* For all AMD GPUs via OpenCL */
                plugins[idx++] = "clblas";   /* Alternative OpenCL */
                break;
            case FB_GPU_VENDOR_INTEL:
                plugins[idx++] = "onemkl";   /* Intel Arc/Xe → oneMKL */
                plugins[idx++] = "clblast";  /* Fallback: OpenCL */
                break;
            case FB_GPU_VENDOR_APPLE:
                plugins[idx++] = "metal";    /* Primary: Metal */
                plugins[idx++] = "clblast";  /* Fallback: OpenCL */
                break;
            default:
                plugins[idx++] = "clblast";
                plugins[idx++] = "clblas";
                break;
        }
    }
    
    /* CPU devices */
    else if (device->properties.type == FB_DEVICE_TYPE_CPU) {
        switch (device->properties.cpu.vendor) {
            case FB_CPU_VENDOR_AMD:
                plugins[idx++] = "aocl-blis";
                plugins[idx++] = "openblas";
                break;
            case FB_CPU_VENDOR_INTEL:
                plugins[idx++] = "mkl";
                plugins[idx++] = "openblas";
                break;
            case FB_CPU_VENDOR_ARM:
                #ifdef __APPLE__
                plugins[idx++] = "accelerate";
                #endif
                plugins[idx++] = "openblas";
                break;
            default:
                plugins[idx++] = "openblas";
                plugins[idx++] = "standard-blis";
                break;
        }
    }
    
    *count = idx;
    return plugins;
}

/* ============================================================================
 * Backend Instance Management
 * ========================================================================== */

static int initialize_backend_instance(fb_backend_instance_t* instance,
                                       fb_compute_device_t* device,
                                       const char* plugin_name) {
    if (!instance || !device) {
        return -1;
    }
    
    memset(instance, 0, sizeof(fb_backend_instance_t));
    instance->device = device;
    
    /* Load the plugin */
    printf("[INFO] Loading backend for device %u (%s)\n", 
           device->properties.device_id, device->properties.name);
    
    const fb_backend_plugin_t* plugin = fb_load_best_plugin(
        plugin_name,  /* Can be NULL for auto-selection */
        NULL,         /* search_paths */
        &instance->plugin_ctx
    );
    
    if (!plugin) {
        fprintf(stderr, "[ERROR] Failed to load plugin for device %u\n", device->properties.device_id);
        
        /* Don't fall back to reference backend here - let caller try next plugin */
        return -1;
    }
    
    instance->plugin = plugin;
    
    /* Get the vtable from the plugin */
    fprintf(stderr, "[DEBUG] About to call plugin->get_vtable(%p)\n", instance->plugin_ctx);
    instance->vtable = plugin->get_vtable(instance->plugin_ctx);
    fprintf(stderr, "[DEBUG] plugin->get_vtable returned %p\n", (void*)instance->vtable);
    if (!instance->vtable) {
        fprintf(stderr, "[ERROR] Plugin returned NULL vtable\n");
        if (plugin->shutdown) {
            fprintf(stderr, "[DEBUG] Calling plugin->shutdown(%p)\n", instance->plugin_ctx);
            plugin->shutdown(instance->plugin_ctx);
            fprintf(stderr, "[DEBUG] plugin->shutdown returned\n");
        }
        return -1;
    }
    
    instance->initialized = true;
    instance->operation_count = 0;
    instance->total_time_seconds = 0.0;
    
    printf("[INFO] Successfully loaded backend: %s (version %s)\n",
           plugin->metadata->name, plugin->metadata->version);
    
    fprintf(stderr, "[DEBUG] Returning 0 from initialize_backend_instance\n");
    return 0;
}

fb_backend_instance_t* fb_get_backend_for_device(fb_compute_device_t* device) {
    if (!device) {
        fprintf(stderr, "[ERROR] NULL device passed to fb_get_backend_for_device\n");
        return NULL;
    }
    
    if (!g_instance_manager.initialized) {
        if (fb_backend_instance_manager_init() != 0) {
            return NULL;
        }
    }
    
    /* Check if we already have a backend instance for this device */
    for (uint32_t i = 0; i < g_instance_manager.num_instances; i++) {
        fb_backend_instance_t* inst = &g_instance_manager.instances[i];
        if (inst->device == device && inst->initialized) {
            return inst; /* Cache hit */
        }
    }
    
    /* Need to create a new instance */
    if (g_instance_manager.num_instances >= g_instance_manager.capacity) {
        /* Grow array */
        uint32_t new_capacity = g_instance_manager.capacity * 2;
        fb_backend_instance_t* new_instances = (fb_backend_instance_t*)realloc(
            g_instance_manager.instances,
            new_capacity * sizeof(fb_backend_instance_t)
        );
        
        if (!new_instances) {
            fprintf(stderr, "[ERROR] Failed to grow instance array\n");
            return NULL;
        }
        
        g_instance_manager.instances = new_instances;
        g_instance_manager.capacity = new_capacity;
    }
    
    /* Select plugins for this device */
    int plugin_count = 0;
    const char** plugin_names = select_plugins_for_device(device, &plugin_count);
    
    /* Try to load backends in priority order */
    for (int i = 0; i < plugin_count; i++) {
        /* Create new instance */
        if (g_instance_manager.num_instances >= g_instance_manager.capacity) {
            /* Grow array */
            uint32_t new_capacity = g_instance_manager.capacity * 2;
            fb_backend_instance_t* new_instances = (fb_backend_instance_t*)realloc(
                g_instance_manager.instances,
                new_capacity * sizeof(fb_backend_instance_t)
            );
            
            if (!new_instances) {
                fprintf(stderr, "[ERROR] Failed to grow instance array\n");
                continue; /* Try next plugin */
            }
            
            g_instance_manager.instances = new_instances;
            g_instance_manager.capacity = new_capacity;
        }
        
        fb_backend_instance_t* instance = &g_instance_manager.instances[g_instance_manager.num_instances];
        fprintf(stderr, "[DEBUG] Calling initialize_backend_instance for plugin '%s'\n", plugin_names[i]);
        if (initialize_backend_instance(instance, device, plugin_names[i]) == 0) {
            fprintf(stderr, "[DEBUG] initialize_backend_instance succeeded, incrementing num_instances from %u to %u\n",
                   g_instance_manager.num_instances, g_instance_manager.num_instances + 1);
            g_instance_manager.num_instances++;
            fprintf(stderr, "[DEBUG] Successfully incremented num_instances\n");
            /* Successfully loaded one backend - keep trying others */
        } else {
            fprintf(stderr, "[DEBUG] initialize_backend_instance failed for plugin '%s'\n", plugin_names[i]);
        }
    }
    
    fprintf(stderr, "[DEBUG] Finished plugin loop, searching for best instance for device %p\n", device);
    /* Return the first successfully loaded instance (highest priority) */
    for (uint32_t i = 0; i < g_instance_manager.num_instances; i++) {
        fprintf(stderr, "[DEBUG] Checking instance %u (device=%p, initialized=%d)\n", 
               i, g_instance_manager.instances[i].device, g_instance_manager.instances[i].initialized);
        fb_backend_instance_t* inst = &g_instance_manager.instances[i];
        if (inst->device == device && inst->initialized) {
            fprintf(stderr, "[DEBUG] Found matching instance at index %u\n", i);
            return inst;
        }
    }
    
    fprintf(stderr, "[DEBUG] No matching instance found, returning NULL\n");
    return NULL; /* No backends loaded */
}

fb_backend_instance_t* fb_load_backend_for_device(fb_compute_device_t* device,
                                                   const char* plugin_name) {
    if (!device) {
        return NULL;
    }
    
    /* Check if already loaded */
    fb_backend_instance_t* existing = fb_get_backend_for_device(device);
    if (existing) {
        /* Already loaded - if wrong plugin, unload first */
        if (plugin_name && existing->plugin &&
            strcmp(existing->plugin->metadata->name, plugin_name) != 0) {
            fb_unload_backend_for_device(device);
        } else {
            return existing; /* Correct plugin already loaded */
        }
    }
    
    /* Load with specific plugin */
    if (!g_instance_manager.initialized) {
        if (fb_backend_instance_manager_init() != 0) {
            return NULL;
        }
    }
    
    if (g_instance_manager.num_instances >= g_instance_manager.capacity) {
        uint32_t new_capacity = g_instance_manager.capacity * 2;
        fb_backend_instance_t* new_instances = (fb_backend_instance_t*)realloc(
            g_instance_manager.instances,
            new_capacity * sizeof(fb_backend_instance_t)
        );
        if (!new_instances) return NULL;
        g_instance_manager.instances = new_instances;
        g_instance_manager.capacity = new_capacity;
    }
    
    fb_backend_instance_t* instance = &g_instance_manager.instances[g_instance_manager.num_instances];
    if (initialize_backend_instance(instance, device, plugin_name) != 0) {
        return NULL;
    }
    
    g_instance_manager.num_instances++;
    return instance;
}

void fb_unload_backend_for_device(fb_compute_device_t* device) {
    if (!device || !g_instance_manager.initialized) {
        return;
    }
    
    /* Find and remove instance */
    for (uint32_t i = 0; i < g_instance_manager.num_instances; i++) {
        fb_backend_instance_t* inst = &g_instance_manager.instances[i];
        if (inst->device == device && inst->initialized) {
            /* Shutdown the backend */
            if (inst->plugin && inst->plugin->shutdown) {
                inst->plugin->shutdown(inst->plugin_ctx);
            }
            
            /* Remove from array (swap with last element) */
            if (i < g_instance_manager.num_instances - 1) {
                g_instance_manager.instances[i] = 
                    g_instance_manager.instances[g_instance_manager.num_instances - 1];
            }
            g_instance_manager.num_instances--;
            
            printf("[INFO] Unloaded backend for device %u\n", device->properties.device_id);
            return;
        }
    }
}

/* ============================================================================
 * Vtable Access
 * ========================================================================== */

/* Forward declarations for backend-specific active instance setters */
extern void clblast_set_active_instance(const fb_backend_vtable_t* vtable);

const fb_backend_vtable_t* fb_backend_get_vtable(fb_backend_instance_t* instance) {
    fprintf(stderr, "[DEBUG] get_vtable called with instance=%p\n", instance);
    if (!instance || !instance->initialized) {
        fprintf(stderr, "[DEBUG] get_vtable: NULL or uninitialized\n");
        return NULL;
    }
    
    fprintf(stderr, "[DEBUG] get_vtable: instance OK, plugin=%p\n", instance->plugin);
    /* For CLBlast backend, set the active instance in thread-local storage */
    if (instance->vtable && instance->plugin) {
        const char* plugin_name = instance->plugin->metadata ? instance->plugin->metadata->name : "";
        fprintf(stderr, "[DEBUG] get_vtable: plugin_name=%s\n", plugin_name ? plugin_name : "NULL");
        if (plugin_name && strstr(plugin_name, "clblast")) {
            fprintf(stderr, "[DEBUG] get_vtable: About to call clblast_set_active_instance\n");
            clblast_set_active_instance(instance->vtable);
            fprintf(stderr, "[DEBUG] get_vtable: Returned from clblast_set_active_instance\n");
        }
        /* TODO: Add similar calls for cuBLAS, rocBLAS, etc. when they're refactored */
    }
    
    fprintf(stderr, "[DEBUG] get_vtable: Returning vtable=%p\n", instance->vtable);
    return instance->vtable;
}

/* ============================================================================
 * Fallback Execution
 * ========================================================================== */

int fb_execute_with_fallback(fb_backend_instance_t* instance,
                             const char* operation_name,
                             fb_operation_exec_func_t exec_func,
                             void* user_data) {
    if (!instance || !exec_func) {
        return -1;
    }
    
    /* Try primary backend */
    if (instance->vtable) {
        int result = exec_func(instance->vtable, user_data);
        if (result == 0) {
            instance->operation_count++;
            return 0; /* Success */
        }
        
        fprintf(stderr, "[WARN] Operation '%s' failed on primary backend\n", 
               operation_name ? operation_name : "unknown");
    }
    
    /* Fallback to reference backend */
    const fb_backend_vtable_t* reference = fb_reference_get_vtable();
    if (reference) {
        fprintf(stderr, "[INFO] Falling back to reference backend for '%s'\n",
               operation_name ? operation_name : "unknown");
        
        int result = exec_func(reference, user_data);
        if (result == 0) {
            instance->operation_count++;
            return 0;
        }
    }
    
    fprintf(stderr, "[ERROR] Operation '%s' failed on all backends\n",
           operation_name ? operation_name : "unknown");
    return -1;
}

/* ============================================================================
 * Statistics
 * ========================================================================== */

int fb_backend_get_stats(fb_backend_instance_t* instance,
                        uint64_t* out_operation_count,
                        double* out_total_time) {
    if (!instance || !instance->initialized) {
        return -1;
    }
    
    if (out_operation_count) {
        *out_operation_count = instance->operation_count;
    }
    if (out_total_time) {
        *out_total_time = instance->total_time_seconds;
    }
    
    return 0;
}

void fb_backend_reset_stats(fb_backend_instance_t* instance) {
    if (instance && instance->initialized) {
        instance->operation_count = 0;
        instance->total_time_seconds = 0.0;
    }
}

/* ============================================================================
 * Utilities
 * ========================================================================== */

void fb_backend_print_instance(const fb_backend_instance_t* instance) {
    if (!instance || !instance->initialized) {
        printf("  [UNINITIALIZED]\n");
        return;
    }
    
    printf("  Device: %u - %s\n", instance->device->properties.device_id, instance->device->properties.name);
    
    if (instance->plugin && instance->plugin->metadata) {
        printf("  Backend: %s (version %s, vendor: %s)\n",
               instance->plugin->metadata->name,
               instance->plugin->metadata->version,
               instance->plugin->metadata->vendor);
    } else {
        printf("  Backend: Reference (fallback)\n");
    }
    
    printf("  Operations executed: %llu\n", (unsigned long long)instance->operation_count);
    printf("  Total time: %.3f seconds\n", instance->total_time_seconds);
}

int fb_backend_list_instances(fb_backend_instance_t** instances, int max_count) {
    if (!g_instance_manager.initialized) {
        return 0;
    }
    
    int count = (int)g_instance_manager.num_instances;
    
    if (instances && max_count > 0) {
        for (int i = 0; i < count && i < max_count; i++) {
            instances[i] = &g_instance_manager.instances[i];
        }
    }
    
    return count;
}
