/**
 * @file device_api.c
 * @brief Unified device API implementation
 * 
 * Provides a uniform interface for memory management, stream control,
 * and backend properties by delegating to the backend vtable.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/device_api.h"
#include "faster-blaster/backend_instance.h"
#include "faster-blaster/compute_device.h"
#include "backends/backend_interface.h"
#include <stdio.h>

/**
 * @brief Get backend instance from compute device
 * 
 * Internal helper to retrieve the backend instance associated with a device.
 * 
 * @param device Compute device handle
 * @return Backend instance, or NULL if not available
 */
fb_backend_instance_t* fb_device_get_backend_instance(fb_compute_device_t* device) {
    if (!device) {
        return NULL;
    }
    return fb_get_backend_for_device(device);
}

/* Forward declaration for proper vtable access (sets up active instance for GPU backends) */
extern const fb_backend_vtable_t* fb_backend_get_vtable(fb_backend_instance_t* instance);

/**
 * @brief Get the backend vtable, setting up active instance context for GPU backends
 * 
 * This helper MUST be used instead of instance->vtable directly for operations that
 * need the active GPU instance set (e.g., CLBlast memory/stream ops).
 */
static const fb_backend_vtable_t* get_vtable_with_active(fb_backend_instance_t* instance) {
    return fb_backend_get_vtable(instance);
}

/**
 * @brief Allocate device memory
 */
int fb_device_alloc(fb_compute_device_t* device, void** ptr, size_t size) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return -1;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return -1;
    
    if (vtable->mem_alloc) {
        return vtable->mem_alloc(instance->plugin_ctx, ptr, size);
    }
    
    /* CPU backends may not implement mem_alloc (uses system malloc) */
    return 0;
}

/**
 * @brief Free device memory
 */
void fb_device_free(fb_compute_device_t* device, void* ptr) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return;
    
    if (vtable->mem_free) {
        vtable->mem_free(instance->plugin_ctx, ptr);
    }
}

/**
 * @brief Upload data to device (host → device)
 */
int fb_device_upload(fb_compute_device_t* device, void* dst, const void* src, size_t size) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return -1;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return -1;
    
    if (vtable->mem_upload) {
        return vtable->mem_upload(instance->plugin_ctx, dst, src, size);
    }
    
    /* CPU backends: no-op (data already accessible) */
    return 0;
}

/**
 * @brief Download data from device (device → host)
 */
int fb_device_download(fb_compute_device_t* device, void* dst, const void* src, size_t size) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return -1;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return -1;
    
    if (vtable->mem_download) {
        return vtable->mem_download(instance->plugin_ctx, dst, src, size);
    }
    
    /* CPU backends: no-op (data already accessible) */
    return 0;
}

/**
 * @brief Copy data on device (device → device)
 */
int fb_device_copy(fb_compute_device_t* device, void* dst, const void* src, size_t size) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return -1;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return -1;
    
    if (vtable->mem_copy) {
        return vtable->mem_copy(instance->plugin_ctx, dst, src, size);
    }
    
    /* CPU backends: no-op or could use memcpy */
    return 0;
}

/**
 * @brief Create a new stream for async operations
 */
int fb_device_stream_create(fb_compute_device_t* device, void** stream) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return -1;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return -1;
    
    if (vtable->stream_create) {
        return vtable->stream_create(instance->plugin_ctx, stream);
    }
    
    /* CPU backends: no streams (operations are synchronous) */
    if (stream) {
        *stream = NULL;
    }
    return 0;
}

/**
 * @brief Destroy a stream
 */
void fb_device_stream_destroy(fb_compute_device_t* device, void* stream) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return;
    
    if (vtable->stream_destroy) {
        vtable->stream_destroy(instance->plugin_ctx, stream);
    }
}

/**
 * @brief Synchronize all pending operations on a device or stream
 */
int fb_device_sync(fb_compute_device_t* device, void* stream) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return -1;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return -1;
    
    if (vtable->stream_sync) {
        return vtable->stream_sync(instance->plugin_ctx, stream);
    }
    
    /* CPU backends: already synchronous, nothing to do */
    return 0;
}

/**
 * @brief Set the active stream for subsequent operations
 */
void fb_device_stream_set(fb_compute_device_t* device, void* stream) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return;
    
    if (vtable->stream_set) {
        vtable->stream_set(instance->plugin_ctx, stream);
    }
}

/**
 * @brief Get backend capabilities
 */
uint32_t fb_device_get_capabilities(fb_compute_device_t* device) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return 0;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return 0;
    
    if (vtable->get_capabilities) {
        return vtable->get_capabilities(instance->plugin_ctx);
    }
    
    return 0;
}

/**
 * @brief Get number of threads
 */
int fb_device_get_num_threads(fb_compute_device_t* device) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return -1;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return -1;
    
    if (vtable->get_num_threads) {
        return vtable->get_num_threads(instance->plugin_ctx);
    }
    
    /* GPU backends may not have thread control */
    return 1;
}

/**
 * @brief Set number of threads
 */
void fb_device_set_num_threads(fb_compute_device_t* device, int num_threads) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance) return;
    const fb_backend_vtable_t* vtable = get_vtable_with_active(instance);
    if (!vtable) return;
    
    if (vtable->set_num_threads) {
        vtable->set_num_threads(instance->plugin_ctx, num_threads);
    }
}
