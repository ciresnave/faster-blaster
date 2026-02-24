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

/**
 * @brief Allocate device memory
 */
int fb_device_alloc(fb_compute_device_t* device, void** ptr, size_t size) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return -1;
    }
    
    if (instance->vtable->mem_alloc) {
        return instance->vtable->mem_alloc(instance->plugin_ctx, ptr, size);
    }
    
    /* CPU backends may not implement mem_alloc (uses system malloc) */
    return 0;
}

/**
 * @brief Free device memory
 */
void fb_device_free(fb_compute_device_t* device, void* ptr) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return;
    }
    
    if (instance->vtable->mem_free) {
        instance->vtable->mem_free(instance->plugin_ctx, ptr);
    }
}

/**
 * @brief Upload data to device (host → device)
 */
int fb_device_upload(fb_compute_device_t* device, void* dst, const void* src, size_t size) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return -1;
    }
    
    if (instance->vtable->mem_upload) {
        return instance->vtable->mem_upload(instance->plugin_ctx, dst, src, size);
    }
    
    /* CPU backends: no-op (data already accessible) */
    return 0;
}

/**
 * @brief Download data from device (device → host)
 */
int fb_device_download(fb_compute_device_t* device, void* dst, const void* src, size_t size) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return -1;
    }
    
    if (instance->vtable->mem_download) {
        return instance->vtable->mem_download(instance->plugin_ctx, dst, src, size);
    }
    
    /* CPU backends: no-op (data already accessible) */
    return 0;
}

/**
 * @brief Copy data on device (device → device)
 */
int fb_device_copy(fb_compute_device_t* device, void* dst, const void* src, size_t size) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return -1;
    }
    
    if (instance->vtable->mem_copy) {
        return instance->vtable->mem_copy(instance->plugin_ctx, dst, src, size);
    }
    
    /* CPU backends: no-op or could use memcpy */
    return 0;
}

/**
 * @brief Create a new stream for async operations
 */
int fb_device_stream_create(fb_compute_device_t* device, void** stream) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return -1;
    }
    
    if (instance->vtable->stream_create) {
        return instance->vtable->stream_create(instance->plugin_ctx, stream);
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
    if (!instance || !instance->vtable) {
        return;
    }
    
    if (instance->vtable->stream_destroy) {
        instance->vtable->stream_destroy(instance->plugin_ctx, stream);
    }
}

/**
 * @brief Synchronize all pending operations on a device or stream
 */
int fb_device_sync(fb_compute_device_t* device, void* stream) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return -1;
    }
    
    if (instance->vtable->stream_sync) {
        return instance->vtable->stream_sync(instance->plugin_ctx, stream);
    }
    
    /* CPU backends: already synchronous, nothing to do */
    return 0;
}

/**
 * @brief Set the active stream for subsequent operations
 */
void fb_device_stream_set(fb_compute_device_t* device, void* stream) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return;
    }
    
    if (instance->vtable->stream_set) {
        instance->vtable->stream_set(instance->plugin_ctx, stream);
    }
}

/**
 * @brief Get backend capabilities
 */
uint32_t fb_device_get_capabilities(fb_compute_device_t* device) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return 0;
    }
    
    if (instance->vtable->get_capabilities) {
        return instance->vtable->get_capabilities(instance->plugin_ctx);
    }
    
    return 0;
}

/**
 * @brief Get number of threads
 */
int fb_device_get_num_threads(fb_compute_device_t* device) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return -1;
    }
    
    if (instance->vtable->get_num_threads) {
        return instance->vtable->get_num_threads(instance->plugin_ctx);
    }
    
    /* GPU backends may not have thread control */
    return 1;
}

/**
 * @brief Set number of threads
 */
void fb_device_set_num_threads(fb_compute_device_t* device, int num_threads) {
    fb_backend_instance_t* instance = fb_device_get_backend_instance(device);
    if (!instance || !instance->vtable) {
        return;
    }
    
    if (instance->vtable->set_num_threads) {
        instance->vtable->set_num_threads(instance->plugin_ctx, num_threads);
    }
}
