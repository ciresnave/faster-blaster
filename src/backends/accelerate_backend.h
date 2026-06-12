/**
 * @file accelerate_backend.h
 * @brief Intel accelerate backend adapter
 * 
 * Provides wrapper layer between faster-blaster and Intel accelerate library.
 * accelerate is proprietary but allows redistribution with restrictions.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_accelerate_BACKEND_H
#define FASTER_BLASTER_accelerate_BACKEND_H

#include "backend_interface.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get accelerate backend vtable
 * 
 * @return Backend vtable for accelerate operations
 */
const fb_backend_vtable_t* fb_accelerate_get_vtable(void);

/**
 * Initialize accelerate backend
 * 
 * @return 0 on success, negative on error
 */
int fb_accelerate_init(void);

/**
 * Shutdown accelerate backend
 */
void fb_accelerate_shutdown(void);

/**
 * Check if accelerate is available on this system
 * 
 * @return true if accelerate library can be loaded
 */
bool fb_accelerate_is_available(void);

/**
 * Get accelerate version information
 * 
 * @param major Output for major version
 * @param minor Output for minor version
 * @param update Output for update version
 * @return 0 on success
 */
int fb_accelerate_get_version(int* major, int* minor, int* update);

/**
 * Set number of threads for accelerate operations
 * 
 * @param num_threads Number of threads (0 for automatic)
 */
void fb_accelerate_set_num_threads(int num_threads);

/**
 * Get number of threads currently used by accelerate
 * 
 * @return Number of threads
 */
int fb_accelerate_get_num_threads(void);

/**
 * Set accelerate threading layer
 * 
 * @param layer Threading layer: "intel" (default), "tbb", "gnu", "sequential"
 * @return 0 on success
 */
int fb_accelerate_set_threading_layer(const char* layer);

/**
 * Enable/disable accelerate verbose mode for debugging
 * 
 * @param enable true to enable verbose output
 */
void fb_accelerate_set_verbose(bool enable);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_accelerate_BACKEND_H */
