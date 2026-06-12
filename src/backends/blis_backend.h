/**
 * @file blis_backend.h
 * @brief Intel blis backend adapter
 * 
 * Provides wrapper layer between faster-blaster and Intel blis library.
 * blis is proprietary but allows redistribution with restrictions.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_blis_BACKEND_H
#define FASTER_BLASTER_blis_BACKEND_H

#include "backend_interface.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get blis backend vtable
 * 
 * @return Backend vtable for blis operations
 */
const fb_backend_vtable_t* fb_blis_get_vtable(void);

/**
 * Initialize blis backend
 * 
 * @return 0 on success, negative on error
 */
int fb_blis_init(void);

/**
 * Shutdown blis backend
 */
void fb_blis_shutdown(void);

/**
 * Check if blis is available on this system
 * 
 * @return true if blis library can be loaded
 */
bool fb_blis_is_available(void);

/**
 * Get blis version information
 * 
 * @param major Output for major version
 * @param minor Output for minor version
 * @param update Output for update version
 * @return 0 on success
 */
int fb_blis_get_version(int* major, int* minor, int* update);

/**
 * Set number of threads for blis operations
 * 
 * @param num_threads Number of threads (0 for automatic)
 */
void fb_blis_set_num_threads(int num_threads);

/**
 * Get number of threads currently used by blis
 * 
 * @return Number of threads
 */
int fb_blis_get_num_threads(void);

/**
 * Set blis threading layer
 * 
 * @param layer Threading layer: "intel" (default), "tbb", "gnu", "sequential"
 * @return 0 on success
 */
int fb_blis_set_threading_layer(const char* layer);

/**
 * Enable/disable blis verbose mode for debugging
 * 
 * @param enable true to enable verbose output
 */
void fb_blis_set_verbose(bool enable);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_blis_BACKEND_H */
