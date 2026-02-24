/**
 * @file atlas_backend.h
 * @brief Intel atlas backend adapter
 * 
 * Provides wrapper layer between faster-blaster and Intel atlas library.
 * atlas is proprietary but allows redistribution with restrictions.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_atlas_BACKEND_H
#define FASTER_BLASTER_atlas_BACKEND_H

#include "backend_interface.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get atlas backend vtable
 * 
 * @return Backend vtable for atlas operations
 */
const fb_backend_vtable_t* fb_atlas_get_vtable(void);

/**
 * Initialize atlas backend
 * 
 * @return 0 on success, negative on error
 */
int fb_atlas_init(void);

/**
 * Shutdown atlas backend
 */
void fb_atlas_shutdown(void);

/**
 * Check if atlas is available on this system
 * 
 * @return true if atlas library can be loaded
 */
bool fb_atlas_is_available(void);

/**
 * Get atlas version information
 * 
 * @param major Output for major version
 * @param minor Output for minor version
 * @param update Output for update version
 * @return 0 on success
 */
int fb_atlas_get_version(int* major, int* minor, int* update);

/**
 * Set number of threads for atlas operations
 * 
 * @param num_threads Number of threads (0 for automatic)
 */
void fb_atlas_set_num_threads(int num_threads);

/**
 * Get number of threads currently used by atlas
 * 
 * @return Number of threads
 */
int fb_atlas_get_num_threads(void);

/**
 * Set atlas threading layer
 * 
 * @param layer Threading layer: "intel" (default), "tbb", "gnu", "sequential"
 * @return 0 on success
 */
int fb_atlas_set_threading_layer(const char* layer);

/**
 * Enable/disable atlas verbose mode for debugging
 * 
 * @param enable true to enable verbose output
 */
void fb_atlas_set_verbose(bool enable);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_atlas_BACKEND_H */
