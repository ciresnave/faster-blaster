/**
 * @file aocl_backend.h
 * @brief Intel aocl backend adapter
 * 
 * Provides wrapper layer between faster-blaster and Intel aocl library.
 * aocl is proprietary but allows redistribution with restrictions.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_aocl_BACKEND_H
#define FASTER_BLASTER_aocl_BACKEND_H

#include "backend_interface.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get aocl backend vtable
 * 
 * @return Backend vtable for aocl operations
 */
const fb_backend_vtable_t* fb_aocl_get_vtable(void);

/**
 * Initialize aocl backend
 * 
 * @return 0 on success, negative on error
 */
int fb_aocl_init(void);

/**
 * Shutdown aocl backend
 */
void fb_aocl_shutdown(void);

/**
 * Check if aocl is available on this system
 * 
 * @return true if aocl library can be loaded
 */
bool fb_aocl_is_available(void);

/**
 * Get aocl version information
 * 
 * @param major Output for major version
 * @param minor Output for minor version
 * @param update Output for update version
 * @return 0 on success
 */
int fb_aocl_get_version(int* major, int* minor, int* update);

/**
 * Set number of threads for aocl operations
 * 
 * @param num_threads Number of threads (0 for automatic)
 */
void fb_aocl_set_num_threads(int num_threads);

/**
 * Get number of threads currently used by aocl
 * 
 * @return Number of threads
 */
int fb_aocl_get_num_threads(void);

/**
 * Set aocl threading layer
 * 
 * @param layer Threading layer: "intel" (default), "tbb", "gnu", "sequential"
 * @return 0 on success
 */
int fb_aocl_set_threading_layer(const char* layer);

/**
 * Enable/disable aocl verbose mode for debugging
 * 
 * @param enable true to enable verbose output
 */
void fb_aocl_set_verbose(bool enable);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_aocl_BACKEND_H */
