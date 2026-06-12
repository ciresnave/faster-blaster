/**
 * @file mkl_backend.h
 * @brief Intel MKL backend adapter
 * 
 * Provides wrapper layer between faster-blaster and Intel MKL library.
 * MKL is proprietary but allows redistribution with restrictions.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_MKL_BACKEND_H
#define FASTER_BLASTER_MKL_BACKEND_H

#include "backend_interface.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get MKL backend vtable
 * 
 * @return Backend vtable for MKL operations
 */
const fb_backend_vtable_t* fb_mkl_get_vtable(void);

/**
 * Initialize MKL backend
 * 
 * @return 0 on success, negative on error
 */
int fb_mkl_init(void);

/**
 * Shutdown MKL backend
 */
void fb_mkl_shutdown(void);

/**
 * Check if MKL is available on this system
 * 
 * @return true if MKL library can be loaded
 */
bool fb_mkl_is_available(void);

/**
 * Get MKL version information
 * 
 * @param major Output for major version
 * @param minor Output for minor version
 * @param update Output for update version
 * @return 0 on success
 */
int fb_mkl_get_version(int* major, int* minor, int* update);

/**
 * Set number of threads for MKL operations
 * 
 * @param num_threads Number of threads (0 for automatic)
 */
void fb_mkl_set_num_threads(int num_threads);

/**
 * Get number of threads currently used by MKL
 * 
 * @return Number of threads
 */
int fb_mkl_get_num_threads(void);

/**
 * Set MKL threading layer
 * 
 * @param layer Threading layer: "intel" (default), "tbb", "gnu", "sequential"
 * @return 0 on success
 */
int fb_mkl_set_threading_layer(const char* layer);

/**
 * Enable/disable MKL verbose mode for debugging
 * 
 * @param enable true to enable verbose output
 */
void fb_mkl_set_verbose(bool enable);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_MKL_BACKEND_H */
