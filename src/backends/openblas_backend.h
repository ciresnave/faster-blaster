/**
 * @file openblas_backend.h
 * @brief OpenBLAS backend adapter
 * 
 * Provides wrapper layer between faster-blaster and OpenBLAS library.
 * OpenBLAS uses BSD-3-Clause license and can be bundled.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_OPENBLAS_BACKEND_H
#define FASTER_BLASTER_OPENBLAS_BACKEND_H

#include "backend_interface.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get OpenBLAS backend vtable
 * 
 * @return Backend vtable for OpenBLAS operations
 */
const fb_backend_vtable_t* fb_openblas_get_vtable(void);

/**
 * Initialize OpenBLAS backend
 * 
 * @return 0 on success, negative on error
 */
int fb_openblas_init(void);

/**
 * Shutdown OpenBLAS backend
 */
void fb_openblas_shutdown(void);

/**
 * Check if OpenBLAS is available on this system
 * 
 * @return true if OpenBLAS library can be loaded
 */
bool fb_openblas_is_available(void);

/**
 * Get OpenBLAS version string
 * 
 * @return Version string (e.g., "0.3.21"), or NULL if not available
 */
const char* fb_openblas_get_version(void);

/**
 * Set number of threads for OpenBLAS operations
 * 
 * @param num_threads Number of threads (0 for automatic)
 */
void fb_openblas_set_num_threads(int num_threads);

/**
 * Get number of threads currently used by OpenBLAS
 * 
 * @return Number of threads
 */
int fb_openblas_get_num_threads(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_OPENBLAS_BACKEND_H */
