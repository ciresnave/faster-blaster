/**
 * @file faster_blaster_reference_backend.h
 * @brief Integration wrapper for faster-blaster-reference DLL
 *
 * Provides a backend interface to the complete, authoritative BLAS/LAPACK
 * reference implementation (1248 operations across all dtypes). This backend
 * serves as the "ground truth" for correctness validation of all other
 * backends.
 *
 * The reference implementation is loaded from faster_blaster_reference.dll and
 * wrapped to conform to the faster-blaster backend vtable interface.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_FASTER_BLASTER_REFERENCE_BACKEND_H
#define FASTER_BLASTER_FASTER_BLASTER_REFERENCE_BACKEND_H

#include "backend_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the reference backend
 *
 * Loads faster_blaster_reference.dll (or .so/.dylib on Unix) and populates
 * the vtable with all 1248 BLAS/LAPACK operations.
 *
 * @return 0 on success, non-zero on failure (DLL not found, etc.)
 */
int fb_blr_init(void);

/**
 * @brief Finalize the reference backend
 *
 * Unloads the DLL and cleans up resources.
 */
void fb_blr_finalize(void);

/**
 * @brief Get the reference backend vtable
 *
 * @return Pointer to fully populated fb_backend_vtable_t with all 1248
 * operations
 */
const fb_backend_vtable_t *fb_blr_get_vtable(void);

/**
 * @brief Check if the reference backend is available
 *
 * @return true if DLL is loaded and all critical functions are available
 */
bool fb_blr_is_available(void);

/**
 * @brief Get the path to faster_blaster_reference.dll
 *
 * Searches:
 *   1. FB_REFERENCE_BACKEND_PATH environment variable
 *   2. Same directory as faster-blaster DLL
 *   3. System PATH
 *
 * @return String with DLL path, or NULL if not found
 */
const char *fb_blr_get_dll_path(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_FASTER_BLASTER_REFERENCE_BACKEND_H */
