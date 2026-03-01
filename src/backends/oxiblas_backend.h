/**
 * @file oxiblas_backend.h
 * @brief OxiBLAS backend adapter (test / legacy-compatibility API)
 *
 * Thin runtime-loadable adapter for the OxiBLAS FFI library.  This header
 * exposes the minimal API needed by the standalone test suite
 * (test_oxiblas_backend.c).  For production dispatch, use the plugin
 * interface in src/plugins/plugin_oxiblas.c.
 *
 * OxiBLAS exports Fortran-convention symbols only (saxpy_, dgemm_, …).
 * This adapter wraps them with CBLAS-style call signatures.
 *
 * @copyright Copyright (c) 2025
 * @license   MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_OXIBLAS_BACKEND_H
#define FASTER_BLASTER_OXIBLAS_BACKEND_H

#include "backend_interface.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Check whether the OxiBLAS FFI shared library is available at runtime.
 * Probes for "saxpy_" and "dgemm_" Fortran symbols.
 *
 * @return true if liboxiblas_ffi can be loaded and exports the expected symbols
 */
bool fb_oxiblas_is_available(void);

/**
 * Load and initialise the OxiBLAS backend.
 *
 * @return 0 on success, negative on error
 */
int fb_oxiblas_init(void);

/**
 * Shut down the OxiBLAS backend and release the library handle.
 */
void fb_oxiblas_shutdown(void);

/**
 * Return the OxiBLAS backend vtable.
 * The vtable is populated after a successful call to fb_oxiblas_init().
 *
 * @return Pointer to vtable, or NULL if not initialised
 */
const fb_backend_vtable_t *fb_oxiblas_get_vtable(void);

/**
 * Return the OxiBLAS library version string.
 * OxiBLAS 0.1.0 does not export a version query symbol; this returns the
 * compile-time known version "0.1.0".
 *
 * @return Version string, e.g. "0.1.0"
 */
const char *fb_oxiblas_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_OXIBLAS_BACKEND_H */
