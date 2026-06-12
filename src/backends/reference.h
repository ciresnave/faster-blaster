/**
 * @file reference.h
 * @brief Reference backend API
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_REFERENCE_H
#define FASTER_BLASTER_REFERENCE_H

#include "backend_interface.h"
#include "faster-blaster/backend_plugin.h"  /* fb_lib_handle_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the reference backend vtable.
 * Always returns a valid (non-NULL) pointer.  Fields are populated lazily on
 * the first call; fb_reference_init() supplements them with DLL export data.
 */
const fb_backend_vtable_t *fb_reference_backend(void);

/**
 * Scan the loaded reference DLL for additional cblas_* exports and fill any
 * ext_ops slots not already wired by the static named-field block.
 *
 * Call once after loading faster_blaster_reference.dll/.so via
 * fb_plugin_load_library().  Safe to call with lib_handle == NULL (no-op).
 */
void fb_reference_init(fb_lib_handle_t lib_handle);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_REFERENCE_H */
