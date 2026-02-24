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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the reference backend vtable
 * 
 * @return Pointer to reference backend vtable
 */
const fb_backend_vtable_t *fb_reference_backend(void);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_REFERENCE_H */
