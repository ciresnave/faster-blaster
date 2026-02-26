/**
 * @file fb_types.h
 * @brief Canonical complex scalar and status types for faster-blaster
 *
 * Single authoritative definition of:
 *   - fb_complex_float_t  (float _Complex)
 *   - fb_complex_double_t (double _Complex)
 *   - fb_status_t         (operation result code)
 *
 * Requires Clang or GCC.  MSVC (cl.exe) is not supported; use clang or
 * clang++ exclusively.  Do NOT include <complex.h> anywhere in this project.
 *
 * Both backend_interface.h and vtable_autofill.h include this header so that
 * these types are defined exactly once regardless of include order.
 *
 * @copyright Copyright (c) 2025-2026
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_FB_TYPES_H
#define FASTER_BLASTER_FB_TYPES_H

/* Complex scalar types using C99/C11 _Complex keyword.
 * Clang (and GCC) fully support _Complex on all platforms including Windows.
 * Never use "float complex" (requires <complex.h>); use "float _Complex"
 * directly — _Complex is a keyword, not a macro. */
typedef float  _Complex fb_complex_float_t;
typedef double _Complex fb_complex_double_t;

/* Operation status codes */
typedef enum {
    FB_STATUS_SUCCESS          =  0,
    FB_STATUS_ERROR            = -1,
    FB_STATUS_NOT_SUPPORTED    = -2,
    FB_STATUS_INVALID_ARGUMENT = -3,
    FB_STATUS_OUT_OF_MEMORY    = -4,
    FB_STATUS_OVERFLOW         = -5
} fb_status_t;

#endif /* FASTER_BLASTER_FB_TYPES_H */
