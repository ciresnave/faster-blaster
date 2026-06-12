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

/* Canonical precision type — shared by backend_interface.h and compute_manager.h */
typedef enum {
    FB_PREC_FP32   = 0,  /* Single precision float */
    FB_PREC_FP64   = 1,  /* Double precision float */
    FB_PREC_C64    = 2,  /* Complex single precision */
    FB_PREC_C128   = 3,  /* Complex double precision */
    FB_PREC_FP16   = 4,  /* Half precision (optional) */
    FB_PREC_BF16   = 5,  /* BFloat16 (optional) */
    FB_PREC_FP8    = 6,  /* FP8 (optional) */
    FB_PREC_INT8   = 7,  /* 8-bit integer (optional) */
    FB_PREC_INT32  = 8   /* 32-bit integer (optional) */
} fb_precision_t;
/* Aliased names used in compute_manager.h public API */
#define FB_PRECISION_FP32 FB_PREC_FP32
#define FB_PRECISION_FP64 FB_PREC_FP64
#define FB_PRECISION_FP16 FB_PREC_FP16
#define FB_PRECISION_BF16 FB_PREC_BF16

/** Type-erased function pointer for generic op-level dispatch.
 *  Cast to the concrete fn typedef before calling. */
typedef void (*fb_generic_fn)(void);

#endif /* FASTER_BLASTER_FB_TYPES_H */
