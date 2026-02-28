/**
 * @file backend_auto_detect.h
 * @brief Automatic dlsym-based vtable population for faster-blaster backends.
 *
 * Provides fb_auto_populate_ext_ops() — the "input side" of the vtable
 * fill pipeline:
 *
 *   Typical backend get_vtable() call order
 *   ────────────────────────────────────────
 *   1. Static struct initialiser fills named typed fields  (BLAS L1/L2/L3)
 *   2. fb_auto_populate_ext_ops()  dlsym-fills ext_ops[]   (LAPACK/MKL-ext/…)
 *      — NULL check: typed wrappers already assigned win
 *   3. plugin_registry calls fb_finalize_plugin_vtable()
 *      → fb_vtable_sync_ext_ops() mirrors named fields → ext_ops[] (only NULL)
 *      → autofill strategies generate missing variants
 *
 * Usage (in a backend's get_vtable()):
 *   fb_auto_populate_ext_ops(&g_openblas_vtable, g_openblas.handle,
 *                             k_lapacke_symbols, k_lapacke_symbols_count);
 */

#ifndef FB_BACKEND_AUTO_DETECT_H
#define FB_BACKEND_AUTO_DETECT_H

#include "backend_interface.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Symbol table entry — maps an operation ID to the name of a symbol that
 * should be resolved from the backend's shared library at runtime.
 *
 * For compile-time-linked backends (cuBLAS, Metal) use fb_fn_entry_t instead.
 */
typedef struct {
    uint32_t    op_id;   /**< FB_OP_* constant from judge_op_ids.h          */
    const char *symbol;  /**< Exact name to pass to GetProcAddress / dlsym   */
} fb_sym_entry_t;

/**
 * Function-pointer table entry — for backends that are linked at compile time
 * (cuBLAS, Metal, Intel oneMKL when statically linked) where dlsym is not
 * available.  A backend builds a static array of these and passes it to
 * fb_auto_populate_fn_ops().
 */
typedef struct {
    uint32_t       op_id;  /**< FB_OP_* constant                             */
    fb_generic_fn  fn;     /**< Cast of the concrete function pointer        */
} fb_fn_entry_t;

/**
 * Scan @p table (length @p count), resolve each entry's symbol from
 * @p lib_handle via GetProcAddress / dlsym, and write the result into
 * @p vtable->ext_ops[op_id] **only if the slot is currently NULL**.
 *
 * This ensures that hand-written typed wrappers (assigned in the static
 * vtable initialiser or filled by fb_vtable_sync_ext_ops) always take
 * precedence over the raw dlsym'd pointer.
 *
 * Thread safety: call once per backend while the library handle is valid.
 * All backends are initialised before any concurrent dispatch begins.
 *
 * @param vtable      Non-NULL vtable to fill.
 * @param lib_handle  Shared-library handle (HMODULE on Windows, void* on POSIX).
 * @param table       Array of { op_id, symbol_name } entries.
 * @param count       Number of entries in @p table.
 */
void fb_auto_populate_ext_ops(fb_backend_vtable_t  *vtable,
                               void                 *lib_handle,
                               const fb_sym_entry_t *table,
                               size_t                count);

/**
 * Scan @p table (length @p count) of already-resolved function pointers and
 * write each into vtable->ext_ops[op_id] if the slot is currently NULL.
 * Intended for compile-time-linked backends (cuBLAS, Metal).
 *
 * @param vtable  Non-NULL vtable to fill.
 * @param table   Array of { op_id, fn_ptr } entries.
 * @param count   Number of entries in @p table.
 */
void fb_auto_populate_fn_ops(fb_backend_vtable_t *vtable,
                              const fb_fn_entry_t *table,
                              size_t               count);

#ifdef __cplusplus
}
#endif

#endif /* FB_BACKEND_AUTO_DETECT_H */
