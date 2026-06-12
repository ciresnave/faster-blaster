/**
 * @file backend_auto_detect.h
 * @brief Automatic dlsym / export-scan vtable population for faster-blaster
 *        backends.
 *
 * Two complementary entry points:
 *
 *   fb_auto_populate_ext_ops()   — fill from a caller-supplied {op_id, symbol}
 *                                  table (pre-generated tables in sym_tables/)
 *   fb_enumerate_and_populate()  — auto-scan ALL exports from a DLL/SO, classify
 *                                  each by naming convention, fill 2-D ext_ops
 *
 *   Typical backend get_vtable() call order
 *   ────────────────────────────────────────
 *   1. Static struct initialiser fills named typed fields  (BLAS L1/L2/L3)
 *   2. fb_enumerate_and_populate()  walks all DLL exports, classifies names,
 *      fills ext_ops[op_id][conv] for CBLAS, Fortran, and _ref slots.
 *      (OR: fb_auto_populate_ext_ops() with a pre-generated table — slower.)
 *   3. plugin_registry calls fb_finalize_plugin_vtable()
 *      → fb_vtable_sync_ext_ops() mirrors named fields → ext_ops[][FB_CONV_CBLAS]
 *      → Strategy 5 fills empty conv slots via static conv_thunks.c thunks
 *      → Strategies 1-4 generate missing precision/batch variants
 *
 * Usage (in a backend's init()):
 *   fb_enumerate_and_populate(&g_openblas_vtable, g_openblas.handle);
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
 * Symbol table entry — maps an operation ID and calling convention to the
 * exact symbol name that should be resolved from the backend's shared library.
 *
 * The `conv` field was added alongside the 2-D ext_ops expansion.  Existing
 * pre-generated tables in sym_tables/ are all FB_CONV_CBLAS; ScaLAPACK tables
 * use FB_CONV_FORTRAN.
 *
 * For compile-time-linked backends (cuBLAS, Metal) use fb_fn_entry_t instead.
 */
typedef struct {
    uint32_t    op_id;   /**< FB_OP_* constant from judge_op_ids.h           */
    fb_conv_t   conv;    /**< Which ext_ops[][conv] slot to fill              */
    const char *symbol;  /**< Exact name to pass to GetProcAddress / dlsym   */
} fb_sym_entry_t;

/**
 * Function-pointer table entry — for backends that are linked at compile time
 * (cuBLAS, Metal, Intel oneMKL when statically linked) where dlsym is not
 * available.  A backend builds a static array of these and passes it to
 * fb_auto_populate_fn_ops().  Function pointers in these tables are assumed
 * to use CBLAS convention (FB_CONV_CBLAS).
 */
typedef struct {
    uint32_t       op_id;  /**< FB_OP_* constant                              */
    fb_generic_fn  fn;     /**< Cast of the concrete function pointer         */
} fb_fn_entry_t;

/**
 * Enumerate ALL exported symbols from @p lib_handle, classify each by naming
 * convention (cblas_* / *_ / *_ref), look up the canonical stem in the
 * reverse stem→FB_OP_* table, and fill ext_ops[op_id][conv] if the slot is
 * currently NULL.
 *
 * Platform support:
 *   Windows  — walks the PE Export Directory (IMAGE_EXPORT_DIRECTORY)
 *   Linux    — iterates .dynsym via dl_iterate_phdr + ElfW(Sym)
 *   macOS    — walks Mach-O LC_DYSYMTAB / nlist via dyld APIs
 *
 * This replaces ~100 lines of per-symbol manual typedef + GetProcAddress in
 * plugin_openblas.c, plugin_aocl_blis.c, plugin_mkl.c, etc.
 *
 * @param vtable      Non-NULL vtable to fill.
 * @param lib_handle  Shared-library handle (HMODULE on Windows, void* on POSIX).
 */
void fb_enumerate_and_populate(fb_backend_vtable_t *vtable,
                                void                *lib_handle);

/**
 * Classify a single exported symbol name into a (op_id, conv) pair.
 *
 * Patterns recognised:
 *   "cblas_<stem>"  → FB_CONV_CBLAS,   *out_op_id = stem lookup
 *   "<stem>_"       → FB_CONV_FORTRAN, *out_op_id = stem lookup
 *
 * @param name       Null-terminated exported symbol name.
 * @param out_op_id  Receives the FB_OP_* constant on success.
 * @return  Recognised convention, or FB_CONV_COUNT if not a BLAS/LAPACK op.
 */
fb_conv_t fb_classify_symbol(const char *name, uint32_t *out_op_id);

/**
 * Scan @p table (length @p count), resolve each entry's symbol from
 * @p lib_handle via GetProcAddress / dlsym, and write the result into
 * @p vtable->ext_ops[op_id][conv] **only if the slot is currently NULL**.
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
 * @param table       Array of { op_id, conv, symbol_name } entries.
 * @param count       Number of entries in @p table.
 */
void fb_auto_populate_ext_ops(fb_backend_vtable_t  *vtable,
                               void                 *lib_handle,
                               const fb_sym_entry_t *table,
                               size_t                count);

/**
 * Scan @p table (length @p count) of already-resolved function pointers and
 * write each into vtable->ext_ops[op_id][FB_CONV_CBLAS] if the slot is
 * currently NULL.  Intended for compile-time-linked backends (cuBLAS, Metal).
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
