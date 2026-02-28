/**
 * @file backend_auto_detect.c
 * @brief Implementation of automatic dlsym-based vtable population.
 *
 * See backend_auto_detect.h for usage documentation.
 */

/* ── Platform DLL symbol lookup ─────────────────────────────────────────── */
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  define FB_LOAD_SYM(handle, sym_name) \
       ((void *)GetProcAddress((HMODULE)(handle), (sym_name)))
#else
#  include <dlfcn.h>
#  define FB_LOAD_SYM(handle, sym_name) dlsym((handle), (sym_name))
#endif

#include "../backends/backend_auto_detect.h" /* fb_sym_entry_t, fb_fn_entry_t  */
#include "../judge/judge_op_ids.h"           /* FB_JUDGE_MAX_OPERATIONS       */

/* ── fb_auto_populate_ext_ops ───────────────────────────────────────────── */

void fb_auto_populate_ext_ops(fb_backend_vtable_t  *vtable,
                               void                 *lib_handle,
                               const fb_sym_entry_t *table,
                               size_t                count)
{
    if (!vtable || !lib_handle || !table || count == 0) return;

    for (size_t i = 0; i < count; i++) {
        uint32_t id = table[i].op_id;

        /* Bounds check — protects against stale generated tables after a
           judge_op_ids.h regeneration that reduced the op count.           */
        if (id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) continue;

        /* Never overwrite a typed wrapper that was already placed by the
           static vtable initialiser or by a previous fb_auto_populate call. */
        if (vtable->ext_ops[id] != NULL) continue;

        void *fn = FB_LOAD_SYM(lib_handle, table[i].symbol);
        if (fn) {
            vtable->ext_ops[id] = (fb_generic_fn)fn;
        }
    }
}

/* ── fb_auto_populate_fn_ops ────────────────────────────────────────────── */

void fb_auto_populate_fn_ops(fb_backend_vtable_t *vtable,
                              const fb_fn_entry_t *table,
                              size_t               count)
{
    if (!vtable || !table || count == 0) return;

    for (size_t i = 0; i < count; i++) {
        uint32_t id = table[i].op_id;

        if (id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) continue;
        if (vtable->ext_ops[id] != NULL) continue;
        if (table[i].fn == NULL) continue;

        vtable->ext_ops[id] = table[i].fn;
    }
}
