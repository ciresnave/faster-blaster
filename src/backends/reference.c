/**
 * @file reference.c
 * @brief Reference backend — runtime-loaded via LoadLibrary/dlopen.
 *
 * faster-blaster-reference is a separate DLL loaded at runtime by
 * plugin_reference.c using LoadLibrary (Windows) / dlopen (POSIX).
 * No import library, no static link, no fbr headers needed here.
 *
 * All vtable slots are populated by fb_enumerate_and_populate() which walks
 * the DLL export table, matches cblas_* symbol names against the op-stem map,
 * and stores raw function pointers in ext_ops[][FB_CONV_CBLAS].
 * fb_vtable_fill_named_from_ext_ops() then propagates ext_ops -> named fields
 * so the judge and dispatch code can reach every operation directly.
 *
 * If the DLL is absent, the vtable has all-NULL function pointers and the
 * reference backend simply scores 3 (lowest priority) with no operations.
 *
 * Plugin score: 5 (DLL found) / 3 (DLL absent).
 * Purpose     : correctness oracle and last-resort fallback.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "reference.h"
#include "backend_interface.h"
#include "faster-blaster/backend_plugin.h"  /* fb_lib_handle_t                */
#include "faster-blaster/vtable_autofill.h" /* fb_vtable_sync_ext_ops,
                                               fb_vtable_fill_named_from_ext_ops */
#include "backend_auto_detect.h"            /* fb_enumerate_and_populate      */

/* ============================================================================
 * Backend vtable
 * ========================================================================= */

const fb_backend_vtable_t *fb_reference_backend(void) {
  static fb_backend_vtable_t vt = {0};
  static int ready = 0;
  if (ready)
    return &vt;

  vt.info.name = "reference";
  vt.info.version = "3.0.0";
  vt.info.vendor = "faster-blaster-reference";
  vt.info.capabilities = FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3;
  vt.info.hw_type = FB_HW_CPU_INTEL;
  vt.info.thread_safe = true;
  vt.info.min_efficient_size = 0;

  /* No manual vtable wiring. All function pointers are filled by
   * fb_enumerate_and_populate() in fb_reference_init() once the DLL is
   * loaded.  fb_vtable_fill_named_from_ext_ops() propagates ext_ops[]
   * -> named fields after the scan. */

  ready = 1;
  return &vt;
}

/* Convenience alias used by backend_instance.c and dispatch_unified.c. */
const fb_backend_vtable_t *fb_reference_get_vtable(void) {
  return fb_reference_backend();
}

/* ============================================================================
 * DLL initialisation
 * ========================================================================= */

/**
 * fb_reference_init — call once after loading faster_blaster_reference.dll/so.
 *
 * Walks the DLL export table for cblas_* symbols, matches each against the
 * op-stem map (fb_classify_symbol), and stores the raw function pointer in
 * ext_ops[op_id][FB_CONV_CBLAS].  fb_vtable_fill_named_from_ext_ops() then
 * propagates those pointers into the named typed fields used by the judge and
 * dispatch code.  fb_vtable_sync_ext_ops() keeps the two arrays consistent.
 *
 * Safe to call with lib_handle == NULL (no-op).
 */
void fb_reference_init(fb_lib_handle_t lib_handle) {
  /* Ensure the static info block has been initialised. */
  fb_backend_vtable_t *vt = (fb_backend_vtable_t *)fb_reference_backend();

  if (lib_handle) {
    /* 1. Walk DLL export table -> fill ext_ops[][FB_CONV_CBLAS]. */
    fb_enumerate_and_populate(vt, lib_handle);
    /* 2. Finalize ext_ops so Fortran-only exports gain C-ABI thunk slots. */
    fb_finalize_plugin_vtable(vt);
    /* 3. Propagate ext_ops -> named typed fields for direct judge access. */
    fb_vtable_fill_named_from_ext_ops(vt);
    /* 4. Mirror named -> ext_ops to keep everything consistent. */
    fb_vtable_sync_ext_ops(vt);
  }
}
