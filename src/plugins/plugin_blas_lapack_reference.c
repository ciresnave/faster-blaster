/**
 * @file plugin_blas_lapack_reference.c
 * @brief Plugin wrapper for the dynamically-loaded faster-blaster-reference
 * DLL.
 *
 * This plugin loads faster_blaster_reference.dll (or .so/.dylib) at runtime
 * and exposes the full 1248-operation BLAS/LAPACK reference implementation
 * through the standard faster-blaster plugin vtable interface.
 *
 * Design intent:
 *   - Score 10 when the DLL is found on disk: preferred over the statically
 *     linked reference backend (score 5) but below any optimised backend.
 *   - Score  0 when the DLL cannot be located: plugin is transparently skipped.
 *   - Backend ID: FB_BACKEND_ID_BLR (12) — distinct from
 * FB_BACKEND_ID_REFERENCE (11) so judge profiles for both backends can coexist
 * on disk.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "../backends/blas_lapack_reference_backend.h"
#include "faster-blaster/backend_ids.h"
#include "faster-blaster/backend_plugin.h"


#include <stdbool.h>
#include <stdio.h>
#include <string.h>


/* =========================================================================
 * Plugin metadata
 * ========================================================================= */

static const fb_plugin_metadata_t g_blr_plugin_metadata = {
    .name = "blr",
    .version = "1.0.0",
    .vendor = "faster-blaster",
    .description =
        "Dynamic faster-blaster-reference DLL loader. "
        "Provides 1248 BLAS/LAPACK operations from the authoritative "
        "pure-C reference implementation for correctness validation.",
    .api_version = 1,
    .capabilities =
        (uint32_t)(FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 |
                   FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
                   FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC |
                   FB_PLUGIN_CAP_COMPLEX | FB_PLUGIN_CAP_THREADSAFE),
};

/* =========================================================================
 * Probe — score 10 if DLL found, 0 otherwise
 * ========================================================================= */

static fb_plugin_probe_result_t blr_probe(fb_lib_handle_t unused_lib_handle,
                                          const char **unused_search_paths) {
  (void)unused_lib_handle;
  (void)unused_search_paths;

  const char *dll_path = fb_blr_get_dll_path();
  if (dll_path && dll_path[0] != '\0') {
    return (fb_plugin_probe_result_t){
        .score = 10,
        .library_path = dll_path,
        .reason = "faster-blaster-reference DLL found",
    };
  }

  return (fb_plugin_probe_result_t){
      .score = 0,
      .library_path = NULL,
      .reason = "faster-blaster-reference DLL not found",
  };
}

/* =========================================================================
 * Init / shutdown
 * ========================================================================= */

static int blr_init(fb_lib_handle_t unused_lib_handle,
                    fb_plugin_context_t **ctx_out) {
  (void)unused_lib_handle;

  if (fb_blr_init() != 0) {
    fprintf(stderr,
            "[faster-blaster] blr plugin: failed to load reference DLL\n");
    return -1;
  }

  /* Use a non-NULL sentinel so callers can distinguish initialised state. */
  *ctx_out = (fb_plugin_context_t *)(uintptr_t)2u;
  return 0;
}

static void blr_shutdown(fb_plugin_context_t *ctx) {
  (void)ctx;
  fb_blr_finalize();
}

/* =========================================================================
 * Vtable accessor
 * ========================================================================= */

static const fb_backend_vtable_t *blr_get_vtable(fb_plugin_context_t *ctx) {
  (void)ctx;
  return fb_blr_get_vtable();
}

static void *blr_get_context(fb_plugin_context_t *ctx) { return ctx; }

/* =========================================================================
 * Plugin descriptor
 * ========================================================================= */

static const fb_backend_plugin_t g_blr_plugin = {
    .metadata = &g_blr_plugin_metadata,
    .probe = blr_probe,
    .init = blr_init,
    .get_vtable = blr_get_vtable,
    .get_context = blr_get_context,
    .shutdown = blr_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL,
};

/* =========================================================================
 * Registration
 * ========================================================================= */

/**
 * Register the dynamic faster-blaster-reference DLL plugin.
 * Called from fb_init_plugins() in plugin_registry.c.
 */
void fb_register_blas_lapack_reference_plugin(void) {
  fb_register_plugin(&g_blr_plugin);
}
