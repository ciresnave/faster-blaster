/**
 * @file plugin_reference.c
 * @brief Reference backend plugin — loads faster_blaster_reference.dll/so.
 *
 * Probes for the companion faster-blaster-reference shared library, loads it,
 * calls fb_reference_init() to scan its cblas_* exports via
 * fb_enumerate_and_populate(), and exposes the wired vtable to the registry.
 *
 * Score 5: always lowest priority — chosen only when no optimised backend is
 * available.  Primary role: correctness oracle.
 *
 * Search order for the DLL:
 *   1. Directory containing faster-blaster.dll itself (deployment layout)
 *   2. ../faster-blaster-reference/build-extended/  (development layout)
 *   3. search_paths[] passed by the caller
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include "../backends/backend_auto_detect.h"   /* fb_enumerate_and_populate  */
#include "../backends/reference.h"             /* fb_reference_backend, fb_reference_init */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#  define FB_REF_LIB_NAME  "libfaster_blaster_reference.dll"
#  define FB_REF_LIB_NAME_ALT  "faster_blaster_reference.dll"
#elif defined(__APPLE__)
#  define FB_REF_LIB_NAME  "libfaster_blaster_reference.dylib"
#else
#  define FB_REF_LIB_NAME  "libfaster_blaster_reference.so"
#endif

/* Plugin context — just holds the library handle. */
typedef struct {
    fb_lib_handle_t lib_handle;
} reference_plugin_context_t;

static reference_plugin_context_t *g_reference_ctx = NULL;

/* =========================================================================
 * Plugin metadata
 * ========================================================================= */

static const fb_plugin_metadata_t g_reference_metadata = {
    .name        = "reference",
    .version     = "1.0.0",
    .vendor      = "faster-blaster",
    .description = "Reference BLAS/LAPACK implementation (correct, slow). "
                   "Correctness oracle and last-resort fallback.",
    .api_version = 1,
    .capabilities = (uint32_t)(
        FB_PLUGIN_CAP_CPU         |
        FB_PLUGIN_CAP_LEVEL1      |
        FB_PLUGIN_CAP_LEVEL2      |
        FB_PLUGIN_CAP_LEVEL3      |
        FB_PLUGIN_CAP_SINGLE_PREC |
        FB_PLUGIN_CAP_DOUBLE_PREC |
        FB_PLUGIN_CAP_COMPLEX     |
        FB_PLUGIN_CAP_THREADSAFE
    ),
};

/* =========================================================================
 * Default search paths
 * ========================================================================= */

static const char *k_default_search_paths[] = {
#ifdef _WIN32
    /* Development layout: sibling build directory */
    "..\\faster-blaster-reference\\build-extended",
    /* Possible install locations */
    "C:\\libraries\\faster-blaster-reference\\bin",
#else
    "../faster-blaster-reference/build-extended",
    "/usr/local/lib",
    "/usr/lib",
#endif
    NULL
};

static const char *k_lib_names[] = {
    FB_REF_LIB_NAME,
#ifdef _WIN32
    FB_REF_LIB_NAME_ALT,
#endif
    NULL
};

/* =========================================================================
 * Probe
 * ========================================================================= */

static fb_plugin_probe_result_t reference_probe(
    fb_lib_handle_t  unused_lib_handle,
    const char     **search_paths)
{
    (void)unused_lib_handle;

    fb_lib_handle_t h = fb_plugin_load_library(
        k_lib_names,
        search_paths ? search_paths : k_default_search_paths);

    if (h) {
        fb_plugin_unload_library(h);
        return (fb_plugin_probe_result_t){
            .score        = 5,
            .library_path = NULL,
            .reason       = "Found reference DLL — correctness oracle available",
        };
    }

    /* Fall back: the static named-field vtable in reference.c is always
     * available even without the DLL (192 manually wired ops). */
    return (fb_plugin_probe_result_t){
        .score        = 3,
        .library_path = NULL,
        .reason       = "Reference DLL not found; using statically wired vtable only",
    };
}

/* =========================================================================
 * Init
 * ========================================================================= */

static int reference_init(
    fb_lib_handle_t       provided_handle,
    fb_plugin_context_t **ctx_out)
{
    if (!ctx_out) return -1;

    fb_lib_handle_t h = provided_handle;
    if (!h) {
        h = fb_plugin_load_library(k_lib_names, k_default_search_paths);
        /* h == NULL is acceptable: fb_reference_init(NULL) is a no-op and the
         * 192 statically wired named fields are still usable. */
    }

    /* Populate vtable: static named fields first (lazy init inside
     * fb_reference_backend), then DLL scan for any additional cblas_* syms. */
    fb_reference_init(h);

    reference_plugin_context_t *ctx =
        (reference_plugin_context_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        if (h) fb_plugin_unload_library(h);
        return -2;
    }
    ctx->lib_handle = h;   /* keep DLL loaded for the lifetime of the plugin */
    g_reference_ctx = ctx;

    *ctx_out = (fb_plugin_context_t *)ctx;
    return 0;
}

/* =========================================================================
 * Shutdown
 * ========================================================================= */

static void reference_plugin_shutdown(fb_plugin_context_t *ctx)
{
    reference_plugin_context_t *rctx = (reference_plugin_context_t *)ctx;
    if (rctx) {
        /* Do NOT dlclose: the vtable function pointers inside reference.c
         * still point into the DLL's text segment.  Keep it loaded. */
        free(rctx);
    }
    g_reference_ctx = NULL;
}

/* =========================================================================
 * Vtable / context accessors
 * ========================================================================= */

static const fb_backend_vtable_t *reference_get_vtable(fb_plugin_context_t *ctx)
{
    (void)ctx;
    return fb_reference_backend();
}

static void *reference_get_context(fb_plugin_context_t *ctx)
{
    return ctx;
}

/* =========================================================================
 * Plugin descriptor
 * ========================================================================= */

static const fb_backend_plugin_t g_reference_plugin = {
    .metadata        = &g_reference_metadata,
    .probe           = reference_probe,
    .init            = reference_init,
    .get_vtable      = reference_get_vtable,
    .get_context     = reference_get_context,
    .shutdown        = reference_plugin_shutdown,
    .set_num_threads = NULL,
    .get_num_threads = NULL,
};

/* =========================================================================
 * Registration
 * ========================================================================= */

void fb_register_reference_plugin(void)
{
    fb_register_plugin(&g_reference_plugin);
}
