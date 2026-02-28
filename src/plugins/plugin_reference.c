/**
 * @file plugin_reference.c
 * @brief Built-in reference backend plugin.
 *
 * Wraps the always-available reference implementation (compiled directly into
 * the faster-blaster library) as a proper plugin.  Because the implementation
 * is statically linked, no DLL search is required.
 *
 * Design intent:
 *   - Score 5: always available but chosen last (safety net / correctness oracle).
 *   - Exposes FB_BACKEND_ID_REFERENCE (11) to the plugin registry so the judge
 *     module and ranked dispatch tables can profile and select it.
 *   - No dynamic library loading; probe() always succeeds.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include "../backends/backend_interface.h"
#include "../backends/reference.h"   /* fb_reference_backend() */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/* =========================================================================
 * Plugin metadata
 * ========================================================================= */

static const fb_plugin_metadata_t g_reference_metadata = {
    .name        = "reference",
    .version     = "1.0.0",
    .vendor      = "faster-blaster",
    .description = "Built-in reference BLAS implementation (correct, slow). "
                   "Used as correctness oracle and last-resort fallback.",
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
 * Probe — always available; lowest priority score
 * ========================================================================= */

static fb_plugin_probe_result_t reference_probe(
    fb_lib_handle_t  unused_lib_handle,
    const char     **unused_search_paths)
{
    (void)unused_lib_handle;
    (void)unused_search_paths;

    return (fb_plugin_probe_result_t){
        .score        = 5,          /* Lowest priority — last resort */
        .library_path = NULL,       /* Statically linked, no external library */
        .reason       = "Built-in reference implementation always available",
    };
}

/* =========================================================================
 * Init / shutdown — nothing to do for a statically linked backend
 * ========================================================================= */

static int reference_init(
    fb_lib_handle_t       unused_lib_handle,
    fb_plugin_context_t **ctx_out)
{
    (void)unused_lib_handle;
    /* No dynamic context needed; pass a non-NULL sentinel so the caller can
     * distinguish "initialised" from "not initialised". */
    *ctx_out = (fb_plugin_context_t *)(uintptr_t)1u;
    return 0;
}

static void reference_plugin_shutdown(fb_plugin_context_t *ctx)
{
    (void)ctx;  /* Nothing to free — statically linked */
}

/* =========================================================================
 * Vtable
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
    .set_num_threads = NULL,  /* Single-threaded — no threading knob */
    .get_num_threads = NULL,
};

/* =========================================================================
 * Registration
 * ========================================================================= */

/**
 * Register the built-in reference plugin.
 * Called from fb_init_plugins() in plugin_registry.c.
 */
void fb_register_reference_plugin(void)
{
    fb_register_plugin(&g_reference_plugin);
}
