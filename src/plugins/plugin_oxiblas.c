/**
 * @file plugin_oxiblas.c
 * @brief OxiBLAS backend plugin
 *
 * Integrates OxiBLAS (pure-Rust BLAS/LAPACK with SIMD) as a faster-blaster
 * CPU backend.  OxiBLAS exports Fortran-convention symbols only (saxpy_,
 * dgemm_, …) via its C-ABI oxiblas-ffi crate.  fb_enumerate_and_populate()
 * fills FB_CONV_FORTRAN slots automatically; conv_thunks.c fills FB_CONV_CBLAS
 * slots at plugin finalisation time — no manual typedef boilerplate needed.
 *
 * Performance notes (oxiblas 0.1.0, source benchmarks):
 *   Apple M3 f32 SGEMM : 97–172 % of OpenBLAS  (outstanding)
 *   Linux x86_64 SGEMM : 80–112 % of OpenBLAS  (competitive)
 *   Windows            : EXPERIMENTAL — benchmark data pending
 *
 * Backend ID : FB_BACKEND_ID_OXIBLAS (13)
 *
 * @copyright Copyright (c) 2025
 * @license   MIT OR Apache-2.0
 */

#include "../../include/faster-blaster/backend_plugin.h"
#include "../../include/faster-blaster/backend_ids.h"
#include "../../include/faster-blaster/vtable_autofill.h"
#include "../backends/backend_interface.h"
#include "../backends/backend_auto_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Platform-portable dynamic-loader macros (same as other plugins)
 * ======================================================================== */
#ifdef _WIN32
#  include <windows.h>
#  define FB_LOAD_LIBRARY(path)            LoadLibraryA(path)
#  define FB_GET_PROC_ADDRESS(hdl, name)   GetProcAddress((HMODULE)(hdl), name)
#  define FB_FREE_LIBRARY(hdl)             FreeLibrary((HMODULE)(hdl))
#else
#  include <dlfcn.h>
#  define FB_LOAD_LIBRARY(path)            dlopen(path, RTLD_LAZY | RTLD_LOCAL)
#  define FB_GET_PROC_ADDRESS(hdl, name)   dlsym((hdl), (name))
#  define FB_FREE_LIBRARY(hdl)             dlclose(hdl)
#endif

/* =========================================================================
 * Candidate library names — ordered: shared preferred over static because
 * fb_enumerate_and_populate() requires a loaded (mapped) library handle.
 * ======================================================================== */
static const char *const k_oxiblas_lib_names[] = {
#if defined(_WIN32)
    "oxiblas_ffi.dll",
    "liboxiblas_ffi.dll",
#elif defined(__APPLE__)
    "liboxiblas_ffi.dylib",
    "liboxiblas_ffi.0.dylib",
#else  /* Linux / other ELF */
    "liboxiblas_ffi.so",
    "liboxiblas_ffi.so.0",
    "liboxiblas_ffi.so.1",
#endif
    NULL
};

/* =========================================================================
 * Plugin context
 * ======================================================================== */
typedef struct {
    fb_lib_handle_t  lib_handle;
} oxiblas_plugin_context_t;

static fb_backend_vtable_t          g_oxiblas_vtable;
static oxiblas_plugin_context_t    *g_oxiblas_context = NULL;

/* =========================================================================
 * Plugin metadata
 * ======================================================================== */
static const fb_plugin_metadata_t g_oxiblas_metadata = {
    .name        = "oxiblas",
    .version     = "0.1.0",
    .vendor      = "cool-japan",
    .description = "Pure-Rust BLAS/LAPACK with SIMD (oxiblas-ffi C ABI)",

    /* OxiBLAS covers all BLAS levels + a LAPACK subset.
     * Convention: Fortran (*_) — CBLAS slots filled by conv_thunks at load. */
    .capabilities = (
          FB_PLUGIN_CAP_CPU
        | FB_PLUGIN_CAP_LEVEL1
        | FB_PLUGIN_CAP_LEVEL2
        | FB_PLUGIN_CAP_LEVEL3
        | FB_PLUGIN_CAP_LAPACK
        | FB_PLUGIN_CAP_SINGLE_PREC
        | FB_PLUGIN_CAP_DOUBLE_PREC
        | FB_PLUGIN_CAP_COMPLEX
    ),
};

/* =========================================================================
 * Helpers
 * ======================================================================== */

/** Try to load the library from a list of candidate names / paths. */
static fb_lib_handle_t oxiblas_try_load(const char **search_paths)
{
    for (int i = 0; k_oxiblas_lib_names[i]; ++i) {
        /* Plain name — relies on system library path / LD_LIBRARY_PATH */
        fb_lib_handle_t h = FB_LOAD_LIBRARY(k_oxiblas_lib_names[i]);
        if (h) return h;

        /* Try each explicit search path */
        if (search_paths) {
            for (int j = 0; search_paths[j]; ++j) {
                char buf[1024];
                snprintf(buf, sizeof(buf), "%s/%s",
                         search_paths[j], k_oxiblas_lib_names[i]);
                h = FB_LOAD_LIBRARY(buf);
                if (h) return h;
            }
        }
    }
    return NULL;
}

/* =========================================================================
 * Plugin lifecycle: probe
 * ======================================================================== */
static fb_plugin_probe_result_t oxiblas_probe(
        [[maybe_unused]] fb_lib_handle_t unused_lib_handle,
        const char **search_paths)
{
    fb_plugin_probe_result_t result = { .score = 0, .reason = NULL };

    fb_lib_handle_t h = oxiblas_try_load(search_paths);
    if (!h) {
        result.reason = "liboxiblas_ffi not found";
        return result;
    }

    /* OxiBLAS exports Fortran-convention symbols only.
     * Verify the two most fundamental ones are present. */
    void *saxpy_sym  = FB_GET_PROC_ADDRESS(h, "saxpy_");
    void *dgemm_sym  = FB_GET_PROC_ADDRESS(h, "dgemm_");

    FB_FREE_LIBRARY(h);

    if (!saxpy_sym || !dgemm_sym) {
        result.reason = "liboxiblas_ffi found but saxpy_/dgemm_ missing";
        return result;
    }

    /* Score: strong GEMM on AArch64; competitive on x86_64; experimental
     * overall (no Windows benchmark data yet). */
    result.score  = 85;
    result.reason = "OxiBLAS 0.1.0 — pure-Rust BLAS/LAPACK (EXPERIMENTAL)";
    return result;
}

/* =========================================================================
 * Plugin lifecycle: init
 * ======================================================================== */
static int oxiblas_init(
        fb_lib_handle_t lib_handle,
        fb_plugin_context_t **ctx_out)
{
    if (!ctx_out) return -1;

    /* Use the library handle provided by probe(); re-load if NULL (fallback). */
    fb_lib_handle_t h = lib_handle ? lib_handle : oxiblas_try_load(NULL);
    if (!h) {
        fprintf(stderr, "[OxiBLAS] init: liboxiblas_ffi not found\n");
        return -2;
    }

    /* Quick sanity check */
    if (!FB_GET_PROC_ADDRESS(h, "saxpy_") ||
        !FB_GET_PROC_ADDRESS(h, "dgemm_")) {
        fprintf(stderr, "[OxiBLAS] init: required Fortran symbols missing\n");
        FB_FREE_LIBRARY(h);
        return -3;
    }

    /* Allocate plugin context */
    oxiblas_plugin_context_t *ctx =
        (oxiblas_plugin_context_t *)calloc(1, sizeof(*ctx));
    if (!ctx) { FB_FREE_LIBRARY(h); return -4; }
    ctx->lib_handle = h;

    /* ------------------------------------------------------------------ *
     * Auto-populate vtable from exported Fortran symbols.                *
     * fb_enumerate_and_populate() walks the PE/ELF/Mach-O export table,  *
     * classifies each symbol (saxpy_ → FB_CONV_FORTRAN, etc.) and fills  *
     * ext_ops[op_id][FB_CONV_FORTRAN].  fb_vtable_sync_ext_ops() then    *
     * mirrors FB_CONV_CBLAS slots (filled by thunks) into named fields.  *
     * ------------------------------------------------------------------ */
    memset(&g_oxiblas_vtable, 0, sizeof(g_oxiblas_vtable));
    fb_enumerate_and_populate(&g_oxiblas_vtable, h);
    fb_vtable_sync_ext_ops(&g_oxiblas_vtable);

    /* Validate essentials */
    if (!g_oxiblas_vtable.saxpy || !g_oxiblas_vtable.dgemm) {
        fprintf(stderr,
                "[OxiBLAS] init: vtable sync produced NULL saxpy/dgemm — "
                "conv_thunks may be missing for this platform\n");
        free(ctx);
        FB_FREE_LIBRARY(h);
        return -5;
    }

    g_oxiblas_context = ctx;
    *ctx_out = (fb_plugin_context_t *)ctx;

    printf("[OxiBLAS] Loaded %s (Fortran FFI; %zu ops auto-populated)\n",
           k_oxiblas_lib_names[0],
           (size_t)FB_JUDGE_MAX_OPERATIONS);

    return 0;
}

/* =========================================================================
 * Plugin lifecycle: shutdown
 * ======================================================================== */
static void oxiblas_shutdown(fb_plugin_context_t *ctx)
{
    if (!ctx) return;
    oxiblas_plugin_context_t *oc = (oxiblas_plugin_context_t *)ctx;
    if (oc->lib_handle) {
        FB_FREE_LIBRARY(oc->lib_handle);
        oc->lib_handle = NULL;
    }
    free(oc);
    g_oxiblas_context = NULL;
    memset(&g_oxiblas_vtable, 0, sizeof(g_oxiblas_vtable));
}

/* =========================================================================
 * Plugin lifecycle: get_vtable / get_context
 * ======================================================================== */
static const fb_backend_vtable_t *oxiblas_get_vtable(
        [[maybe_unused]] fb_plugin_context_t *ctx)
{
    return &g_oxiblas_vtable;
}

static void *oxiblas_get_context([[maybe_unused]] fb_plugin_context_t *ctx)
{
    return (void *)g_oxiblas_context;
}

/* =========================================================================
 * Plugin descriptor
 * ======================================================================== */
static const fb_backend_plugin_t g_oxiblas_plugin = {
    .metadata    = &g_oxiblas_metadata,
    .probe       = oxiblas_probe,
    .init        = oxiblas_init,
    .shutdown    = oxiblas_shutdown,
    .get_vtable  = oxiblas_get_vtable,
    .get_context = oxiblas_get_context,
};

/* =========================================================================
 * Public registration entry point — called by plugin_registry.c on startup
 * ======================================================================== */
void fb_register_oxiblas_plugin(void)
{
    fb_register_plugin(&g_oxiblas_plugin);
}
