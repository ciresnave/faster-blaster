/**
 * @file plugin_registry.c
 * @brief Plugin discovery and management
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "../../include/faster-blaster/backend_plugin.h"
#include "../../include/faster-blaster/vtable_autofill.h"
#include "../../include/faster-blaster/backend_ids.h"
#include "../../include/faster-blaster/data_tracker.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void fb_register_clblast_plugin(void);
extern void fb_register_clblas_plugin(void);
extern void fb_register_oxiblas_plugin(void);


#ifdef _WIN32
#include <windows.h>
#define FB_LOAD_LIBRARY(path) LoadLibraryA(path)
#define FB_GET_PROC_ADDRESS(handle, name)                                      \
  GetProcAddress((HMODULE)(handle), name)
#define FB_FREE_LIBRARY(handle) FreeLibrary((HMODULE)(handle))
#else
#include <dlfcn.h>
#define FB_LOAD_LIBRARY(path) dlopen(path, RTLD_LAZY)
#define FB_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#define FB_FREE_LIBRARY(handle) dlclose(handle)
#endif

/* Global plugin registry - linked list of registered plugins */
static fb_plugin_registry_entry_t *g_plugin_registry = NULL;

/* ============================================================================
 * Per-backend vtable cache
 *
 * Indexed by the same "scoreable-plugin position" used by exec_dag's
 * snapshot_backends().  Entry is populated either by fb_load_best_plugin()
 * (which initialises one backend) or lazily inside fb_get_vtable_by_id().
 * ========================================================================= */
#define FB_MAX_CACHED_BACKENDS 16

typedef struct {
    fb_plugin_context_t          *ctx;    /* NULL = not yet initialised */
    const fb_backend_vtable_t    *vtable; /* NULL = not yet available   */
} fb_vtable_cache_entry_t;

static fb_vtable_cache_entry_t g_vtable_cache[FB_MAX_CACHED_BACKENDS];

/* ============================================================================
 * Stable-ID vtable cache
 *
 * Indexed by FB_BACKEND_ID_* constants (0..FB_BACKEND_ID__NEXT_FREE-1).
 * Populated on first use by fb_get_vtable_by_backend_id(), or eagerly by
 * fb_load_best_plugin() for the elected backend.
 * ========================================================================= */
static fb_vtable_cache_entry_t g_id_vtable_cache[FB_BACKEND_ID__NEXT_FREE];

/**
 * Static table: FB_BACKEND_ID_* → plugin metadata name string.
 * Must stay in sync with backend_ids.h and the plugin name fields.
 */
typedef struct { uint32_t id; const char *name; } fb_backend_id_name_t;
static const fb_backend_id_name_t k_backend_id_names[] = {
    { FB_BACKEND_ID_AOCL_BLIS,    "aocl-blis"              },
    { FB_BACKEND_ID_BLIS,         "blis"                   },
    { FB_BACKEND_ID_OPENBLAS,     "openblas"               },
    { FB_BACKEND_ID_MKL,          "mkl"                    },
    { FB_BACKEND_ID_ACCELERATE,   "accelerate"             },
    { FB_BACKEND_ID_CUBLAS,       "cublas"                 },
    { FB_BACKEND_ID_ROCBLAS,      "rocblas"                },
    { FB_BACKEND_ID_ONEMKL,       "onemkl"                 },
    { FB_BACKEND_ID_METAL,        "metal"                  },
    { FB_BACKEND_ID_CLBLAST,      "clblast"                },
    { FB_BACKEND_ID_CLBLAS,       "clblas"                 },
    { FB_BACKEND_ID_REFERENCE,    "reference"              },
    { FB_BACKEND_ID_BLR,          "blas-lapack-reference"  },
    { FB_BACKEND_ID_OXIBLAS,       "oxiblas"                },
};
#define K_BACKEND_ID_NAMES_COUNT \
    (sizeof(k_backend_id_names)/sizeof(k_backend_id_names[0]))

/**
 * Walk the registry (same filter as snapshot_backends) and return the
 * scoreable position of @p plugin, or -1 if not found.
 */
static int plugin_scoreable_index(const fb_backend_plugin_t *plugin)
{
    int idx = 0;
    for (const fb_plugin_registry_entry_t *e = g_plugin_registry; e; e = e->next) {
        if (!e->plugin || !e->plugin->probe || !e->plugin->metadata) continue;
        fb_plugin_probe_result_t r = e->plugin->probe(NULL, NULL);
        if (r.score <= 0) continue;
        if (e->plugin == plugin) return idx;
        idx++;
    }
    return -1;
}

int fb_register_plugin(const fb_backend_plugin_t *plugin) {
  if (!plugin || !plugin->metadata || !plugin->probe || !plugin->init ||
      !plugin->get_vtable) {
    return -1; /* Invalid plugin */
  }

  /* Allocate registry entry */
  fb_plugin_registry_entry_t *entry =
      (fb_plugin_registry_entry_t *)malloc(sizeof(fb_plugin_registry_entry_t));
  if (!entry) {
    return -2; /* Out of memory */
  }

  entry->plugin = plugin;
  entry->next = g_plugin_registry;
  g_plugin_registry = entry;

  return 0;
}

const fb_plugin_registry_entry_t *fb_get_registered_plugins(void) {
  return g_plugin_registry;
}

const fb_backend_plugin_t *fb_load_best_plugin(const char *backend_name,
                                               const char **search_paths,
                                               fb_plugin_context_t **ctx_out) {
  if (!ctx_out) {
    return NULL;
  }

  const fb_backend_plugin_t *best_plugin = NULL;
  fb_plugin_probe_result_t best_result = {0};

  /* Probe all registered plugins */
  for (fb_plugin_registry_entry_t *entry = g_plugin_registry; entry;
       entry = entry->next) {
    const fb_backend_plugin_t *plugin = entry->plugin;

    /* If backend_name specified, skip non-matching plugins */
    if (backend_name && strcmp(plugin->metadata->name, backend_name) != 0) {
      continue;
    }

    /* Probe this plugin */
    fb_plugin_probe_result_t result = plugin->probe(NULL, search_paths);

    printf("[DEBUG] Probed %s: score=%d, reason=%s\n", plugin->metadata->name,
           result.score, result.reason ? result.reason : "");

    /* Skip if incompatible */
    if (result.score <= 0) {
      continue;
    }

    /* Check if this is the best so far */
    if (result.score > best_result.score) {
      /* Plugin will load library in init() */
      best_plugin = plugin;
      best_result = result;
    }
  }

  /* Initialize the best plugin */
  if (best_plugin) {
    printf("[DEBUG] Initializing best plugin: %s (score=%d)\n",
           best_plugin->metadata->name, best_result.score);
    int ret =
        best_plugin->init(NULL, ctx_out); /* Plugin loads its own library */
    printf("[DEBUG] Init returned: %d\n", ret);
    if (ret != 0) {
      printf("[DEBUG] Init failed!\n");
      return NULL;
    }
    printf("[DEBUG] Plugin initialized successfully\n");

    /* Get vtable and apply auto-fill strategies */
    const fb_backend_vtable_t *vtable = best_plugin->get_vtable(*ctx_out);
    if (vtable) {
      /* Cast away const for finalization - vtable is modified in-place */
      fb_status_t autofill_status =
          fb_finalize_plugin_vtable((fb_backend_vtable_t *)vtable);
      if (autofill_status != FB_STATUS_SUCCESS) {
        printf("[DEBUG] Vtable auto-fill failed with status %d\n",
               autofill_status);
      } else {
        printf("[DEBUG] Vtable auto-fill completed successfully\n");
      }

      /* Set as active vtable */
      fb_set_active_vtable(vtable);

      /* Cache the vtable so fb_get_vtable_by_id() can locate it later */
      int slot = plugin_scoreable_index(best_plugin);
      if (slot >= 0 && slot < FB_MAX_CACHED_BACKENDS) {
        g_vtable_cache[slot].ctx    = *ctx_out;
        g_vtable_cache[slot].vtable = vtable;
      }

      /* Also populate the stable-ID cache for fb_get_vtable_by_backend_id() */
      const char *best_name = best_plugin->metadata ? best_plugin->metadata->name : NULL;
      if (best_name) {
        for (size_t ni = 0; ni < K_BACKEND_ID_NAMES_COUNT; ni++) {
          if (strcmp(k_backend_id_names[ni].name, best_name) == 0) {
            uint32_t bid = k_backend_id_names[ni].id;
            if (bid < FB_BACKEND_ID__NEXT_FREE) {
              g_id_vtable_cache[bid].ctx    = *ctx_out;
              g_id_vtable_cache[bid].vtable = vtable;
            }
            break;
          }
        }
      }
    }

    return best_plugin;
  }

  return NULL;
}

fb_lib_handle_t fb_plugin_load_library(const char **library_names,
                                       const char **search_paths) {
  if (!library_names) {
    return NULL;
  }

  /* Try each library name */
  for (const char **name_ptr = library_names; *name_ptr; name_ptr++) {
    const char *name = *name_ptr;

    /* Try with search paths first */
    if (search_paths) {
      for (const char **path_ptr = search_paths; *path_ptr; path_ptr++) {
        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", *path_ptr, name);

        fb_lib_handle_t handle = FB_LOAD_LIBRARY(full_path);
        if (handle) {
          return handle;
        }
      }
    }

    /* Try system search (LD_LIBRARY_PATH, PATH, etc.) */
    fb_lib_handle_t handle = FB_LOAD_LIBRARY(name);
    if (handle) {
      return handle;
    }
  }

  return NULL;
}

void *fb_plugin_get_symbol(fb_lib_handle_t lib_handle,
                           const char *symbol_name) {
  if (!lib_handle || !symbol_name) {
    return NULL;
  }
  return FB_GET_PROC_ADDRESS(lib_handle, symbol_name);
}

void fb_plugin_unload_library(fb_lib_handle_t lib_handle) {
  if (lib_handle) {
    FB_FREE_LIBRARY(lib_handle);
  }
}

/**
 * Return the vtable for the backend at scoreable position @p id.
 *
 * "Scoreable position" matches the indices assigned by exec_dag's
 * snapshot_backends(): iterate the registry, skip score<=0 plugins,
 * the n-th surviving plugin is id=n.
 *
 * Lookup order:
 *   1. g_vtable_cache[id] if already initialised (common case).
 *   2. Lazy init: probe the plugin; if score>0, call init()+get_vtable().
 *   3. Fallback: g_active_vtable (single globally active backend).
 */
const fb_backend_vtable_t *fb_get_vtable_by_id(uint32_t id)
{
    /* Fast path: already in cache */
    if (id < FB_MAX_CACHED_BACKENDS && g_vtable_cache[id].vtable) {
        return g_vtable_cache[id].vtable;
    }

    /* Walk to the id-th scoreable plugin */
    uint32_t idx = 0;
    for (const fb_plugin_registry_entry_t *e = g_plugin_registry; e; e = e->next) {
        const fb_backend_plugin_t *p = e->plugin;
        if (!p || !p->probe || !p->metadata || !p->init || !p->get_vtable) continue;

        fb_plugin_probe_result_t r = p->probe(NULL, NULL);
        if (r.score <= 0) continue;

        if (idx == id) {
            /* Lazy init: try to initialise this backend now */
            if (id < FB_MAX_CACHED_BACKENDS && g_vtable_cache[id].ctx == NULL) {
                fb_plugin_context_t *ctx = NULL;
                if (p->init(NULL, &ctx) == 0) {
                    const fb_backend_vtable_t *vt = p->get_vtable(ctx);
                    if (vt) {
                        fb_finalize_plugin_vtable((fb_backend_vtable_t *)vt);
                        g_vtable_cache[id].ctx    = ctx;
                        g_vtable_cache[id].vtable = vt;
                        return vt;
                    }
                    if (p->shutdown) p->shutdown(ctx);
                }
            }
            /* If lazy init failed, fall through to global fallback */
            break;
        }
        idx++;
    }

    /* Fallback: return the globally active vtable */
    return fb_get_active_vtable();
}

/**
 * Return the vtable for the backend identified by a stable FB_BACKEND_ID_*
 * constant.
 *
 * Lookup order:
 *   1. g_id_vtable_cache[id] if already initialised (common case after
 *      fb_load_best_plugin has run for this backend).
 *   2. Lazy init: walk the registry matching plugin metadata name, then
 *      probe/init/get_vtable if score > 0.
 *   3. Fallback: fb_get_active_vtable().
 *
 * @param fb_backend_id  One of FB_BACKEND_ID_AOCL_BLIS … FB_BACKEND_ID_BLR.
 * @return               Vtable pointer, or active vtable as fallback.
 */
const fb_backend_vtable_t *fb_get_vtable_by_backend_id(uint32_t fb_backend_id)
{
    if (fb_backend_id >= FB_BACKEND_ID__NEXT_FREE)
        return fb_get_active_vtable();

    /* Fast path: already cached */
    if (g_id_vtable_cache[fb_backend_id].vtable)
        return g_id_vtable_cache[fb_backend_id].vtable;

    /* Look up the canonical plugin name for this stable ID */
    const char *target_name = NULL;
    for (size_t i = 0; i < K_BACKEND_ID_NAMES_COUNT; i++) {
        if (k_backend_id_names[i].id == fb_backend_id) {
            target_name = k_backend_id_names[i].name;
            break;
        }
    }
    if (!target_name)
        return fb_get_active_vtable();

    /* Walk registry; lazy-init the matching plugin */
    for (fb_plugin_registry_entry_t *e = g_plugin_registry; e; e = e->next) {
        const fb_backend_plugin_t *p = e->plugin;
        if (!p || !p->metadata || !p->probe || !p->init || !p->get_vtable)
            continue;
        if (strcmp(p->metadata->name, target_name) != 0)
            continue;

        /* Found the right plugin — probe to check availability */
        fb_plugin_probe_result_t r = p->probe(NULL, NULL);
        if (r.score <= 0)
            break; /* Incompatible hardware — fall back to active */

        fb_plugin_context_t *ctx = NULL;
        if (p->init(NULL, &ctx) != 0)
            break;

        const fb_backend_vtable_t *vt = p->get_vtable(ctx);
        if (!vt) {
            if (p->shutdown) p->shutdown(ctx);
            break;
        }

        fb_finalize_plugin_vtable((fb_backend_vtable_t *)vt);
        g_id_vtable_cache[fb_backend_id].ctx    = ctx;
        g_id_vtable_cache[fb_backend_id].vtable = vt;
        return vt;
    }

    return fb_get_active_vtable();
}

/**
 * Reverse-map a plugin metadata name to its stable FB_BACKEND_ID_* constant.
 * Returns FB_BACKEND_ID_NONE when the name is not in the lookup table.
 */
uint32_t fb_get_backend_id_for_name(const char *name)
{
    if (!name) return FB_BACKEND_ID_NONE;
    for (size_t i = 0; i < K_BACKEND_ID_NAMES_COUNT; i++) {
        if (strcmp(k_backend_id_names[i].name, name) == 0)
            return k_backend_id_names[i].id;
    }
    return FB_BACKEND_ID_NONE;
}

void fb_init_plugins(void) {
  /* Initialise data-location tracking before any plugin is loaded */
  fb_data_tracker_init();
  /* Register all built-in CPU plugins */
#ifdef FB_ENABLE_AOCL
  fb_register_aocl_plugin();
#endif
#ifdef FB_ENABLE_BLIS
  fb_register_standard_blis_plugin();
#endif
#ifdef FB_ENABLE_OPENBLAS
  fb_register_openblas_plugin();
#endif
#ifdef FB_ENABLE_MKL
  fb_register_mkl_plugin();
#endif
#ifdef FB_ENABLE_ACCELERATE
  fb_register_accelerate_plugin();
#endif

  /* Register all built-in GPU plugins */
  /* fb_register_cublas_plugin(); */ /* Disabled - using pure dynamic loading
                                        via dispatch system */
#ifdef FB_ENABLE_ROCM
  fb_register_rocblas_plugin();
#endif
#ifdef FB_ENABLE_ONEMKL
  fb_register_onemkl_plugin();
#endif
#ifdef FB_ENABLE_METAL
  fb_register_metal_plugin();
#endif
#ifdef FB_ENABLE_CLBLAST
  fb_register_clblast_plugin();
  fb_register_clblas_plugin();
#endif

  /* Reference backend: correctness oracle and last-resort fallback.
   * Always available (statically linked); deliberately lowest score. */
  fb_register_reference_plugin();
  /* fb_register_blas_lapack_reference_plugin(); — removed (dead plugin) */
}
