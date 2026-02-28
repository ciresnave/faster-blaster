/**
 * @file plugin_registry.c
 * @brief Plugin discovery and management
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "../../include/faster-blaster/backend_plugin.h"
#include "../../include/faster-blaster/vtable_autofill.h"
#include "../backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void fb_register_clblast_plugin(void);
extern void fb_register_clblas_plugin(void);


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
  fb_lib_handle_t best_lib_handle = NULL;

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
      best_lib_handle = NULL; /* Plugin init will handle loading */
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

void fb_init_plugins(void) {
  /* Register all built-in CPU plugins */
  fb_register_aocl_plugin();
  fb_register_standard_blis_plugin();
  fb_register_openblas_plugin();
  fb_register_mkl_plugin();
  fb_register_accelerate_plugin();

  /* Register all built-in GPU plugins */
  /* fb_register_cublas_plugin(); */ /* Disabled - using pure dynamic loading
                                        via dispatch system */
  fb_register_rocblas_plugin();
  fb_register_onemkl_plugin();
  fb_register_metal_plugin();
  fb_register_clblast_plugin();
  fb_register_clblas_plugin();

  /* Reference backend: correctness oracle and last-resort fallback.
   * Always available (statically linked); deliberately lowest score. */
  fb_register_reference_plugin();
}
