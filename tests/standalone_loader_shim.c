/**
 * @file standalone_loader_shim.c
 * @brief Minimal runtime loader helpers for standalone test builds.
 */

#include "../include/faster-blaster/backend_plugin.h"

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#define FB_STANDALONE_LOAD_LIBRARY(path) LoadLibraryA(path)
#define FB_STANDALONE_GET_PROC_ADDRESS(handle, name)                          \
  GetProcAddress((HMODULE)(handle), name)
#define FB_STANDALONE_FREE_LIBRARY(handle) FreeLibrary((HMODULE)(handle))
#else
#include <dlfcn.h>
#define FB_STANDALONE_LOAD_LIBRARY(path) dlopen(path, RTLD_LAZY)
#define FB_STANDALONE_GET_PROC_ADDRESS(handle, name) dlsym(handle, name)
#define FB_STANDALONE_FREE_LIBRARY(handle) dlclose(handle)
#endif

fb_lib_handle_t fb_plugin_load_library(const char **library_names,
                                       const char **search_paths) {
  if (!library_names) {
    return NULL;
  }

  for (const char **name_ptr = library_names; *name_ptr; ++name_ptr) {
    const char *name = *name_ptr;

    if (search_paths) {
      for (const char **path_ptr = search_paths; *path_ptr; ++path_ptr) {
        char full_path[1024];
        int written = snprintf(full_path, sizeof(full_path), "%s/%s",
                               *path_ptr, name);
        if (written > 0 && (size_t)written < sizeof(full_path)) {
          fb_lib_handle_t handle = FB_STANDALONE_LOAD_LIBRARY(full_path);
          if (handle) {
            return handle;
          }
        }
      }
    }

    fb_lib_handle_t handle = FB_STANDALONE_LOAD_LIBRARY(name);
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

  return FB_STANDALONE_GET_PROC_ADDRESS(lib_handle, symbol_name);
}

void fb_plugin_unload_library(fb_lib_handle_t lib_handle) {
  if (lib_handle) {
    FB_STANDALONE_FREE_LIBRARY(lib_handle);
  }
}