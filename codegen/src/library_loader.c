/**
 * Runtime library loading implementation
 */

#include "library_loader.h"
#include "logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
    #include <windows.h>
    #define DLL_LOAD(path) LoadLibraryA(path)
    #define DLL_FREE(h) FreeLibrary(h)
    #define DLL_GETFUNC(h, name) GetProcAddress(h, name)
    #define DLL_ERROR() "LoadLibrary failed"
#else
    #include <dlfcn.h>
    #define DLL_LOAD(path) dlopen(path, RTLD_LAZY)
    #define DLL_FREE(h) dlclose(h)
    #define DLL_GETFUNC(h, name) dlsym(h, name)
    #define DLL_ERROR() dlerror()
#endif

static BackendLibrary g_discovered_backends[32];
static int g_backend_count = 0;

bool library_loader_init(void) {
    return true;
}

void* library_loader_load_backend(const char* backend_name) {
    /* Simple stub - would load from manifest in production */
    log_info("Library loader: backend=%s (stub implementation)", backend_name);
    return NULL;
}

void* library_loader_get_function(void* handle, const char* func_name) {
    if (!handle) return NULL;
    return DLL_GETFUNC(handle, func_name);
}

bool library_loader_unload_backend(void* handle) {
    if (!handle) return false;
    #ifdef _WIN32
        return FreeLibrary(handle) != 0;
    #else
        return dlclose(handle) == 0;
    #endif
}

const BackendLibrary* library_loader_get_backends(int* count) {
    *count = g_backend_count;
    return g_discovered_backends;
}
