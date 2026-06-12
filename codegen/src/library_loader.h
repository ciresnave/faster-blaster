/**
 * Dynamic library loader for BLAS/LAPACK implementations
 */

#ifndef LIBRARY_LOADER_H
#define LIBRARY_LOADER_H

#include <stdbool.h>

typedef struct {
    const char* backend_name;
    const char* lib_path;
    void* handle;
    int load_status;
} BackendLibrary;

/**
 * Initialize the library loader
 */
bool library_loader_init(void);

/**
 * Load a specific backend library
 */
void* library_loader_load_backend(const char* backend_name);

/**
 * Get function pointer from loaded backend library
 */
void* library_loader_get_function(void* handle, const char* func_name);

/**
 * Unload backend library
 */
bool library_loader_unload_backend(void* handle);

/**
 * List all discovered backend libraries
 */
const BackendLibrary* library_loader_get_backends(int* count);

#endif
