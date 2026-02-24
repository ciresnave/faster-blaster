/**
 * @file init.c
 * @brief Library initialization and finalization
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster_blaster.h"
#include "../core/dispatch.h"
#include "../backends/reference.h"
#include <stdio.h>

int fb_init(void) {
    /* Initialize global dispatch table */
    if (fb_dispatch_init_global() != 0) {
        fprintf(stderr, "fb_init: Failed to initialize dispatch table\n");
        return -1;
    }
    
    fb_dispatch_table_t *dispatch = fb_dispatch_global();
    
    /* Register reference backend (always available) */
    if (fb_dispatch_register_backend(dispatch, "reference", fb_reference_backend()) != 0) {
        fprintf(stderr, "fb_init: Failed to register reference backend\n");
        return -1;
    }
    
    /* TODO: Register other backends if available */
    /* Example:
     * #ifdef HAVE_OPENBLAS
     *   fb_dispatch_register_backend(dispatch, "openblas", fb_openblas_backend());
     * #endif
     */
    
    /* Auto-select best backend */
    if (fb_dispatch_auto_select(dispatch) != 0) {
        fprintf(stderr, "fb_init: Failed to select backend\n");
        return -1;
    }
    
    const char *active = fb_dispatch_get_active_backend_name(dispatch);
    printf("faster-blaster initialized with backend: %s\n", active ? active : "none");
    
    return 0;
}

void fb_finalize(void) {
    fb_dispatch_finalize_global();
}

const char *fb_get_version_string(void) {
    return "0.1.0";
}

int fb_get_version(int *major, int *minor, int *patch) {
    if (major) *major = 0;
    if (minor) *minor = 1;
    if (patch) *patch = 0;
    return 0;
}
