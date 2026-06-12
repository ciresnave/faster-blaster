/**
 * @file test_plugin_architecture.c
 * @brief Test the plugin architecture
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include "../src/backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(void) {
    printf("Plugin Architecture Test\n");
    printf("========================\n\n");
    
    /* Initialize plugin system */
    printf("Initializing plugin system...\n");
    fb_init_plugins();
    printf("Plugin system initialized\n\n");
    
    /* List all registered plugins */
    printf("Registered Plugins:\n");
    const fb_plugin_registry_entry_t* entry = fb_get_registered_plugins();
    int plugin_count = 0;
    
    while (entry) {
        const fb_backend_plugin_t* plugin = entry->plugin;
        printf("  %d. %s v%s (%s)\n", 
               ++plugin_count,
               plugin->metadata->name,
               plugin->metadata->version,
               plugin->metadata->vendor);
        printf("     %s\n", plugin->metadata->description);
        printf("     Capabilities: 0x%08X\n\n", plugin->metadata->capabilities);
        entry = entry->next;
    }
    
    if (plugin_count == 0) {
        printf("  ERROR: No plugins registered!\n");
        return 1;
    }
    
    printf("Total plugins registered: %d\n\n", plugin_count);
    
    /* Try to load the best plugin */
    printf("Attempting to load best plugin...\n");
    fb_plugin_context_t* ctx = NULL;
    const fb_backend_plugin_t* best_plugin = fb_load_best_plugin(NULL, NULL, &ctx);
    
    if (!best_plugin) {
        printf("  WARNING: No compatible plugin found\n");
        printf("  This is expected if no BLAS libraries are installed\n");
        return 0; /* Not an error - just no BLAS libs available */
    }
    
    printf("  Loaded: %s v%s\n", 
           best_plugin->metadata->name,
           best_plugin->metadata->version);
    
    /* Get vtable and verify it's valid */
    const fb_backend_vtable_t* vtable = best_plugin->get_vtable(ctx);
    if (!vtable) {
        printf("  ERROR: Failed to get vtable from plugin\n");
        return 1;
    }
    
    printf("  Vtable obtained successfully\n");
    
    /* Check for basic Level 1 functions */
    int level1_functions = 0;
    if (vtable->saxpy) level1_functions++;
    if (vtable->sdot) level1_functions++;
    if (vtable->snrm2) level1_functions++;
    if (vtable->sasum) level1_functions++;
    if (vtable->scopy) level1_functions++;
    if (vtable->sscal) level1_functions++;
    if (vtable->sswap) level1_functions++;
    if (vtable->isamax) level1_functions++;
    
    printf("  Level 1 functions available: %d/8\n", level1_functions);
    
    /* Test a basic operation if available */
    if (vtable->sdot) {
        float x[] = {1.0f, 2.0f, 3.0f};
        float y[] = {4.0f, 5.0f, 6.0f};
        
        float result = vtable->sdot(3, x, 1, y, 1);
        float expected = 1.0f*4.0f + 2.0f*5.0f + 3.0f*6.0f; /* 32.0 */
        
        printf("\n  Testing sdot([1,2,3], [4,5,6]):\n");
        printf("    Result: %.1f\n", result);
        printf("    Expected: %.1f\n", expected);
        
        if (result > expected - 0.01f && result < expected + 0.01f) {
            printf("    PASS\n");
        } else {
            printf("    FAIL\n");
            return 1;
        }
    }
    
    /* Cleanup */
    if (best_plugin->shutdown) {
        best_plugin->shutdown(ctx);
    }
    
    printf("\nPlugin architecture test PASSED\n");
    return 0;
}
