/**
 * @file test_plugin_selection_order.c
 * @brief Regression test for plugin selection order semantics
 *
 * Verifies that fb_load_best_plugin() chooses the first compatible plugin in
 * registry order, not the plugin with the highest probe score.
 *
 * @copyright Copyright (c) 2026
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/backend_plugin.h"
#include "backends/backend_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const fb_backend_vtable_t g_dummy_vtable = {0};

static const fb_backend_plugin_t* g_selected_plugin = NULL;

static fb_plugin_probe_result_t probe_high_score(fb_lib_handle_t lib_handle,
                                                 const char** search_paths) {
    (void)lib_handle;
    (void)search_paths;
    return (fb_plugin_probe_result_t){
        .score = 90,
        .library_path = NULL,
        .reason = "High-score compatible plugin",
    };
}

static int init_high_score(fb_lib_handle_t lib_handle,
                           fb_plugin_context_t** ctx_out) {
    (void)lib_handle;
    if (!ctx_out) return -1;
    *ctx_out = NULL;
    return 0;
}

static const fb_backend_vtable_t *get_vtable_high_score(fb_plugin_context_t* ctx) {
    (void)ctx;
    return &g_dummy_vtable;
}

static const fb_backend_plugin_t g_high_score_plugin = {
    .metadata = &(const fb_plugin_metadata_t){
        .name = "high-score-plugin",
        .version = "1.0.0",
        .vendor = "TestVendor",
        .description = "Plugin with higher compatibility score",
        .api_version = 1,
        .capabilities = FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1,
    },
    .probe = probe_high_score,
    .init = init_high_score,
    .get_vtable = get_vtable_high_score,
    .get_context = NULL,
    .shutdown = NULL,
    .set_num_threads = NULL,
    .get_num_threads = NULL,
};

static fb_plugin_probe_result_t probe_low_score(fb_lib_handle_t lib_handle,
                                                const char** search_paths) {
    (void)lib_handle;
    (void)search_paths;
    return (fb_plugin_probe_result_t){
        .score = 10,
        .library_path = NULL,
        .reason = "Low-score compatible plugin",
    };
}

static int init_low_score(fb_lib_handle_t lib_handle,
                          fb_plugin_context_t** ctx_out) {
    (void)lib_handle;
    if (!ctx_out) return -1;
    *ctx_out = NULL;
    return 0;
}

static const fb_backend_vtable_t *get_vtable_low_score(fb_plugin_context_t* ctx) {
    (void)ctx;
    return &g_dummy_vtable;
}

static const fb_backend_plugin_t g_low_score_plugin = {
    .metadata = &(const fb_plugin_metadata_t){
        .name = "low-score-plugin",
        .version = "1.0.0",
        .vendor = "TestVendor",
        .description = "Plugin with lower compatibility score",
        .api_version = 1,
        .capabilities = FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1,
    },
    .probe = probe_low_score,
    .init = init_low_score,
    .get_vtable = get_vtable_low_score,
    .get_context = NULL,
    .shutdown = NULL,
    .set_num_threads = NULL,
    .get_num_threads = NULL,
};

int main(void) {
    printf("Plugin selection order regression test\n");
    printf("=====================================\n");

    /* Register the high-score plugin first, then the low-score plugin. */
    if (fb_register_plugin(&g_high_score_plugin) != 0) {
        fprintf(stderr, "Failed to register high-score plugin\n");
        return 1;
    }
    if (fb_register_plugin(&g_low_score_plugin) != 0) {
        fprintf(stderr, "Failed to register low-score plugin\n");
        return 1;
    }

    fb_plugin_context_t* ctx = NULL;
    const fb_backend_plugin_t* selected = fb_load_best_plugin(NULL, NULL, &ctx);
    if (!selected) {
        fprintf(stderr, "fb_load_best_plugin() returned NULL\n");
        return 1;
    }

    printf("Selected plugin: %s\n", selected->metadata->name);

    if (strcmp(selected->metadata->name, "low-score-plugin") != 0) {
        fprintf(stderr, "Expected low-score-plugin to win by registry order, got %s\n",
                selected->metadata->name);
        return 1;
    }

    printf("PASS: first compatible plugin chosen regardless of score\n");
    if (selected->shutdown) {
        selected->shutdown(ctx);
    }
    return 0;
}
