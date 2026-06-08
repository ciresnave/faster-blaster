/**
 * @file test_reference_loader_audit.c
 * @brief Full-namespace loader audit for the reference backend.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include <string.h>

#include "../include/faster-blaster/backend_plugin.h"
#include "../src/backends/reference.h"

extern uint32_t fb_get_op_stem_map_count(void);
extern const char *fb_get_op_stem_map_stem(uint32_t index);
extern uint32_t fb_get_op_stem_map_op_id(uint32_t index);

#ifndef FB_REFERENCE_DLL_DIR
#define FB_REFERENCE_DLL_DIR "../faster-blaster-reference/build-extended"
#endif

#define FB_EXPECTED_CANONICAL_OPS 3054u
#define FB_MAX_MISSING_TO_PRINT   128u

static bool fb_has_any_reference_slot(const fb_backend_vtable_t *vtable,
                                      uint32_t op_id,
                                      bool *has_cblas,
                                      bool *has_fortran)
{
    bool cblas = vtable->ext_ops[op_id][FB_CONV_CBLAS] != NULL;
    bool fortran = vtable->ext_ops[op_id][FB_CONV_FORTRAN] != NULL;
    if (has_cblas) {
        *has_cblas = cblas;
    }
    if (has_fortran) {
        *has_fortran = fortran;
    }
    return cblas || fortran;
}

int main(void)
{
#if defined(_WIN32)
    const char *ref_names[] = {
        "faster_blaster_reference.dll",
        "libfaster_blaster_reference.dll",
        NULL
    };
#elif defined(__APPLE__)
    const char *ref_names[] = { "libfaster_blaster_reference.dylib", NULL };
#else
    const char *ref_names[] = { "libfaster_blaster_reference.so", NULL };
#endif
    const char *ref_paths[] = {
        FB_REFERENCE_DLL_DIR,
        ".",
        NULL
    };

    fb_lib_handle_t ref_h = fb_plugin_load_library(ref_names, ref_paths);
    if (!ref_h) {
        fprintf(stderr,
                "[FATAL] Reference DLL not found. Tried '%s' and current directory.\n",
                FB_REFERENCE_DLL_DIR);
        return 1;
    }

    fb_reference_init(ref_h);

    const fb_backend_vtable_t *vtable = fb_reference_backend();
    if (!vtable) {
        fprintf(stderr, "[FATAL] fb_reference_backend() returned NULL\n");
        return 1;
    }

    uint32_t canonical_ops = 0;
    uint32_t visible_ops = 0;
    uint32_t cblas_slots = 0;
    uint32_t fortran_slots = 0;
    uint32_t dual_slots = 0;
    uint32_t missing_ops = 0;
    bool seen_ops[FB_JUDGE_MAX_OPERATIONS] = { false };

    const uint32_t stem_count = fb_get_op_stem_map_count();
    for (uint32_t stem_index = 0; stem_index < stem_count; ++stem_index) {
        uint32_t op_id = fb_get_op_stem_map_op_id(stem_index);
        const char *stem = fb_get_op_stem_map_stem(stem_index);

        if (op_id >= FB_JUDGE_MAX_OPERATIONS || !stem || stem[0] == '\0') {
            continue;
        }
        if (seen_ops[op_id]) {
            continue;
        }
        seen_ops[op_id] = true;

        bool has_cblas = false;
        bool has_fortran = false;

        ++canonical_ops;
        if (fb_has_any_reference_slot(vtable, op_id, &has_cblas, &has_fortran)) {
            ++visible_ops;
        } else {
            if (missing_ops < FB_MAX_MISSING_TO_PRINT) {
                fprintf(stderr, "[MISSING] op_id=%u name=%s\n", op_id, stem);
            }
            ++missing_ops;
        }

        if (has_cblas) {
            ++cblas_slots;
        }
        if (has_fortran) {
            ++fortran_slots;
        }
        if (has_cblas && has_fortran) {
            ++dual_slots;
        }
    }

    if (missing_ops > FB_MAX_MISSING_TO_PRINT) {
        fprintf(stderr,
                "[MISSING] ... %u additional ops omitted from log\n",
                missing_ops - FB_MAX_MISSING_TO_PRINT);
    }

    printf("======================================================\n");
    printf("  faster-blaster Reference Loader Audit\n");
    printf("======================================================\n");
    printf("Canonical ops: %u\n", canonical_ops);
    printf("Visible ops:   %u\n", visible_ops);
    printf("CBLAS slots:   %u\n", cblas_slots);
    printf("Fortran slots: %u\n", fortran_slots);
    printf("Dual slots:    %u\n", dual_slots);
    printf("Missing ops:   %u\n", missing_ops);

    if (canonical_ops != FB_EXPECTED_CANONICAL_OPS) {
        fprintf(stderr,
            "[FAIL] Expected %u canonical ops but found %u in loader stem map\n",
                FB_EXPECTED_CANONICAL_OPS,
                canonical_ops);
        return 1;
    }

    if (missing_ops != 0) {
        fprintf(stderr,
                "[FAIL] Reference loader visibility gap: %u/%u canonical ops missing\n",
                missing_ops,
                canonical_ops);
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}