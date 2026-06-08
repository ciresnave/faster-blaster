/**
 * @file test_reference_loader_alias_regression.c
 * @brief Regression checks for plain LAPACK aliases with underscore twins.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../include/faster-blaster/backend_plugin.h"
#include "../src/backends/reference.h"

#ifndef FB_REFERENCE_DLL_DIR
#define FB_REFERENCE_DLL_DIR "../faster-blaster-reference/build-extended"
#endif

extern uint32_t fb_stem_to_op_id(const char *stem);

static void *fb_fn_to_symbol(fb_generic_fn fn)
{
    void *symbol = NULL;
    size_t copy_size = sizeof(symbol) < sizeof(fn) ? sizeof(symbol) : sizeof(fn);
    memcpy(&symbol, &fn, copy_size);
    return symbol;
}

static void *fb_ptr_from_bits(const void *storage, size_t storage_size)
{
    void *symbol = NULL;
    size_t copy_size = sizeof(symbol) < storage_size ? sizeof(symbol) : storage_size;
    memcpy(&symbol, storage, copy_size);
    return symbol;
}

#define FB_PTR_FROM_FIELD(vtable_ptr, field) \
    fb_ptr_from_bits(&(vtable_ptr)->field, sizeof((vtable_ptr)->field))

static fb_lib_handle_t load_reference_library(void)
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

    return fb_plugin_load_library(ref_names, ref_paths);
}

static int check_plain_alias_twin(const fb_backend_vtable_t *vtable,
                                  fb_lib_handle_t ref_h,
                                  const char *stem)
{
    char plain_name[32] = { 0 };
    char fortran_name[33] = { 0 };
    void *plain_export = NULL;
    void *fortran_export = NULL;
    void *cblas_slot = NULL;
    void *fortran_slot = NULL;
    uint32_t op_id = 0;

    if (!vtable || !ref_h || !stem || stem[0] == '\0') {
        fprintf(stderr, "[FAIL] Invalid alias regression inputs for %s\n",
                stem ? stem : "<null>");
        return 1;
    }

    if (snprintf(plain_name, sizeof(plain_name), "%s", stem) <= 0 ||
        snprintf(fortran_name, sizeof(fortran_name), "%s_", stem) <= 0) {
        fprintf(stderr, "[FAIL] Could not format export names for %s\n", stem);
        return 1;
    }

    plain_export = fb_plugin_get_symbol(ref_h, plain_name);
    fortran_export = fb_plugin_get_symbol(ref_h, fortran_name);
    if (!plain_export || !fortran_export) {
        fprintf(stderr,
                "[FAIL] Expected both plain and underscore exports for %s "
                "(plain=%p, underscore=%p)\n",
                stem,
                plain_export,
                fortran_export);
        return 1;
    }

    op_id = fb_stem_to_op_id(stem);
    if (op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) {
        fprintf(stderr, "[FAIL] Unknown op_id for %s\n", stem);
        return 1;
    }

    cblas_slot = fb_fn_to_symbol(vtable->ext_ops[op_id][FB_CONV_CBLAS]);
    fortran_slot = fb_fn_to_symbol(vtable->ext_ops[op_id][FB_CONV_FORTRAN]);

    if (fortran_slot != plain_export && fortran_slot != fortran_export) {
        fprintf(stderr,
                "[FAIL] %s raw export did not land in the Fortran slot "
                "(fortran=%p, plain=%p, underscore=%p)\n",
                stem,
                fortran_slot,
                plain_export,
                fortran_export);
        return 1;
    }

    if (cblas_slot == plain_export || cblas_slot == fortran_export) {
        fprintf(stderr,
                "[FAIL] %s raw export was stored directly in the CBLAS slot "
                "(cblas=%p, plain=%p, underscore=%p)\n",
                stem,
                cblas_slot,
                plain_export,
                fortran_export);
        return 1;
    }

    printf("[PASS] %s plain/underscore twin kept out of raw CBLAS slot\n", stem);
    return 0;
}

static int check_fortran_only_export_binding(const fb_backend_vtable_t *vtable,
                                            fb_lib_handle_t ref_h,
                                            const char *stem,
                                            void *named_field)
{
    char plain_name[32] = { 0 };
    char lapacke_name[40] = { 0 };
    char fortran_name[33] = { 0 };
    void *plain_export = NULL;
    void *lapacke_export = NULL;
    void *fortran_export = NULL;
    void *cblas_slot = NULL;
    void *fortran_slot = NULL;
    uint32_t op_id = 0;

    if (!vtable || !ref_h || !stem || stem[0] == '\0') {
        fprintf(stderr, "[FAIL] Invalid Fortran-only binding inputs for %s\n",
                stem ? stem : "<null>");
        return 1;
    }

    if (snprintf(plain_name, sizeof(plain_name), "%s", stem) <= 0 ||
        snprintf(lapacke_name, sizeof(lapacke_name), "LAPACKE_%s", stem) <= 0 ||
        snprintf(fortran_name, sizeof(fortran_name), "%s_", stem) <= 0) {
        fprintf(stderr, "[FAIL] Could not format export names for %s\n", stem);
        return 1;
    }

    plain_export = fb_plugin_get_symbol(ref_h, plain_name);
    lapacke_export = fb_plugin_get_symbol(ref_h, lapacke_name);
    fortran_export = fb_plugin_get_symbol(ref_h, fortran_name);

    if (plain_export != NULL || lapacke_export != NULL || fortran_export == NULL) {
        fprintf(stderr,
                "[FAIL] Expected underscore-only export pattern for %s "
                "(plain=%p, lapacke=%p, underscore=%p)\n",
                stem,
                plain_export,
                lapacke_export,
                fortran_export);
        return 1;
    }

    op_id = fb_stem_to_op_id(stem);
    if (op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) {
        fprintf(stderr, "[FAIL] Unknown op_id for %s\n", stem);
        return 1;
    }

    cblas_slot = fb_fn_to_symbol(vtable->ext_ops[op_id][FB_CONV_CBLAS]);
    fortran_slot = fb_fn_to_symbol(vtable->ext_ops[op_id][FB_CONV_FORTRAN]);

    if (fortran_slot != fortran_export) {
        fprintf(stderr,
                "[FAIL] %s underscore export did not populate the Fortran slot "
                "(fortran=%p, underscore=%p)\n",
                stem,
                fortran_slot,
                fortran_export);
        return 1;
    }

    if (cblas_slot == NULL) {
        fprintf(stderr,
                "[FAIL] %s did not gain a generated C bridge from its underscore-only export\n",
                stem);
        return 1;
    }

    if (cblas_slot == fortran_export) {
        fprintf(stderr,
                "[FAIL] %s underscore export was stored directly in the CBLAS slot "
                "(cblas=%p, underscore=%p)\n",
                stem,
                cblas_slot,
                fortran_export);
        return 1;
    }

    if (named_field == NULL) {
        fprintf(stderr,
                "[FAIL] %s did not hydrate the named C field from its generated bridge\n",
                stem);
        return 1;
    }

    if (named_field == fortran_export) {
        fprintf(stderr,
                "[FAIL] %s underscore export was stored directly in the named C field "
                "(named=%p, underscore=%p)\n",
                stem,
                named_field,
                fortran_export);
        return 1;
    }

    printf("[PASS] %s gained a generated C bridge from %s\n",
           stem,
           fortran_name);
    return 0;
}

int main(void)
{
    fb_lib_handle_t ref_h = load_reference_library();
    const fb_backend_vtable_t *vtable = NULL;

    if (!ref_h) {
        fprintf(stderr,
                "[FATAL] Reference DLL not found. Tried '%s' and current directory.\n",
                FB_REFERENCE_DLL_DIR);
        return 1;
    }

    fb_reference_init(ref_h);

    vtable = fb_reference_backend();
    if (!vtable) {
        fprintf(stderr, "[FATAL] fb_reference_backend() returned NULL\n");
        return 1;
    }

    if (check_plain_alias_twin(vtable, ref_h, "cgelsy") != 0) {
        return 1;
    }
    if (check_plain_alias_twin(vtable, ref_h, "zgelsy") != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "ssyev",
                                          FB_PTR_FROM_FIELD(vtable, ssyev)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgeev",
                                          FB_PTR_FROM_FIELD(vtable, sgeev)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgesdd",
                                          FB_PTR_FROM_FIELD(vtable, sgesdd)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "ssygv",
                                          FB_PTR_FROM_FIELD(vtable, ssygv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgebrd",
                                          FB_PTR_FROM_FIELD(vtable, sgebrd)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgebrd",
                                          FB_PTR_FROM_FIELD(vtable, dgebrd)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgebrd",
                                          FB_PTR_FROM_FIELD(vtable, cgebrd)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgebrd",
                                          FB_PTR_FROM_FIELD(vtable, zgebrd)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgehd2",
                                          FB_PTR_FROM_FIELD(vtable, sgehd2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgehd2",
                                          FB_PTR_FROM_FIELD(vtable, dgehd2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgehd2",
                                          FB_PTR_FROM_FIELD(vtable, cgehd2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgehd2",
                                          FB_PTR_FROM_FIELD(vtable, zgehd2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgehrd",
                                          FB_PTR_FROM_FIELD(vtable, sgehrd)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgehrd",
                                          FB_PTR_FROM_FIELD(vtable, dgehrd)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgehrd",
                                          FB_PTR_FROM_FIELD(vtable, cgehrd)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgehrd",
                                          FB_PTR_FROM_FIELD(vtable, zgehrd)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgelq2",
                                          FB_PTR_FROM_FIELD(vtable, sgelq2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgelq2",
                                          FB_PTR_FROM_FIELD(vtable, dgelq2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgelq2",
                                          FB_PTR_FROM_FIELD(vtable, cgelq2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgelq2",
                                          FB_PTR_FROM_FIELD(vtable, zgelq2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgeql2",
                                          FB_PTR_FROM_FIELD(vtable, sgeql2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgeql2",
                                          FB_PTR_FROM_FIELD(vtable, dgeql2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgeql2",
                                          FB_PTR_FROM_FIELD(vtable, cgeql2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgeql2",
                                          FB_PTR_FROM_FIELD(vtable, zgeql2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgerq2",
                                          FB_PTR_FROM_FIELD(vtable, sgerq2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgerq2",
                                          FB_PTR_FROM_FIELD(vtable, dgerq2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgerq2",
                                          FB_PTR_FROM_FIELD(vtable, cgerq2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgerq2",
                                          FB_PTR_FROM_FIELD(vtable, zgerq2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgeequ",
                                          FB_PTR_FROM_FIELD(vtable, sgeequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgeequ",
                                          FB_PTR_FROM_FIELD(vtable, dgeequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgecon",
                                          FB_PTR_FROM_FIELD(vtable, sgecon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgecon",
                                          FB_PTR_FROM_FIELD(vtable, dgecon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgecon",
                                          FB_PTR_FROM_FIELD(vtable, cgecon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgecon",
                                          FB_PTR_FROM_FIELD(vtable, zgecon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sposv",
                                          FB_PTR_FROM_FIELD(vtable, sposv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dposv",
                                          FB_PTR_FROM_FIELD(vtable, dposv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cposv",
                                          FB_PTR_FROM_FIELD(vtable, cposv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zposv",
                                          FB_PTR_FROM_FIELD(vtable, zposv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "ssysv",
                                          FB_PTR_FROM_FIELD(vtable, ssysv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dsysv",
                                          FB_PTR_FROM_FIELD(vtable, dsysv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "chesv",
                                          FB_PTR_FROM_FIELD(vtable, chesv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zhesv",
                                          FB_PTR_FROM_FIELD(vtable, zhesv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgeqrf",
                                          FB_PTR_FROM_FIELD(vtable, sgeqrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgeqrf",
                                          FB_PTR_FROM_FIELD(vtable, dgeqrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgeqrf",
                                          FB_PTR_FROM_FIELD(vtable, cgeqrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgeqrf",
                                          FB_PTR_FROM_FIELD(vtable, zgeqrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgelqf",
                                          FB_PTR_FROM_FIELD(vtable, sgelqf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgelqf",
                                          FB_PTR_FROM_FIELD(vtable, dgelqf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgelqf",
                                          FB_PTR_FROM_FIELD(vtable, cgelqf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgelqf",
                                          FB_PTR_FROM_FIELD(vtable, zgelqf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgeqlf",
                                          FB_PTR_FROM_FIELD(vtable, sgeqlf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgeqlf",
                                          FB_PTR_FROM_FIELD(vtable, dgeqlf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgeqlf",
                                          FB_PTR_FROM_FIELD(vtable, cgeqlf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgeqlf",
                                          FB_PTR_FROM_FIELD(vtable, zgeqlf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgerqf",
                                          FB_PTR_FROM_FIELD(vtable, sgerqf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgerqf",
                                          FB_PTR_FROM_FIELD(vtable, dgerqf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgerqf",
                                          FB_PTR_FROM_FIELD(vtable, cgerqf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgerqf",
                                          FB_PTR_FROM_FIELD(vtable, zgerqf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorgqr",
                                          FB_PTR_FROM_FIELD(vtable, sorgqr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorgqr",
                                          FB_PTR_FROM_FIELD(vtable, dorgqr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cungqr",
                                          FB_PTR_FROM_FIELD(vtable, cungqr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zungqr",
                                          FB_PTR_FROM_FIELD(vtable, zungqr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorglq",
                                          FB_PTR_FROM_FIELD(vtable, sorglq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorglq",
                                          FB_PTR_FROM_FIELD(vtable, dorglq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunglq",
                                          FB_PTR_FROM_FIELD(vtable, cunglq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunglq",
                                          FB_PTR_FROM_FIELD(vtable, zunglq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorgql",
                                          FB_PTR_FROM_FIELD(vtable, sorgql)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorgql",
                                          FB_PTR_FROM_FIELD(vtable, dorgql)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cungql",
                                          FB_PTR_FROM_FIELD(vtable, cungql)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zungql",
                                          FB_PTR_FROM_FIELD(vtable, zungql)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorgrq",
                                          FB_PTR_FROM_FIELD(vtable, sorgrq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorgrq",
                                          FB_PTR_FROM_FIELD(vtable, dorgrq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cungrq",
                                          FB_PTR_FROM_FIELD(vtable, cungrq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zungrq",
                                          FB_PTR_FROM_FIELD(vtable, zungrq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sormqr",
                                          FB_PTR_FROM_FIELD(vtable, sormqr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dormqr",
                                          FB_PTR_FROM_FIELD(vtable, dormqr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunmqr",
                                          FB_PTR_FROM_FIELD(vtable, cunmqr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunmqr",
                                          FB_PTR_FROM_FIELD(vtable, zunmqr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sormlq",
                                          FB_PTR_FROM_FIELD(vtable, sormlq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dormlq",
                                          FB_PTR_FROM_FIELD(vtable, dormlq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunmlq",
                                          FB_PTR_FROM_FIELD(vtable, cunmlq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunmlq",
                                          FB_PTR_FROM_FIELD(vtable, zunmlq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sormql",
                                          FB_PTR_FROM_FIELD(vtable, sormql)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dormql",
                                          FB_PTR_FROM_FIELD(vtable, dormql)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunmql",
                                          FB_PTR_FROM_FIELD(vtable, cunmql)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunmql",
                                          FB_PTR_FROM_FIELD(vtable, zunmql)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sormrq",
                                          FB_PTR_FROM_FIELD(vtable, sormrq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dormrq",
                                          FB_PTR_FROM_FIELD(vtable, dormrq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunmrq",
                                          FB_PTR_FROM_FIELD(vtable, cunmrq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunmrq",
                                          FB_PTR_FROM_FIELD(vtable, zunmrq)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sormr3",
                                          FB_PTR_FROM_FIELD(vtable, sormr3)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dormr3",
                                          FB_PTR_FROM_FIELD(vtable, dormr3)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunmr3",
                                          FB_PTR_FROM_FIELD(vtable, cunmr3)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunmr3",
                                          FB_PTR_FROM_FIELD(vtable, zunmr3)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sormrz",
                                          FB_PTR_FROM_FIELD(vtable, sormrz)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dormrz",
                                          FB_PTR_FROM_FIELD(vtable, dormrz)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunmrz",
                                          FB_PTR_FROM_FIELD(vtable, cunmrz)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunmrz",
                                          FB_PTR_FROM_FIELD(vtable, zunmrz)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgbmvx",
                                          FB_PTR_FROM_FIELD(vtable, sgbmvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgbmvx",
                                          FB_PTR_FROM_FIELD(vtable, dgbmvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgbmvx",
                                          FB_PTR_FROM_FIELD(vtable, cgbmvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgbmvx",
                                          FB_PTR_FROM_FIELD(vtable, zgbmvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorg2l",
                                          FB_PTR_FROM_FIELD(vtable, sorg2l)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorg2l",
                                          FB_PTR_FROM_FIELD(vtable, dorg2l)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cung2l",
                                          FB_PTR_FROM_FIELD(vtable, cung2l)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zung2l",
                                          FB_PTR_FROM_FIELD(vtable, zung2l)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorgl2",
                                          FB_PTR_FROM_FIELD(vtable, sorgl2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorgl2",
                                          FB_PTR_FROM_FIELD(vtable, dorgl2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cungl2",
                                          FB_PTR_FROM_FIELD(vtable, cungl2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zungl2",
                                          FB_PTR_FROM_FIELD(vtable, zungl2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorgr2",
                                          FB_PTR_FROM_FIELD(vtable, sorgr2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorgr2",
                                          FB_PTR_FROM_FIELD(vtable, dorgr2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cungr2",
                                          FB_PTR_FROM_FIELD(vtable, cungr2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zungr2",
                                          FB_PTR_FROM_FIELD(vtable, zungr2)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorgtr",
                                          FB_PTR_FROM_FIELD(vtable, sorgtr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorgtr",
                                          FB_PTR_FROM_FIELD(vtable, dorgtr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cungtr",
                                          FB_PTR_FROM_FIELD(vtable, cungtr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zungtr",
                                          FB_PTR_FROM_FIELD(vtable, zungtr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorghr",
                                          FB_PTR_FROM_FIELD(vtable, sorghr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorghr",
                                          FB_PTR_FROM_FIELD(vtable, dorghr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunghr",
                                          FB_PTR_FROM_FIELD(vtable, cunghr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunghr",
                                          FB_PTR_FROM_FIELD(vtable, zunghr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sorgbr",
                                          FB_PTR_FROM_FIELD(vtable, sorgbr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dorgbr",
                                          FB_PTR_FROM_FIELD(vtable, dorgbr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cungbr",
                                          FB_PTR_FROM_FIELD(vtable, cungbr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zungbr",
                                          FB_PTR_FROM_FIELD(vtable, zungbr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sormbr",
                                          FB_PTR_FROM_FIELD(vtable, sormbr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dormbr",
                                          FB_PTR_FROM_FIELD(vtable, dormbr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunmbr",
                                          FB_PTR_FROM_FIELD(vtable, cunmbr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunmbr",
                                          FB_PTR_FROM_FIELD(vtable, zunmbr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sormhr",
                                          FB_PTR_FROM_FIELD(vtable, sormhr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dormhr",
                                          FB_PTR_FROM_FIELD(vtable, dormhr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cunmhr",
                                          FB_PTR_FROM_FIELD(vtable, cunmhr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zunmhr",
                                          FB_PTR_FROM_FIELD(vtable, zunmhr)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgetrf",
                                          FB_PTR_FROM_FIELD(vtable, sgetrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgetrf",
                                          FB_PTR_FROM_FIELD(vtable, dgetrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgetrf",
                                          FB_PTR_FROM_FIELD(vtable, cgetrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgetrf",
                                          FB_PTR_FROM_FIELD(vtable, zgetrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgetrs",
                                          FB_PTR_FROM_FIELD(vtable, sgetrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgetrs",
                                          FB_PTR_FROM_FIELD(vtable, dgetrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgetrs",
                                          FB_PTR_FROM_FIELD(vtable, cgetrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgetrs",
                                          FB_PTR_FROM_FIELD(vtable, zgetrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spotrf",
                                          FB_PTR_FROM_FIELD(vtable, spotrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpotrf",
                                          FB_PTR_FROM_FIELD(vtable, dpotrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpotrf",
                                          FB_PTR_FROM_FIELD(vtable, cpotrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpotrf",
                                          FB_PTR_FROM_FIELD(vtable, zpotrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spotri",
                                          FB_PTR_FROM_FIELD(vtable, spotri)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpotri",
                                          FB_PTR_FROM_FIELD(vtable, dpotri)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpotri",
                                          FB_PTR_FROM_FIELD(vtable, cpotri)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpotri",
                                          FB_PTR_FROM_FIELD(vtable, zpotri)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spotrs",
                                          FB_PTR_FROM_FIELD(vtable, spotrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpotrs",
                                          FB_PTR_FROM_FIELD(vtable, dpotrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpotrs",
                                          FB_PTR_FROM_FIELD(vtable, cpotrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpotrs",
                                          FB_PTR_FROM_FIELD(vtable, zpotrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgesv",
                                          FB_PTR_FROM_FIELD(vtable, sgesv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgesv",
                                          FB_PTR_FROM_FIELD(vtable, dgesv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgesv",
                                          FB_PTR_FROM_FIELD(vtable, cgesv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgesv",
                                          FB_PTR_FROM_FIELD(vtable, zgesv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spbequ",
                                          FB_PTR_FROM_FIELD(vtable, spbequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpbequ",
                                          FB_PTR_FROM_FIELD(vtable, dpbequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpbequ",
                                          FB_PTR_FROM_FIELD(vtable, cpbequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpbequ",
                                          FB_PTR_FROM_FIELD(vtable, zpbequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spbcon",
                                          FB_PTR_FROM_FIELD(vtable, spbcon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpbcon",
                                          FB_PTR_FROM_FIELD(vtable, dpbcon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpbcon",
                                          FB_PTR_FROM_FIELD(vtable, cpbcon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpbcon",
                                          FB_PTR_FROM_FIELD(vtable, zpbcon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spbrfs",
                                          FB_PTR_FROM_FIELD(vtable, spbrfs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpbrfs",
                                          FB_PTR_FROM_FIELD(vtable, dpbrfs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpbrfs",
                                          FB_PTR_FROM_FIELD(vtable, cpbrfs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpbrfs",
                                          FB_PTR_FROM_FIELD(vtable, zpbrfs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgbequ",
                                          FB_PTR_FROM_FIELD(vtable, sgbequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgbequ",
                                          FB_PTR_FROM_FIELD(vtable, dgbequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgbequ",
                                          FB_PTR_FROM_FIELD(vtable, cgbequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgbequ",
                                          FB_PTR_FROM_FIELD(vtable, zgbequ)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgbcon",
                                          FB_PTR_FROM_FIELD(vtable, sgbcon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgbcon",
                                          FB_PTR_FROM_FIELD(vtable, dgbcon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgbcon",
                                          FB_PTR_FROM_FIELD(vtable, cgbcon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgbcon",
                                          FB_PTR_FROM_FIELD(vtable, zgbcon)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgbsvx",
                                          FB_PTR_FROM_FIELD(vtable, sgbsvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgbsvx",
                                          FB_PTR_FROM_FIELD(vtable, dgbsvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgbsvx",
                                          FB_PTR_FROM_FIELD(vtable, cgbsvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgbsvx",
                                          FB_PTR_FROM_FIELD(vtable, zgbsvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgbtrf",
                                          FB_PTR_FROM_FIELD(vtable, sgbtrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgbtrf",
                                          FB_PTR_FROM_FIELD(vtable, dgbtrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgbtrf",
                                          FB_PTR_FROM_FIELD(vtable, cgbtrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgbtrf",
                                          FB_PTR_FROM_FIELD(vtable, zgbtrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgbtrs",
                                          FB_PTR_FROM_FIELD(vtable, sgbtrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgbtrs",
                                          FB_PTR_FROM_FIELD(vtable, dgbtrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgbtrs",
                                          FB_PTR_FROM_FIELD(vtable, cgbtrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgbtrs",
                                          FB_PTR_FROM_FIELD(vtable, zgbtrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "sgbsv",
                                          FB_PTR_FROM_FIELD(vtable, sgbsv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dgbsv",
                                          FB_PTR_FROM_FIELD(vtable, dgbsv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cgbsv",
                                          FB_PTR_FROM_FIELD(vtable, cgbsv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zgbsv",
                                          FB_PTR_FROM_FIELD(vtable, zgbsv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spbsv",
                                          FB_PTR_FROM_FIELD(vtable, spbsv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpbsv",
                                          FB_PTR_FROM_FIELD(vtable, dpbsv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpbsv",
                                          FB_PTR_FROM_FIELD(vtable, cpbsv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpbsv",
                                          FB_PTR_FROM_FIELD(vtable, zpbsv)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spbsvx",
                                          FB_PTR_FROM_FIELD(vtable, spbsvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpbsvx",
                                          FB_PTR_FROM_FIELD(vtable, dpbsvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpbsvx",
                                          FB_PTR_FROM_FIELD(vtable, cpbsvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpbsvx",
                                          FB_PTR_FROM_FIELD(vtable, zpbsvx)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spbtrf",
                                          FB_PTR_FROM_FIELD(vtable, spbtrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpbtrf",
                                          FB_PTR_FROM_FIELD(vtable, dpbtrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpbtrf",
                                          FB_PTR_FROM_FIELD(vtable, cpbtrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpbtrf",
                                          FB_PTR_FROM_FIELD(vtable, zpbtrf)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "spbtrs",
                                          FB_PTR_FROM_FIELD(vtable, spbtrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "dpbtrs",
                                          FB_PTR_FROM_FIELD(vtable, dpbtrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "cpbtrs",
                                          FB_PTR_FROM_FIELD(vtable, cpbtrs)) != 0) {
        return 1;
    }
    if (check_fortran_only_export_binding(vtable, ref_h, "zpbtrs",
                                          FB_PTR_FROM_FIELD(vtable, zpbtrs)) != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}