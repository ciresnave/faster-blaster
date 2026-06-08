#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASV2_TESTS(SUFFIX, TYPE, OP_ID, BASE)                             \
typedef int (*fb_##SUFFIX##_cblas_fn)(TYPE f, TYPE g, TYPE h, TYPE *ssmin,      \
                                      TYPE *ssmax, TYPE *snr, TYPE *csr,        \
                                      TYPE *sn, TYPE *cs);                      \
typedef void (*fb_##SUFFIX##_fortran_fn)(TYPE *f, TYPE *g, TYPE *h, TYPE *ssmin,\
                                         TYPE *ssmax, TYPE *snr, TYPE *csr,     \
                                         TYPE *sn, TYPE *cs);                   \
static struct {                                                                    \
    int called;                                                                    \
    TYPE f;                                                                        \
    TYPE g;                                                                        \
    TYPE h;                                                                        \
    TYPE *ssmin;                                                                   \
    TYPE *ssmax;                                                                   \
    TYPE *snr;                                                                     \
    TYPE *csr;                                                                     \
    TYPE *sn;                                                                      \
    TYPE *cs;                                                                      \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    TYPE f;                                                                        \
    TYPE g;                                                                        \
    TYPE h;                                                                        \
    TYPE *ssmin;                                                                   \
    TYPE *ssmax;                                                                   \
    TYPE *snr;                                                                     \
    TYPE *csr;                                                                     \
    TYPE *sn;                                                                      \
    TYPE *cs;                                                                      \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(TYPE *f, TYPE *g, TYPE *h, TYPE *ssmin,     \
                                    TYPE *ssmax, TYPE *snr, TYPE *csr,          \
                                    TYPE *sn, TYPE *cs)                         \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.f = *f;                                             \
    g_##SUFFIX##_fortran_call.g = *g;                                             \
    g_##SUFFIX##_fortran_call.h = *h;                                             \
    g_##SUFFIX##_fortran_call.ssmin = ssmin;                                      \
    g_##SUFFIX##_fortran_call.ssmax = ssmax;                                      \
    g_##SUFFIX##_fortran_call.snr = snr;                                          \
    g_##SUFFIX##_fortran_call.csr = csr;                                          \
    g_##SUFFIX##_fortran_call.sn = sn;                                            \
    g_##SUFFIX##_fortran_call.cs = cs;                                            \
    *ssmin = (TYPE)((BASE) + 1);                                                  \
    *ssmax = (TYPE)((BASE) + 2);                                                  \
    *snr = (TYPE)((BASE) + 3);                                                    \
    *csr = (TYPE)((BASE) + 4);                                                    \
    *sn = (TYPE)((BASE) + 5);                                                     \
    *cs = (TYPE)((BASE) + 6);                                                     \
}                                                                                 \
static int stub_##SUFFIX##_cblas(TYPE f, TYPE g, TYPE h, TYPE *ssmin,           \
                                 TYPE *ssmax, TYPE *snr, TYPE *csr, TYPE *sn,   \
                                 TYPE *cs)                                       \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.f = f;                                                \
    g_##SUFFIX##_cblas_call.g = g;                                                \
    g_##SUFFIX##_cblas_call.h = h;                                                \
    g_##SUFFIX##_cblas_call.ssmin = ssmin;                                        \
    g_##SUFFIX##_cblas_call.ssmax = ssmax;                                        \
    g_##SUFFIX##_cblas_call.snr = snr;                                            \
    g_##SUFFIX##_cblas_call.csr = csr;                                            \
    g_##SUFFIX##_cblas_call.sn = sn;                                              \
    g_##SUFFIX##_cblas_call.cs = cs;                                              \
    *ssmin = (TYPE)((BASE) + 7);                                                  \
    *ssmax = (TYPE)((BASE) + 8);                                                  \
    *snr = (TYPE)((BASE) + 9);                                                    \
    *csr = (TYPE)((BASE) + 10);                                                   \
    *sn = (TYPE)((BASE) + 11);                                                    \
    *cs = (TYPE)((BASE) + 12);                                                    \
    return (BASE) + 99;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ssmin = (TYPE)0;                                                         \
    TYPE ssmax = (TYPE)0;                                                         \
    TYPE snr = (TYPE)0;                                                           \
    TYPE csr = (TYPE)0;                                                           \
    TYPE sn = (TYPE)0;                                                            \
    TYPE cs = (TYPE)0;                                                            \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    if (thunk((TYPE)((BASE) + 10), (TYPE)((BASE) + 20), (TYPE)((BASE) + 30),    \
              &ssmin, &ssmax, &snr, &csr, &sn, &cs) != 0 ||                      \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.f != (TYPE)((BASE) + 10) ||                    \
        g_##SUFFIX##_fortran_call.g != (TYPE)((BASE) + 20) ||                    \
        g_##SUFFIX##_fortran_call.h != (TYPE)((BASE) + 30) ||                    \
        g_##SUFFIX##_fortran_call.ssmin != &ssmin ||                              \
        g_##SUFFIX##_fortran_call.ssmax != &ssmax ||                              \
        g_##SUFFIX##_fortran_call.snr != &snr ||                                  \
        g_##SUFFIX##_fortran_call.csr != &csr ||                                  \
        g_##SUFFIX##_fortran_call.sn != &sn ||                                    \
        g_##SUFFIX##_fortran_call.cs != &cs ||                                    \
        ssmin != (TYPE)((BASE) + 1) || ssmax != (TYPE)((BASE) + 2) ||           \
        snr != (TYPE)((BASE) + 3) || csr != (TYPE)((BASE) + 4) ||               \
        sn != (TYPE)((BASE) + 5) || cs != (TYPE)((BASE) + 6)) {                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASV2 scalar inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASV2 scalar inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    TYPE f = (TYPE)((BASE) + 40);                                                 \
    TYPE g = (TYPE)((BASE) + 50);                                                 \
    TYPE h = (TYPE)((BASE) + 60);                                                 \
    TYPE ssmin = (TYPE)0;                                                         \
    TYPE ssmax = (TYPE)0;                                                         \
    TYPE snr = (TYPE)0;                                                           \
    TYPE csr = (TYPE)0;                                                           \
    TYPE sn = (TYPE)0;                                                            \
    TYPE cs = (TYPE)0;                                                            \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&f, &g, &h, &ssmin, &ssmax, &snr, &csr, &sn, &cs);                     \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.f != (TYPE)((BASE) + 40) ||                      \
        g_##SUFFIX##_cblas_call.g != (TYPE)((BASE) + 50) ||                      \
        g_##SUFFIX##_cblas_call.h != (TYPE)((BASE) + 60) ||                      \
        g_##SUFFIX##_cblas_call.ssmin != &ssmin ||                                \
        g_##SUFFIX##_cblas_call.ssmax != &ssmax ||                                \
        g_##SUFFIX##_cblas_call.snr != &snr ||                                    \
        g_##SUFFIX##_cblas_call.csr != &csr ||                                    \
        g_##SUFFIX##_cblas_call.sn != &sn ||                                      \
        g_##SUFFIX##_cblas_call.cs != &cs ||                                      \
        ssmin != (TYPE)((BASE) + 7) || ssmax != (TYPE)((BASE) + 8) ||           \
        snr != (TYPE)((BASE) + 9) || csr != (TYPE)((BASE) + 10) ||              \
        sn != (TYPE)((BASE) + 11) || cs != (TYPE)((BASE) + 12)) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASV2 scalar inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASV2 scalar inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LASV2_TESTS(slasv2, float, FB_OP_SLASV2, 100)
DEFINE_LASV2_TESTS(dlasv2, double, FB_OP_DLASV2, 300)

int main(void)
{
    int status = 0;

    status |= check_slasv2_fortran_to_cblas();
    status |= check_slasv2_cblas_to_fortran();
    status |= check_dlasv2_fortran_to_cblas();
    status |= check_dlasv2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}