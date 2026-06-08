#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_SPEVX_TESTS(SUFFIX, TYPE, OP_ID, BASE)                            \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char jobz,          \
                                      char range, fb_uplo_t uplo, int n,      \
                                      TYPE *ap, TYPE vl, TYPE vu, int il,     \
                                      int iu, TYPE abstol, int *m, TYPE *w,   \
                                      TYPE *z, int ldz, int *ifail);          \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *jobz, char *range, char *uplo,  \
                                         int *n, TYPE *ap, TYPE *vl, TYPE *vu, \
                                         int *il, int *iu, TYPE *abstol, int *m,\
                                         TYPE *w, TYPE *z, int *ldz, TYPE *work,\
                                         int *iwork, int *ifail, int *info);   \
static struct {                                                                   \
    int called;                                                                   \
    char jobz;                                                                    \
    char range;                                                                   \
    char uplo;                                                                    \
    int n;                                                                        \
    TYPE ap_snapshot[6];                                                          \
    TYPE vl;                                                                      \
    TYPE vu;                                                                      \
    int il;                                                                       \
    int iu;                                                                       \
    TYPE abstol;                                                                  \
    int *m;                                                                       \
    TYPE *w;                                                                      \
    TYPE *z;                                                                      \
    int ldz;                                                                      \
    int work_nonnull;                                                             \
    int iwork_nonnull;                                                            \
    int *ifail;                                                                   \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char jobz;                                                                    \
    char range;                                                                   \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    TYPE *ap;                                                                     \
    TYPE vl;                                                                      \
    TYPE vu;                                                                      \
    int il;                                                                       \
    int iu;                                                                       \
    TYPE abstol;                                                                  \
    int *m;                                                                       \
    TYPE *w;                                                                      \
    TYPE *z;                                                                      \
    int ldz;                                                                      \
    int *ifail;                                                                   \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *jobz, char *range, char *uplo, int *n,\
                                    TYPE *ap, TYPE *vl, TYPE *vu, int *il,     \
                                    int *iu, TYPE *abstol, int *m, TYPE *w,    \
                                    TYPE *z, int *ldz, TYPE *work, int *iwork, \
                                    int *ifail, int *info)                     \
{                                                                                 \
    memcpy(g_##SUFFIX##_fortran_call.ap_snapshot, ap, (size_t)(*n * (*n + 1) / 2) * sizeof(TYPE)); \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.jobz = *jobz;                                       \
    g_##SUFFIX##_fortran_call.range = *range;                                     \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.vl = *vl;                                           \
    g_##SUFFIX##_fortran_call.vu = *vu;                                           \
    g_##SUFFIX##_fortran_call.il = *il;                                           \
    g_##SUFFIX##_fortran_call.iu = *iu;                                           \
    g_##SUFFIX##_fortran_call.abstol = *abstol;                                   \
    g_##SUFFIX##_fortran_call.m = m;                                              \
    g_##SUFFIX##_fortran_call.w = w;                                              \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                         \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    g_##SUFFIX##_fortran_call.iwork_nonnull = (iwork != NULL);                    \
    g_##SUFFIX##_fortran_call.ifail = ifail;                                      \
    ap[0] = (TYPE)41; ap[1] = (TYPE)42; ap[2] = (TYPE)52;                        \
    ap[3] = (TYPE)43; ap[4] = (TYPE)53; ap[5] = (TYPE)63;                        \
    *m = 2;                                                                       \
    w[0] = (TYPE)((BASE) + 1); w[1] = (TYPE)((BASE) + 2);                        \
    z[0] = (TYPE)101; z[1] = (TYPE)201; z[2] = (TYPE)301;                        \
    z[3] = (TYPE)102; z[4] = (TYPE)202; z[5] = (TYPE)302;                        \
    ifail[0] = 7; ifail[1] = 8; ifail[2] = 9;                                    \
    *info = 0;                                                                    \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char jobz, char range,    \
                                 fb_uplo_t uplo, int n, TYPE *ap, TYPE vl,     \
                                 TYPE vu, int il, int iu, TYPE abstol, int *m, \
                                 TYPE *w, TYPE *z, int ldz, int *ifail)        \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.jobz = jobz;                                          \
    g_##SUFFIX##_cblas_call.range = range;                                        \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.vl = vl;                                              \
    g_##SUFFIX##_cblas_call.vu = vu;                                              \
    g_##SUFFIX##_cblas_call.il = il;                                              \
    g_##SUFFIX##_cblas_call.iu = iu;                                              \
    g_##SUFFIX##_cblas_call.abstol = abstol;                                      \
    g_##SUFFIX##_cblas_call.m = m;                                                \
    g_##SUFFIX##_cblas_call.w = w;                                                \
    g_##SUFFIX##_cblas_call.z = z;                                                \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                            \
    g_##SUFFIX##_cblas_call.ifail = ifail;                                        \
    *m = 2;                                                                       \
    w[0] = (TYPE)((BASE) + 3);                                                    \
    z[0] = (TYPE)((BASE) + 4);                                                    \
    ifail[0] = (BASE) + 5;                                                        \
    return (BASE) + 6;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ap_row[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE expected_ap_in[6] = { (TYPE)11, (TYPE)12, (TYPE)22, (TYPE)13, (TYPE)23, (TYPE)33 }; \
    TYPE expected_ap_out[6] = { (TYPE)41, (TYPE)42, (TYPE)43, (TYPE)52, (TYPE)53, (TYPE)63 }; \
    TYPE w[3] = { 0, 0, 0 };                                                      \
    TYPE z[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };                                   \
    TYPE expected_z[9] = { (TYPE)101, (TYPE)102, (TYPE)201, (TYPE)202, (TYPE)301, (TYPE)302, 0, 0, 0 }; \
    int ifail[3] = { 0, 0, 0 };                                                   \
    int expected_ifail[3] = { 7, 8, 9 };                                          \
    int m = 0;                                                                    \
    int info = 0;                                                                 \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 'I', FB_UPPER, 3, ap_row, (TYPE)0,   \
                 (TYPE)0, 1, 2, (TYPE)0.5, &m, w, z, 2, ifail);                  \
    if (info != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                    \
        g_##SUFFIX##_fortran_call.jobz != 'V' ||                                  \
        g_##SUFFIX##_fortran_call.range != 'I' ||                                 \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 ||                                       \
        memcmp(g_##SUFFIX##_fortran_call.ap_snapshot, expected_ap_in, sizeof(expected_ap_in)) != 0 || \
        g_##SUFFIX##_fortran_call.vl != (TYPE)0 || g_##SUFFIX##_fortran_call.vu != (TYPE)0 || \
        g_##SUFFIX##_fortran_call.il != 1 || g_##SUFFIX##_fortran_call.iu != 2 || \
        g_##SUFFIX##_fortran_call.abstol != (TYPE)0.5 || g_##SUFFIX##_fortran_call.m != &m || \
        g_##SUFFIX##_fortran_call.w != w || g_##SUFFIX##_fortran_call.ldz != 3 || \
        !g_##SUFFIX##_fortran_call.work_nonnull || !g_##SUFFIX##_fortran_call.iwork_nonnull || \
        g_##SUFFIX##_fortran_call.ifail != ifail || m != 2 ||                    \
        memcmp(ap_row, expected_ap_out, sizeof(expected_ap_out)) != 0 ||         \
        w[0] != (TYPE)((BASE) + 1) || w[1] != (TYPE)((BASE) + 2) ||              \
        memcmp(z, expected_z, sizeof(expected_z)) != 0 ||                        \
        memcmp(ifail, expected_ifail, sizeof(expected_ifail)) != 0) {            \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate packed SPEVX state correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk translates packed storage and copies back selected eigenvectors\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char jobz = 'V';                                                              \
    char range = 'I';                                                             \
    char uplo = 'L';                                                              \
    int n = 3;                                                                    \
    TYPE ap[6] = { (TYPE)7, (TYPE)8, (TYPE)9, (TYPE)10, (TYPE)11, (TYPE)12 };   \
    TYPE vl = (TYPE)1.5;                                                          \
    TYPE vu = (TYPE)2.5;                                                          \
    int il = 1;                                                                   \
    int iu = 2;                                                                   \
    TYPE abstol = (TYPE)0.25;                                                     \
    int m = 0;                                                                    \
    TYPE w[3] = { 0, 0, 0 };                                                      \
    TYPE z[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };                                   \
    int ldz = 3;                                                                  \
    TYPE work[24] = { 0 };                                                        \
    int iwork[15] = { 0 };                                                        \
    int ifail[3] = { 0, 0, 0 };                                                   \
    int info = -999;                                                              \
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
    thunk(&jobz, &range, &uplo, &n, ap, &vl, &vu, &il, &iu, &abstol, &m, w, z, &ldz, work, iwork, ifail, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.jobz != 'V' || g_##SUFFIX##_cblas_call.range != 'I' || \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 3 ||                                         \
        g_##SUFFIX##_cblas_call.ap != ap ||                                       \
        g_##SUFFIX##_cblas_call.vl != (TYPE)1.5 || g_##SUFFIX##_cblas_call.vu != (TYPE)2.5 || \
        g_##SUFFIX##_cblas_call.il != 1 || g_##SUFFIX##_cblas_call.iu != 2 ||     \
        g_##SUFFIX##_cblas_call.abstol != (TYPE)0.25 || g_##SUFFIX##_cblas_call.m != &m || \
        g_##SUFFIX##_cblas_call.w != w || g_##SUFFIX##_cblas_call.z != z ||       \
        g_##SUFFIX##_cblas_call.ldz != 3 || g_##SUFFIX##_cblas_call.ifail != ifail || \
        info != (BASE) + 6 || m != 2 || w[0] != (TYPE)((BASE) + 3) ||            \
        z[0] != (TYPE)((BASE) + 4) || ifail[0] != (BASE) + 5) {                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward SPEVX arguments correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards SPEVX arguments\n"); \
    return 0;                                                                     \
}

DEFINE_SPEVX_TESTS(sspevx, float, FB_OP_SSPEVX, 100)
DEFINE_SPEVX_TESTS(dspevx, double, FB_OP_DSPEVX, 300)

int main(void)
{
    int status = 0;

    status |= check_sspevx_fortran_to_cblas();
    status |= check_sspevx_cblas_to_fortran();
    status |= check_dspevx_fortran_to_cblas();
    status |= check_dspevx_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}