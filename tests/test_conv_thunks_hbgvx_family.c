#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    __real__ value = real_value;
    __imag__ value = 0.0f;
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    __real__ value = real_value;
    __imag__ value = 0.0;
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    return __real__ value;
}

#define DEFINE_HBGVX_TESTS(SUFFIX, TYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char jobz,            \
                                      char range, fb_uplo_t uplo, int n,        \
                                      int ka, int kb, TYPE *ab, int ldab,       \
                                      TYPE *bb, int ldbb, TYPE *q, int ldq,     \
                                      REAL_TYPE vl, REAL_TYPE vu, int il, int iu,\
                                      REAL_TYPE abstol, int *m, REAL_TYPE *w,   \
                                      TYPE *z, int ldz, int *ifail);            \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *jobz, char *range, char *uplo,   \
                                         int *n, int *ka, int *kb, TYPE *ab,    \
                                         int *ldab, TYPE *bb, int *ldbb, TYPE *q,\
                                         int *ldq, REAL_TYPE *vl, REAL_TYPE *vu,\
                                         int *il, int *iu, REAL_TYPE *abstol,   \
                                         int *m, REAL_TYPE *w, TYPE *z, int *ldz,\
                                         TYPE *work, REAL_TYPE *rwork, int *iwork,\
                                         int *ifail, int *info);                \
static struct {                                                                    \
    int called;                                                                    \
    char jobz;                                                                     \
    char range;                                                                    \
    char uplo;                                                                     \
    int n;                                                                         \
    int ka;                                                                        \
    int kb;                                                                        \
    REAL_TYPE ab_snapshot[6];                                                      \
    REAL_TYPE bb_snapshot[6];                                                      \
    int ldab;                                                                      \
    int ldbb;                                                                      \
    TYPE *q;                                                                       \
    int ldq;                                                                       \
    REAL_TYPE vl;                                                                  \
    REAL_TYPE vu;                                                                  \
    int il;                                                                        \
    int iu;                                                                        \
    REAL_TYPE abstol;                                                              \
    int *m;                                                                        \
    REAL_TYPE *w;                                                                  \
    TYPE *z;                                                                       \
    int ldz;                                                                       \
    int work_nonnull;                                                              \
    int rwork_nonnull;                                                             \
    int iwork_nonnull;                                                             \
    int *ifail;                                                                    \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    fb_layout_t layout;                                                            \
    char jobz;                                                                     \
    char range;                                                                    \
    fb_uplo_t uplo;                                                                \
    int n;                                                                         \
    int ka;                                                                        \
    int kb;                                                                        \
    TYPE *ab;                                                                      \
    int ldab;                                                                      \
    TYPE *bb;                                                                      \
    int ldbb;                                                                      \
    TYPE *q;                                                                       \
    int ldq;                                                                       \
    REAL_TYPE vl;                                                                  \
    REAL_TYPE vu;                                                                  \
    int il;                                                                        \
    int iu;                                                                        \
    REAL_TYPE abstol;                                                              \
    int *m;                                                                        \
    REAL_TYPE *w;                                                                  \
    TYPE *z;                                                                       \
    int ldz;                                                                       \
    int *ifail;                                                                    \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(char *jobz, char *range, char *uplo, int *n,\
                                    int *ka, int *kb, TYPE *ab, int *ldab,     \
                                    TYPE *bb, int *ldbb, TYPE *q, int *ldq,    \
                                    REAL_TYPE *vl, REAL_TYPE *vu, int *il,      \
                                    int *iu, REAL_TYPE *abstol, int *m, REAL_TYPE *w,\
                                    TYPE *z, int *ldz, TYPE *work, REAL_TYPE *rwork,\
                                    int *iwork, int *ifail, int *info)          \
{                                                                                  \
    size_t index = 0;                                                              \
    for (index = 0; index < 6; ++index) {                                         \
        g_##SUFFIX##_fortran_call.ab_snapshot[index] = REAL_PART(ab[index]);      \
        g_##SUFFIX##_fortran_call.bb_snapshot[index] = REAL_PART(bb[index]);      \
    }                                                                              \
    g_##SUFFIX##_fortran_call.called += 1;                                         \
    g_##SUFFIX##_fortran_call.jobz = *jobz;                                        \
    g_##SUFFIX##_fortran_call.range = *range;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                        \
    g_##SUFFIX##_fortran_call.n = *n;                                              \
    g_##SUFFIX##_fortran_call.ka = *ka;                                            \
    g_##SUFFIX##_fortran_call.kb = *kb;                                            \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                        \
    g_##SUFFIX##_fortran_call.ldbb = *ldbb;                                        \
    g_##SUFFIX##_fortran_call.q = q;                                               \
    g_##SUFFIX##_fortran_call.ldq = *ldq;                                          \
    g_##SUFFIX##_fortran_call.vl = *vl;                                            \
    g_##SUFFIX##_fortran_call.vu = *vu;                                            \
    g_##SUFFIX##_fortran_call.il = *il;                                            \
    g_##SUFFIX##_fortran_call.iu = *iu;                                            \
    g_##SUFFIX##_fortran_call.abstol = *abstol;                                    \
    g_##SUFFIX##_fortran_call.m = m;                                               \
    g_##SUFFIX##_fortran_call.w = w;                                               \
    g_##SUFFIX##_fortran_call.z = z;                                               \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                          \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                       \
    g_##SUFFIX##_fortran_call.rwork_nonnull = (rwork != NULL);                     \
    g_##SUFFIX##_fortran_call.iwork_nonnull = (iwork != NULL);                     \
    g_##SUFFIX##_fortran_call.ifail = ifail;                                       \
    ab[0] = MAKE((REAL_TYPE)0); ab[1] = MAKE((REAL_TYPE)41); ab[2] = MAKE((REAL_TYPE)42); \
    ab[3] = MAKE((REAL_TYPE)52); ab[4] = MAKE((REAL_TYPE)53); ab[5] = MAKE((REAL_TYPE)63); \
    bb[0] = MAKE((REAL_TYPE)0); bb[1] = MAKE((REAL_TYPE)71); bb[2] = MAKE((REAL_TYPE)72); \
    bb[3] = MAKE((REAL_TYPE)82); bb[4] = MAKE((REAL_TYPE)83); bb[5] = MAKE((REAL_TYPE)93); \
    q[0] = MAKE((REAL_TYPE)11); q[1] = MAKE((REAL_TYPE)21); q[2] = MAKE((REAL_TYPE)31); \
    q[3] = MAKE((REAL_TYPE)12); q[4] = MAKE((REAL_TYPE)22); q[5] = MAKE((REAL_TYPE)32); \
    q[6] = MAKE((REAL_TYPE)13); q[7] = MAKE((REAL_TYPE)23); q[8] = MAKE((REAL_TYPE)33); \
    *m = 2;                                                                        \
    w[0] = (REAL_TYPE)((BASE) + 1); w[1] = (REAL_TYPE)((BASE) + 2);               \
    z[0] = MAKE((REAL_TYPE)101); z[1] = MAKE((REAL_TYPE)201); z[2] = MAKE((REAL_TYPE)301); \
    z[3] = MAKE((REAL_TYPE)102); z[4] = MAKE((REAL_TYPE)202); z[5] = MAKE((REAL_TYPE)302); \
    z[6] = MAKE((REAL_TYPE)103); z[7] = MAKE((REAL_TYPE)203); z[8] = MAKE((REAL_TYPE)303); \
    ifail[0] = 7; ifail[1] = 8; ifail[2] = 9;                                     \
    *info = 0;                                                                     \
}                                                                                  \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char jobz, char range,     \
                                 fb_uplo_t uplo, int n, int ka, int kb,         \
                                 TYPE *ab, int ldab, TYPE *bb, int ldbb,        \
                                 TYPE *q, int ldq, REAL_TYPE vl, REAL_TYPE vu,  \
                                 int il, int iu, REAL_TYPE abstol, int *m,      \
                                 REAL_TYPE *w, TYPE *z, int ldz, int *ifail)    \
{                                                                                  \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.jobz = jobz;                                          \
    g_##SUFFIX##_cblas_call.range = range;                                        \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.ka = ka;                                              \
    g_##SUFFIX##_cblas_call.kb = kb;                                              \
    g_##SUFFIX##_cblas_call.ab = ab;                                              \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                          \
    g_##SUFFIX##_cblas_call.bb = bb;                                              \
    g_##SUFFIX##_cblas_call.ldbb = ldbb;                                          \
    g_##SUFFIX##_cblas_call.q = q;                                                \
    g_##SUFFIX##_cblas_call.ldq = ldq;                                            \
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
    q[0] = MAKE((REAL_TYPE)((BASE) + 7));                                         \
    w[0] = (REAL_TYPE)((BASE) + 4);                                               \
    z[0] = MAKE((REAL_TYPE)((BASE) + 5));                                         \
    ifail[0] = (BASE) + 8;                                                        \
    return (BASE) + 6;                                                            \
}                                                                                  \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                  \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ab[6] = { MAKE((REAL_TYPE)0), MAKE((REAL_TYPE)12), MAKE((REAL_TYPE)23), MAKE((REAL_TYPE)11), MAKE((REAL_TYPE)22), MAKE((REAL_TYPE)33) }; \
    TYPE bb[6] = { MAKE((REAL_TYPE)0), MAKE((REAL_TYPE)45), MAKE((REAL_TYPE)56), MAKE((REAL_TYPE)44), MAKE((REAL_TYPE)55), MAKE((REAL_TYPE)66) }; \
    REAL_TYPE expected_ab_in[6] = { (REAL_TYPE)0, (REAL_TYPE)11, (REAL_TYPE)12, (REAL_TYPE)22, (REAL_TYPE)23, (REAL_TYPE)33 }; \
    REAL_TYPE expected_bb_in[6] = { (REAL_TYPE)0, (REAL_TYPE)44, (REAL_TYPE)45, (REAL_TYPE)55, (REAL_TYPE)56, (REAL_TYPE)66 }; \
    REAL_TYPE expected_ab_out[6] = { (REAL_TYPE)0, (REAL_TYPE)42, (REAL_TYPE)53, (REAL_TYPE)41, (REAL_TYPE)52, (REAL_TYPE)63 }; \
    REAL_TYPE expected_bb_out[6] = { (REAL_TYPE)0, (REAL_TYPE)72, (REAL_TYPE)83, (REAL_TYPE)71, (REAL_TYPE)82, (REAL_TYPE)93 }; \
    TYPE q[9] = { 0 };                                                            \
    REAL_TYPE expected_q[9] = { (REAL_TYPE)11, (REAL_TYPE)12, (REAL_TYPE)13, (REAL_TYPE)21, (REAL_TYPE)22, (REAL_TYPE)23, (REAL_TYPE)31, (REAL_TYPE)32, (REAL_TYPE)33 }; \
    REAL_TYPE w[3] = { 0, 0, 0 };                                                 \
    TYPE z[9] = { 0 };                                                            \
    REAL_TYPE expected_z[9] = { (REAL_TYPE)101, (REAL_TYPE)102, (REAL_TYPE)103, (REAL_TYPE)201, (REAL_TYPE)202, (REAL_TYPE)203, (REAL_TYPE)301, (REAL_TYPE)302, (REAL_TYPE)303 }; \
    int ifail[3] = { 0, 0, 0 };                                                   \
    int expected_ifail[3] = { 7, 8, 9 };                                          \
    int m = 0;                                                                    \
    int info = 0;                                                                 \
    size_t index = 0;                                                             \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 'I', FB_UPPER, 3, 1, 1, ab, 3, bb, 3, q, 3, (REAL_TYPE)0, (REAL_TYPE)0, 1, 2, (REAL_TYPE)0.5, &m, w, z, 3, ifail); \
    if (info != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                    \
        g_##SUFFIX##_fortran_call.jobz != 'V' || g_##SUFFIX##_fortran_call.range != 'I' || \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 || g_##SUFFIX##_fortran_call.ka != 1 ||  \
        g_##SUFFIX##_fortran_call.kb != 1 ||                                      \
        memcmp(g_##SUFFIX##_fortran_call.ab_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.bb_snapshot, expected_bb_in, sizeof(expected_bb_in)) != 0 || \
        g_##SUFFIX##_fortran_call.ldab != 2 || g_##SUFFIX##_fortran_call.ldbb != 2 || \
        g_##SUFFIX##_fortran_call.q == NULL || g_##SUFFIX##_fortran_call.ldq != 3 || \
        g_##SUFFIX##_fortran_call.vl != (REAL_TYPE)0 || g_##SUFFIX##_fortran_call.vu != (REAL_TYPE)0 || \
        g_##SUFFIX##_fortran_call.il != 1 || g_##SUFFIX##_fortran_call.iu != 2 || \
        g_##SUFFIX##_fortran_call.abstol != (REAL_TYPE)0.5 || g_##SUFFIX##_fortran_call.m != &m || \
        g_##SUFFIX##_fortran_call.w != w || g_##SUFFIX##_fortran_call.ldz != 3 || \
        !g_##SUFFIX##_fortran_call.work_nonnull || !g_##SUFFIX##_fortran_call.rwork_nonnull || \
        !g_##SUFFIX##_fortran_call.iwork_nonnull ||                               \
        g_##SUFFIX##_fortran_call.ifail != ifail || m != 2) {                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route HBGVX calls\n"); \
        return 1;                                                                 \
    }                                                                             \
    for (index = 0; index < 6; ++index) {                                        \
        if (g_##SUFFIX##_fortran_call.ab_snapshot[index] != expected_ab_in[index] || \
            g_##SUFFIX##_fortran_call.bb_snapshot[index] != expected_bb_in[index] || \
            REAL_PART(ab[index]) != expected_ab_out[index] ||                     \
            REAL_PART(bb[index]) != expected_bb_out[index]) {                     \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate HBGVX band storage correctly\n"); \
            return 1;                                                              \
        }                                                                          \
    }                                                                              \
    for (index = 0; index < 9; ++index) {                                         \
        if (REAL_PART(q[index]) != expected_q[index] ||                           \
            REAL_PART(z[index]) != expected_z[index]) {                           \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not copy back HBGVX Q/Z outputs correctly\n"); \
            return 1;                                                              \
        }                                                                          \
    }                                                                              \
    if (w[0] != (REAL_TYPE)((BASE) + 1) || w[1] != (REAL_TYPE)((BASE) + 2) ||     \
        memcmp(ifail, expected_ifail, sizeof(expected_ifail)) != 0) {             \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not propagate HBGVX eigen metadata correctly\n"); \
        return 1;                                                                  \
    }                                                                              \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk translates A/B/Q/Z band-selection state and copies outputs back\n"); \
    return 0;                                                                      \
}                                                                                  \
static int check_##SUFFIX##_cblas_to_fortran(void)                                 \
{                                                                                  \
    fb_backend_vtable_t vtable;                                                    \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                         \
    char jobz = 'V';                                                               \
    char range = 'I';                                                              \
    char uplo = 'L';                                                               \
    int n = 3;                                                                     \
    int ka = 1;                                                                    \
    int kb = 1;                                                                    \
    TYPE ab[6] = { MAKE((REAL_TYPE)4), MAKE((REAL_TYPE)5), MAKE((REAL_TYPE)6), MAKE((REAL_TYPE)7), MAKE((REAL_TYPE)8), MAKE((REAL_TYPE)9) }; \
    TYPE bb[6] = { MAKE((REAL_TYPE)10), MAKE((REAL_TYPE)11), MAKE((REAL_TYPE)12), MAKE((REAL_TYPE)13), MAKE((REAL_TYPE)14), MAKE((REAL_TYPE)15) }; \
    int ldab = 2;                                                                  \
    int ldbb = 2;                                                                  \
    TYPE q[9] = { 0 };                                                             \
    int ldq = 3;                                                                   \
    REAL_TYPE vl = (REAL_TYPE)1.5;                                                 \
    REAL_TYPE vu = (REAL_TYPE)2.5;                                                 \
    int il = 1;                                                                    \
    int iu = 2;                                                                    \
    REAL_TYPE abstol = (REAL_TYPE)0.25;                                            \
    int m = 0;                                                                     \
    REAL_TYPE w[3] = { 0, 0, 0 };                                                  \
    TYPE z[9] = { 0 };                                                             \
    int ldz = 3;                                                                   \
    TYPE work[3] = { MAKE((REAL_TYPE)0), MAKE((REAL_TYPE)0), MAKE((REAL_TYPE)0) }; \
    REAL_TYPE rwork[21] = { 0 };                                                   \
    int iwork[15] = { 0 };                                                         \
    int ifail[3] = { 0, 0, 0 };                                                    \
    int info = -999;                                                               \
    memset(&vtable, 0, sizeof(vtable));                                            \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));          \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                         \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                      \
    fb_install_conv_thunks(&vtable, OP_ID);                                        \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];      \
    if (!thunk) {                                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                  \
    }                                                                              \
    thunk(&jobz, &range, &uplo, &n, &ka, &kb, ab, &ldab, bb, &ldbb, q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, w, z, &ldz, work, rwork, iwork, ifail, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                     \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                   \
        g_##SUFFIX##_cblas_call.jobz != 'V' || g_##SUFFIX##_cblas_call.range != 'I' || \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                                \
        g_##SUFFIX##_cblas_call.n != 3 || g_##SUFFIX##_cblas_call.ka != 1 ||       \
        g_##SUFFIX##_cblas_call.kb != 1 ||                                         \
        g_##SUFFIX##_cblas_call.ab != ab || g_##SUFFIX##_cblas_call.ldab != 2 ||   \
        g_##SUFFIX##_cblas_call.bb != bb || g_##SUFFIX##_cblas_call.ldbb != 2 ||   \
        g_##SUFFIX##_cblas_call.q != q || g_##SUFFIX##_cblas_call.ldq != 3 ||      \
        g_##SUFFIX##_cblas_call.vl != (REAL_TYPE)1.5 || g_##SUFFIX##_cblas_call.vu != (REAL_TYPE)2.5 || \
        g_##SUFFIX##_cblas_call.il != 1 || g_##SUFFIX##_cblas_call.iu != 2 ||      \
        g_##SUFFIX##_cblas_call.abstol != (REAL_TYPE)0.25 || g_##SUFFIX##_cblas_call.m != &m || \
        g_##SUFFIX##_cblas_call.w != w || g_##SUFFIX##_cblas_call.z != z ||        \
        g_##SUFFIX##_cblas_call.ldz != 3 || g_##SUFFIX##_cblas_call.ifail != ifail || \
        info != (BASE) + 6 || m != 2 || REAL_PART(q[0]) != (REAL_TYPE)((BASE) + 7) || \
        w[0] != (REAL_TYPE)((BASE) + 4) || REAL_PART(z[0]) != (REAL_TYPE)((BASE) + 5) || \
        ifail[0] != (BASE) + 8) {                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward HBGVX arguments correctly\n"); \
        return 1;                                                                  \
    }                                                                              \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards HBGVX arguments\n"); \
    return 0;                                                                      \
}

DEFINE_HBGVX_TESTS(chbgvx, fb_complex_float_t, float, FB_OP_CHBGVX, 100, make_cfloat, cfloat_real)
DEFINE_HBGVX_TESTS(zhbgvx, fb_complex_double_t, double, FB_OP_ZHBGVX, 300, make_cdouble, cdouble_real)

int main(void)
{
    int status = 0;

    status |= check_chbgvx_fortran_to_cblas();
    status |= check_chbgvx_cblas_to_fortran();
    status |= check_zhbgvx_fortran_to_cblas();
    status |= check_zhbgvx_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}