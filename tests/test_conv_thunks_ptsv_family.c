#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cfloat(float real_value, float imag_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static float cfloat_imag(fb_complex_float_t value)
{
    return __imag__ value;
}

static fb_complex_double_t make_cdouble(double real_value, double imag_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    return __real__ value;
}

static double cdouble_imag(fb_complex_double_t value)
{
    return __imag__ value;
}

#define DEFINE_PTSV_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                        \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int n, int nrhs,     \
                                      TYPE *d, TYPE *e, TYPE *b, int ldb);     \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, int *nrhs, TYPE *d, TYPE *e,  \
                                         TYPE *b, int *ldb, int *info);        \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    int nrhs;                                                                     \
    TYPE d_snapshot[3];                                                           \
    TYPE e_snapshot[2];                                                           \
    TYPE b_snapshot[6];                                                           \
    int ldb;                                                                      \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    int n;                                                                        \
    int nrhs;                                                                     \
    TYPE *d;                                                                      \
    TYPE *e;                                                                      \
    TYPE *b;                                                                      \
    int ldb;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(int *n, int *nrhs, TYPE *d, TYPE *e,        \
                                    TYPE *b, int *ldb, int *info)               \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    memcpy(g_##SUFFIX##_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_fortran_call.e_snapshot, e, (size_t)((*n) - 1) * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_fortran_call.b_snapshot, b, (size_t)(*ldb) * (size_t)(*nrhs) * sizeof(TYPE)); \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    d[0] = (TYPE)((BASE) + 1);                                                    \
    e[0] = (TYPE)((BASE) + 2);                                                    \
    b[0] = (TYPE)((BASE) + 3);                                                    \
    b[1] = (TYPE)((BASE) + 4);                                                    \
    b[2] = (TYPE)((BASE) + 5);                                                    \
    b[3] = (TYPE)((BASE) + 6);                                                    \
    b[4] = (TYPE)((BASE) + 7);                                                    \
    b[5] = (TYPE)((BASE) + 8);                                                    \
    *info = (BASE) + 9;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int n, int nrhs, TYPE *d,  \
                                 TYPE *e, TYPE *b, int ldb)                      \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                          \
    g_##SUFFIX##_cblas_call.d = d;                                                \
    g_##SUFFIX##_cblas_call.e = e;                                                \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                            \
    d[0] = (TYPE)((BASE) + 10);                                                   \
    e[0] = (TYPE)((BASE) + 11);                                                   \
    b[0] = (TYPE)((BASE) + 12);                                                   \
    return (BASE) + 13;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                   \
    TYPE e[2] = { (TYPE)4, (TYPE)5 };                                             \
    TYPE b[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };       \
    TYPE expected_b_in[6] = { (TYPE)1, (TYPE)3, (TYPE)5, (TYPE)2, (TYPE)4, (TYPE)6 }; \
    TYPE expected_b_out[6] = { (TYPE)((BASE) + 3), (TYPE)((BASE) + 6), (TYPE)((BASE) + 4), (TYPE)((BASE) + 7), (TYPE)((BASE) + 5), (TYPE)((BASE) + 8) }; \
    int info = 0;                                                                 \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS solve thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 2, d, e, b, 2);                         \
    if (info != (BASE) + 9 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 ||                                       \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                    \
        g_##SUFFIX##_fortran_call.d_snapshot[0] != (TYPE)1 ||                    \
        g_##SUFFIX##_fortran_call.e_snapshot[0] != (TYPE)4 ||                    \
        memcmp(g_##SUFFIX##_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 || \
        g_##SUFFIX##_fortran_call.ldb != 3 ||                                     \
        d[0] != (TYPE)((BASE) + 1) || e[0] != (TYPE)((BASE) + 2) ||              \
        memcmp(b, expected_b_out, sizeof(expected_b_out)) != 0) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS solve thunk did not route PTSV inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS solve thunk routes PTSV inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int n = 2;                                                                    \
    int nrhs = 1;                                                                 \
    TYPE d[2] = { (TYPE)6, (TYPE)7 };                                             \
    TYPE e[1] = { (TYPE)8 };                                                      \
    TYPE b[2] = { (TYPE)9, (TYPE)10 };                                            \
    int ldb = 2;                                                                  \
    int info = -999;                                                              \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran solve thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&n, &nrhs, d, e, b, &ldb, &info);                                      \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                      \
        g_##SUFFIX##_cblas_call.d != d ||                                         \
        g_##SUFFIX##_cblas_call.e != e ||                                         \
        g_##SUFFIX##_cblas_call.b != b ||                                         \
        g_##SUFFIX##_cblas_call.ldb != 2 ||                                       \
        info != (BASE) + 13 || d[0] != (TYPE)((BASE) + 10) ||                    \
        e[0] != (TYPE)((BASE) + 11) || b[0] != (TYPE)((BASE) + 12)) {            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran solve thunk did not propagate PTSV info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran solve thunk propagates PTSV info\n"); \
    return 0;                                                                     \
}

#define DEFINE_PTSV_COMPLEX_TESTS(SUFFIX, CTYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART, IMAG_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int n, int nrhs,     \
                                      REAL_TYPE *d, CTYPE *e, CTYPE *b, int ldb); \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, int *nrhs, REAL_TYPE *d,      \
                                         CTYPE *e, CTYPE *b, int *ldb, int *info); \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    int nrhs;                                                                     \
    REAL_TYPE d_snapshot[3];                                                      \
    REAL_TYPE e_real_snapshot[2];                                                 \
    REAL_TYPE e_imag_snapshot[2];                                                 \
    REAL_TYPE b_real_snapshot[6];                                                 \
    REAL_TYPE b_imag_snapshot[6];                                                 \
    int ldb;                                                                      \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    int n;                                                                        \
    int nrhs;                                                                     \
    REAL_TYPE *d;                                                                 \
    CTYPE *e;                                                                     \
    CTYPE *b;                                                                     \
    int ldb;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(int *n, int *nrhs, REAL_TYPE *d, CTYPE *e,  \
                                    CTYPE *b, int *ldb, int *info)              \
{                                                                                 \
    size_t index = 0;                                                             \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    memcpy(g_##SUFFIX##_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(REAL_TYPE)); \
    for (index = 0; index < (size_t)((*n) - 1); ++index) {                       \
        g_##SUFFIX##_fortran_call.e_real_snapshot[index] = REAL_PART(e[index]);  \
        g_##SUFFIX##_fortran_call.e_imag_snapshot[index] = IMAG_PART(e[index]);  \
    }                                                                             \
    for (index = 0; index < (size_t)(*ldb) * (size_t)(*nrhs); ++index) {         \
        g_##SUFFIX##_fortran_call.b_real_snapshot[index] = REAL_PART(b[index]);  \
        g_##SUFFIX##_fortran_call.b_imag_snapshot[index] = IMAG_PART(b[index]);  \
    }                                                                             \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    d[0] = (REAL_TYPE)((BASE) + 1);                                               \
    e[0] = MAKE((REAL_TYPE)((BASE) + 2), (REAL_TYPE)((BASE) + 20));              \
    b[0] = MAKE((REAL_TYPE)((BASE) + 3), (REAL_TYPE)((BASE) + 30));              \
    b[1] = MAKE((REAL_TYPE)((BASE) + 4), (REAL_TYPE)((BASE) + 40));              \
    b[2] = MAKE((REAL_TYPE)((BASE) + 5), (REAL_TYPE)((BASE) + 50));              \
    b[3] = MAKE((REAL_TYPE)((BASE) + 6), (REAL_TYPE)((BASE) + 60));              \
    b[4] = MAKE((REAL_TYPE)((BASE) + 7), (REAL_TYPE)((BASE) + 70));              \
    b[5] = MAKE((REAL_TYPE)((BASE) + 8), (REAL_TYPE)((BASE) + 80));              \
    *info = (BASE) + 9;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int n, int nrhs, REAL_TYPE *d, \
                                 CTYPE *e, CTYPE *b, int ldb)                    \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                          \
    g_##SUFFIX##_cblas_call.d = d;                                                \
    g_##SUFFIX##_cblas_call.e = e;                                                \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                            \
    d[0] = (REAL_TYPE)((BASE) + 10);                                              \
    e[0] = MAKE((REAL_TYPE)((BASE) + 11), (REAL_TYPE)((BASE) + 110));            \
    b[0] = MAKE((REAL_TYPE)((BASE) + 12), (REAL_TYPE)((BASE) + 120));            \
    return (BASE) + 13;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    REAL_TYPE d[3] = { (REAL_TYPE)1, (REAL_TYPE)2, (REAL_TYPE)3 };               \
    CTYPE e[2] = { MAKE((REAL_TYPE)4, (REAL_TYPE)40), MAKE((REAL_TYPE)5, (REAL_TYPE)50) }; \
    CTYPE b[6] = { MAKE((REAL_TYPE)1, (REAL_TYPE)10), MAKE((REAL_TYPE)2, (REAL_TYPE)20), MAKE((REAL_TYPE)3, (REAL_TYPE)30), MAKE((REAL_TYPE)4, (REAL_TYPE)40), MAKE((REAL_TYPE)5, (REAL_TYPE)50), MAKE((REAL_TYPE)6, (REAL_TYPE)60) }; \
    int info = 0;                                                                 \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS solve thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 2, d, e, b, 2);                         \
    if (info != (BASE) + 9 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 ||                                       \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                    \
        g_##SUFFIX##_fortran_call.d_snapshot[0] != (REAL_TYPE)1 ||               \
        g_##SUFFIX##_fortran_call.e_real_snapshot[0] != (REAL_TYPE)4 ||          \
        g_##SUFFIX##_fortran_call.e_imag_snapshot[0] != (REAL_TYPE)40 ||         \
        g_##SUFFIX##_fortran_call.b_real_snapshot[0] != (REAL_TYPE)1 ||          \
        g_##SUFFIX##_fortran_call.b_real_snapshot[1] != (REAL_TYPE)3 ||          \
        g_##SUFFIX##_fortran_call.b_real_snapshot[2] != (REAL_TYPE)5 ||          \
        g_##SUFFIX##_fortran_call.ldb != 3 ||                                     \
        d[0] != (REAL_TYPE)((BASE) + 1) ||                                        \
        REAL_PART(e[0]) != (REAL_TYPE)((BASE) + 2) ||                             \
        IMAG_PART(e[0]) != (REAL_TYPE)((BASE) + 20) ||                            \
        REAL_PART(b[0]) != (REAL_TYPE)((BASE) + 3) ||                             \
        IMAG_PART(b[0]) != (REAL_TYPE)((BASE) + 30) ||                            \
        REAL_PART(b[1]) != (REAL_TYPE)((BASE) + 6) ||                             \
        IMAG_PART(b[1]) != (REAL_TYPE)((BASE) + 60) ||                            \
        REAL_PART(b[2]) != (REAL_TYPE)((BASE) + 4) ||                             \
        IMAG_PART(b[2]) != (REAL_TYPE)((BASE) + 40)) {                            \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS solve thunk did not route complex PTSV inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS solve thunk routes complex PTSV inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int n = 2;                                                                    \
    int nrhs = 1;                                                                 \
    REAL_TYPE d[2] = { (REAL_TYPE)6, (REAL_TYPE)7 };                             \
    CTYPE e[1] = { MAKE((REAL_TYPE)8, (REAL_TYPE)80) };                          \
    CTYPE b[2] = { MAKE((REAL_TYPE)9, (REAL_TYPE)90), MAKE((REAL_TYPE)10, (REAL_TYPE)100) }; \
    int ldb = 2;                                                                  \
    int info = -999;                                                              \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran solve thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&n, &nrhs, d, e, b, &ldb, &info);                                      \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                      \
        g_##SUFFIX##_cblas_call.d != d ||                                         \
        g_##SUFFIX##_cblas_call.e != e ||                                         \
        g_##SUFFIX##_cblas_call.b != b ||                                         \
        g_##SUFFIX##_cblas_call.ldb != 2 ||                                       \
        info != (BASE) + 13 || d[0] != (REAL_TYPE)((BASE) + 10) ||               \
        REAL_PART(e[0]) != (REAL_TYPE)((BASE) + 11) ||                            \
        IMAG_PART(e[0]) != (REAL_TYPE)((BASE) + 110) ||                           \
        REAL_PART(b[0]) != (REAL_TYPE)((BASE) + 12) ||                            \
        IMAG_PART(b[0]) != (REAL_TYPE)((BASE) + 120)) {                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran solve thunk did not propagate complex PTSV info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran solve thunk propagates complex PTSV info\n"); \
    return 0;                                                                     \
}

DEFINE_PTSV_REAL_TESTS(sptsv, float, FB_OP_SPTSV, 100)
DEFINE_PTSV_REAL_TESTS(dptsv, double, FB_OP_DPTSV, 300)
DEFINE_PTSV_COMPLEX_TESTS(cptsv, fb_complex_float_t, float, FB_OP_CPTSV, 500, make_cfloat, cfloat_real, cfloat_imag)
DEFINE_PTSV_COMPLEX_TESTS(zptsv, fb_complex_double_t, double, FB_OP_ZPTSV, 700, make_cdouble, cdouble_real, cdouble_imag)

int main(void)
{
    int status = 0;

    status |= check_sptsv_fortran_to_cblas();
    status |= check_sptsv_cblas_to_fortran();
    status |= check_dptsv_fortran_to_cblas();
    status |= check_dptsv_cblas_to_fortran();
    status |= check_cptsv_fortran_to_cblas();
    status |= check_cptsv_cblas_to_fortran();
    status |= check_zptsv_fortran_to_cblas();
    status |= check_zptsv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}