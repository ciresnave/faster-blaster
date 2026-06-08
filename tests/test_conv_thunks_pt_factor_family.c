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

#define DEFINE_PTTRF_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                      \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int n, TYPE *d,      \
                                      TYPE *e);                                \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *d, TYPE *e, int *info); \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    TYPE d_snapshot[3];                                                          \
    TYPE e_snapshot[2];                                                          \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    int n;                                                                       \
    TYPE *d;                                                                     \
    TYPE *e;                                                                     \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *n, TYPE *d, TYPE *e, int *info)       \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    memcpy(g_##SUFFIX##_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_fortran_call.e_snapshot, e, (size_t)((*n) - 1) * sizeof(TYPE)); \
    d[0] = (TYPE)((BASE) + 1);                                                   \
    e[0] = (TYPE)((BASE) + 2);                                                   \
    *info = (BASE) + 3;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int n, TYPE *d, TYPE *e)  \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.e = e;                                               \
    d[0] = (TYPE)((BASE) + 4);                                                   \
    e[0] = (TYPE)((BASE) + 5);                                                   \
    return (BASE) + 6;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE e[2] = { (TYPE)4, (TYPE)5 };                                            \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));    \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                  \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factor thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, d, e);                                 \
    if (info != (BASE) + 3 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        g_##SUFFIX##_fortran_call.d_snapshot[0] != (TYPE)1 ||                    \
        g_##SUFFIX##_fortran_call.e_snapshot[0] != (TYPE)4 ||                    \
        d[0] != (TYPE)((BASE) + 1) || e[0] != (TYPE)((BASE) + 2)) {             \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factor thunk did not route PTTRF inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS factor thunk routes PTTRF inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int n = 2;                                                                   \
    TYPE d[2] = { (TYPE)6, (TYPE)7 };                                            \
    TYPE e[1] = { (TYPE)8 };                                                     \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));        \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factor thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&n, d, e, &info);                                                      \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.e != e ||                                        \
        info != (BASE) + 6 || d[0] != (TYPE)((BASE) + 4) ||                     \
        e[0] != (TYPE)((BASE) + 5)) {                                            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factor thunk did not propagate PTTRF info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran factor thunk propagates PTTRF info\n"); \
    return 0;                                                                    \
}

#define DEFINE_PTTRF_COMPLEX_TESTS(SUFFIX, CTYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART, IMAG_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int n, REAL_TYPE *d, \
                                      CTYPE *e);                               \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, REAL_TYPE *d, CTYPE *e, int *info); \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    REAL_TYPE d_snapshot[3];                                                     \
    REAL_TYPE e_real_snapshot[2];                                                \
    REAL_TYPE e_imag_snapshot[2];                                                \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    int n;                                                                       \
    REAL_TYPE *d;                                                                \
    CTYPE *e;                                                                    \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *n, REAL_TYPE *d, CTYPE *e, int *info) \
{                                                                                \
    size_t index = 0;                                                            \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    memcpy(g_##SUFFIX##_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(REAL_TYPE)); \
    for (index = 0; index < (size_t)((*n) - 1); ++index) {                      \
        g_##SUFFIX##_fortran_call.e_real_snapshot[index] = REAL_PART(e[index]);  \
        g_##SUFFIX##_fortran_call.e_imag_snapshot[index] = IMAG_PART(e[index]);  \
    }                                                                            \
    d[0] = (REAL_TYPE)((BASE) + 1);                                              \
    e[0] = MAKE((REAL_TYPE)((BASE) + 2), (REAL_TYPE)((BASE) + 20));             \
    *info = (BASE) + 3;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int n, REAL_TYPE *d,      \
                                 CTYPE *e)                                      \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.e = e;                                               \
    d[0] = (REAL_TYPE)((BASE) + 4);                                              \
    e[0] = MAKE((REAL_TYPE)((BASE) + 5), (REAL_TYPE)((BASE) + 50));             \
    return (BASE) + 6;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    REAL_TYPE d[3] = { (REAL_TYPE)1, (REAL_TYPE)2, (REAL_TYPE)3 };              \
    CTYPE e[2] = { MAKE((REAL_TYPE)4, (REAL_TYPE)40), MAKE((REAL_TYPE)5, (REAL_TYPE)50) }; \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));    \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                  \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factor thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, d, e);                                 \
    if (info != (BASE) + 3 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        g_##SUFFIX##_fortran_call.d_snapshot[0] != (REAL_TYPE)1 ||               \
        g_##SUFFIX##_fortran_call.e_real_snapshot[0] != (REAL_TYPE)4 ||          \
        g_##SUFFIX##_fortran_call.e_imag_snapshot[0] != (REAL_TYPE)40 ||         \
        d[0] != (REAL_TYPE)((BASE) + 1) ||                                       \
        REAL_PART(e[0]) != (REAL_TYPE)((BASE) + 2) ||                            \
        IMAG_PART(e[0]) != (REAL_TYPE)((BASE) + 20)) {                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factor thunk did not route complex PTTRF inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS factor thunk routes complex PTTRF inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int n = 2;                                                                   \
    REAL_TYPE d[2] = { (REAL_TYPE)6, (REAL_TYPE)7 };                            \
    CTYPE e[1] = { MAKE((REAL_TYPE)8, (REAL_TYPE)80) };                         \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));        \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factor thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&n, d, e, &info);                                                      \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.e != e ||                                        \
        info != (BASE) + 6 || d[0] != (REAL_TYPE)((BASE) + 4) ||                \
        REAL_PART(e[0]) != (REAL_TYPE)((BASE) + 5) ||                            \
        IMAG_PART(e[0]) != (REAL_TYPE)((BASE) + 50)) {                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factor thunk did not propagate complex PTTRF info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran factor thunk propagates complex PTTRF info\n"); \
    return 0;                                                                    \
}

#define DEFINE_PTTRS_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                      \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, int nrhs, const TYPE *d,          \
                                      const TYPE *e, TYPE *b, int ldb);        \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nrhs,       \
                                         const TYPE *d, const TYPE *e, TYPE *b,\
                                         int *ldb, int *info);                 \
static struct {                                                                  \
    int called;                                                                  \
    char uplo;                                                                   \
    int n;                                                                       \
    int nrhs;                                                                    \
    TYPE d_snapshot[3];                                                          \
    TYPE e_snapshot[2];                                                          \
    TYPE b_snapshot[6];                                                          \
    int ldb;                                                                     \
} g_##SUFFIX##_solve_fortran_call;                                               \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    fb_uplo_t uplo;                                                              \
    int n;                                                                       \
    int nrhs;                                                                    \
    const TYPE *d;                                                               \
    const TYPE *e;                                                               \
    TYPE *b;                                                                     \
    int ldb;                                                                     \
} g_##SUFFIX##_solve_cblas_call;                                                 \
static void stub_##SUFFIX##_solve_fortran(char *uplo, int *n, int *nrhs,       \
                                          const TYPE *d, const TYPE *e, TYPE *b,\
                                          int *ldb, int *info)                  \
{                                                                                \
    g_##SUFFIX##_solve_fortran_call.called += 1;                                 \
    g_##SUFFIX##_solve_fortran_call.uplo = *uplo;                                \
    g_##SUFFIX##_solve_fortran_call.n = *n;                                      \
    g_##SUFFIX##_solve_fortran_call.nrhs = *nrhs;                                \
    memcpy(g_##SUFFIX##_solve_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_solve_fortran_call.e_snapshot, e, (size_t)((*n) - 1) * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_solve_fortran_call.b_snapshot, b, (size_t)(*ldb) * (size_t)(*nrhs) * sizeof(TYPE)); \
    g_##SUFFIX##_solve_fortran_call.ldb = *ldb;                                  \
    b[0] = (TYPE)((BASE) + 1); b[1] = (TYPE)((BASE) + 2); b[2] = (TYPE)((BASE) + 3); \
    b[3] = (TYPE)((BASE) + 4); b[4] = (TYPE)((BASE) + 5); b[5] = (TYPE)((BASE) + 6); \
    *info = (BASE) + 7;                                                          \
}                                                                                \
static int stub_##SUFFIX##_solve_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,\
                                       int nrhs, const TYPE *d, const TYPE *e, \
                                       TYPE *b, int ldb)                        \
{                                                                                \
    g_##SUFFIX##_solve_cblas_call.called += 1;                                   \
    g_##SUFFIX##_solve_cblas_call.layout = layout;                               \
    g_##SUFFIX##_solve_cblas_call.uplo = uplo;                                   \
    g_##SUFFIX##_solve_cblas_call.n = n;                                         \
    g_##SUFFIX##_solve_cblas_call.nrhs = nrhs;                                   \
    g_##SUFFIX##_solve_cblas_call.d = d;                                         \
    g_##SUFFIX##_solve_cblas_call.e = e;                                         \
    g_##SUFFIX##_solve_cblas_call.b = b;                                         \
    g_##SUFFIX##_solve_cblas_call.ldb = ldb;                                     \
    b[0] = (TYPE)((BASE) + 8);                                                   \
    return (BASE) + 9;                                                           \
}                                                                                \
static int check_##SUFFIX##_solve_fortran_to_cblas(void)                         \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                   \
    TYPE e[2] = { (TYPE)4, (TYPE)5 };                                             \
    TYPE b[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };       \
    TYPE expected_b_in[6] = { (TYPE)1, (TYPE)3, (TYPE)5, (TYPE)2, (TYPE)4, (TYPE)6 }; \
    TYPE expected_b_out[6] = { (TYPE)((BASE) + 1), (TYPE)((BASE) + 4), (TYPE)((BASE) + 2), (TYPE)((BASE) + 5), (TYPE)((BASE) + 3), (TYPE)((BASE) + 6) }; \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_solve_fortran_call, 0, sizeof(g_##SUFFIX##_solve_fortran_call)); \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_solve_fortran;            \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factored-solve thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 3, 2, d, e, b, 2);              \
    if (info != (BASE) + 7 ||                                                    \
        g_##SUFFIX##_solve_fortran_call.called != 1 ||                           \
        g_##SUFFIX##_solve_fortran_call.uplo != 'U' ||                           \
        g_##SUFFIX##_solve_fortran_call.n != 3 ||                                \
        g_##SUFFIX##_solve_fortran_call.nrhs != 2 ||                             \
        g_##SUFFIX##_solve_fortran_call.d_snapshot[0] != (TYPE)1 ||              \
        g_##SUFFIX##_solve_fortran_call.e_snapshot[0] != (TYPE)4 ||              \
        memcmp(g_##SUFFIX##_solve_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 || \
        g_##SUFFIX##_solve_fortran_call.ldb != 3 ||                              \
        d[0] != (TYPE)1 || e[0] != (TYPE)4 ||                                    \
        memcmp(b, expected_b_out, sizeof(expected_b_out)) != 0) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factored-solve thunk did not route PTTRS inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS factored-solve thunk routes PTTRS inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_solve_cblas_to_fortran(void)                         \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char uplo = 'L';                                                             \
    int n = 2;                                                                   \
    int nrhs = 1;                                                                \
    TYPE d[2] = { (TYPE)6, (TYPE)7 };                                            \
    TYPE e[1] = { (TYPE)8 };                                                     \
    TYPE b[2] = { (TYPE)9, (TYPE)10 };                                           \
    int ldb = 2;                                                                 \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_solve_cblas_call, 0, sizeof(g_##SUFFIX##_solve_cblas_call)); \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_solve_cblas;              \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factored-solve thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&uplo, &n, &nrhs, d, e, b, &ldb, &info);                               \
    if (g_##SUFFIX##_solve_cblas_call.called != 1 ||                             \
        g_##SUFFIX##_solve_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||           \
        g_##SUFFIX##_solve_cblas_call.uplo != FB_LOWER ||                        \
        g_##SUFFIX##_solve_cblas_call.n != 2 ||                                  \
        g_##SUFFIX##_solve_cblas_call.nrhs != 1 ||                               \
        g_##SUFFIX##_solve_cblas_call.d != d ||                                  \
        g_##SUFFIX##_solve_cblas_call.e != e ||                                  \
        g_##SUFFIX##_solve_cblas_call.b != b ||                                  \
        g_##SUFFIX##_solve_cblas_call.ldb != 2 ||                                \
        info != (BASE) + 9 || b[0] != (TYPE)((BASE) + 8)) {                     \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factored-solve thunk did not propagate PTTRS info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran factored-solve thunk propagates PTTRS info\n"); \
    return 0;                                                                    \
}

#define DEFINE_PTTRS_COMPLEX_TESTS(SUFFIX, CTYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART, IMAG_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, int nrhs, const REAL_TYPE *d,     \
                                      const CTYPE *e, CTYPE *b, int ldb);      \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nrhs,       \
                                         const REAL_TYPE *d, const CTYPE *e,   \
                                         CTYPE *b, int *ldb, int *info);       \
static struct {                                                                  \
    int called;                                                                  \
    char uplo;                                                                   \
    int n;                                                                       \
    int nrhs;                                                                    \
    REAL_TYPE d_snapshot[3];                                                     \
    REAL_TYPE e_real_snapshot[2];                                                \
    REAL_TYPE e_imag_snapshot[2];                                                \
    REAL_TYPE b_real_snapshot[6];                                                \
    REAL_TYPE b_imag_snapshot[6];                                                \
    int ldb;                                                                     \
} g_##SUFFIX##_solve_fortran_call;                                               \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    fb_uplo_t uplo;                                                              \
    int n;                                                                       \
    int nrhs;                                                                    \
    const REAL_TYPE *d;                                                          \
    const CTYPE *e;                                                              \
    CTYPE *b;                                                                    \
    int ldb;                                                                     \
} g_##SUFFIX##_solve_cblas_call;                                                 \
static void stub_##SUFFIX##_solve_fortran(char *uplo, int *n, int *nrhs,       \
                                          const REAL_TYPE *d, const CTYPE *e,  \
                                          CTYPE *b, int *ldb, int *info)        \
{                                                                                \
    size_t index = 0;                                                            \
    g_##SUFFIX##_solve_fortran_call.called += 1;                                 \
    g_##SUFFIX##_solve_fortran_call.uplo = *uplo;                                \
    g_##SUFFIX##_solve_fortran_call.n = *n;                                      \
    g_##SUFFIX##_solve_fortran_call.nrhs = *nrhs;                                \
    memcpy(g_##SUFFIX##_solve_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(REAL_TYPE)); \
    for (index = 0; index < (size_t)((*n) - 1); ++index) {                      \
        g_##SUFFIX##_solve_fortran_call.e_real_snapshot[index] = REAL_PART(e[index]); \
        g_##SUFFIX##_solve_fortran_call.e_imag_snapshot[index] = IMAG_PART(e[index]); \
    }                                                                            \
    for (index = 0; index < (size_t)(*ldb) * (size_t)(*nrhs); ++index) {        \
        g_##SUFFIX##_solve_fortran_call.b_real_snapshot[index] = REAL_PART(b[index]); \
        g_##SUFFIX##_solve_fortran_call.b_imag_snapshot[index] = IMAG_PART(b[index]); \
    }                                                                            \
    g_##SUFFIX##_solve_fortran_call.ldb = *ldb;                                  \
    b[0] = MAKE((REAL_TYPE)((BASE) + 1), (REAL_TYPE)((BASE) + 11));             \
    b[1] = MAKE((REAL_TYPE)((BASE) + 2), (REAL_TYPE)((BASE) + 12));             \
    b[2] = MAKE((REAL_TYPE)((BASE) + 3), (REAL_TYPE)((BASE) + 13));             \
    b[3] = MAKE((REAL_TYPE)((BASE) + 4), (REAL_TYPE)((BASE) + 14));             \
    b[4] = MAKE((REAL_TYPE)((BASE) + 5), (REAL_TYPE)((BASE) + 15));             \
    b[5] = MAKE((REAL_TYPE)((BASE) + 6), (REAL_TYPE)((BASE) + 16));             \
    *info = (BASE) + 7;                                                          \
}                                                                                \
static int stub_##SUFFIX##_solve_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,\
                                       int nrhs, const REAL_TYPE *d, const CTYPE *e,\
                                       CTYPE *b, int ldb)                       \
{                                                                                \
    g_##SUFFIX##_solve_cblas_call.called += 1;                                   \
    g_##SUFFIX##_solve_cblas_call.layout = layout;                               \
    g_##SUFFIX##_solve_cblas_call.uplo = uplo;                                   \
    g_##SUFFIX##_solve_cblas_call.n = n;                                         \
    g_##SUFFIX##_solve_cblas_call.nrhs = nrhs;                                   \
    g_##SUFFIX##_solve_cblas_call.d = d;                                         \
    g_##SUFFIX##_solve_cblas_call.e = e;                                         \
    g_##SUFFIX##_solve_cblas_call.b = b;                                         \
    g_##SUFFIX##_solve_cblas_call.ldb = ldb;                                     \
    b[0] = MAKE((REAL_TYPE)((BASE) + 8), (REAL_TYPE)((BASE) + 18));             \
    return (BASE) + 9;                                                           \
}                                                                                \
static int check_##SUFFIX##_solve_fortran_to_cblas(void)                         \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    REAL_TYPE d[3] = { (REAL_TYPE)1, (REAL_TYPE)2, (REAL_TYPE)3 };              \
    CTYPE e[2] = { MAKE((REAL_TYPE)4, (REAL_TYPE)40), MAKE((REAL_TYPE)5, (REAL_TYPE)50) }; \
    CTYPE b[6] = { MAKE((REAL_TYPE)1, (REAL_TYPE)10), MAKE((REAL_TYPE)2, (REAL_TYPE)20), MAKE((REAL_TYPE)3, (REAL_TYPE)30), MAKE((REAL_TYPE)4, (REAL_TYPE)40), MAKE((REAL_TYPE)5, (REAL_TYPE)50), MAKE((REAL_TYPE)6, (REAL_TYPE)60) }; \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_solve_fortran_call, 0, sizeof(g_##SUFFIX##_solve_fortran_call)); \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_solve_fortran;            \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factored-solve thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 3, 2, d, e, b, 2);              \
    if (info != (BASE) + 7 ||                                                    \
        g_##SUFFIX##_solve_fortran_call.called != 1 ||                           \
        g_##SUFFIX##_solve_fortran_call.uplo != 'U' ||                           \
        g_##SUFFIX##_solve_fortran_call.n != 3 ||                                \
        g_##SUFFIX##_solve_fortran_call.nrhs != 2 ||                             \
        g_##SUFFIX##_solve_fortran_call.d_snapshot[0] != (REAL_TYPE)1 ||         \
        g_##SUFFIX##_solve_fortran_call.e_real_snapshot[0] != (REAL_TYPE)4 ||    \
        g_##SUFFIX##_solve_fortran_call.e_imag_snapshot[0] != (REAL_TYPE)40 ||   \
        g_##SUFFIX##_solve_fortran_call.b_real_snapshot[0] != (REAL_TYPE)1 ||    \
        g_##SUFFIX##_solve_fortran_call.b_real_snapshot[1] != (REAL_TYPE)3 ||    \
        g_##SUFFIX##_solve_fortran_call.b_real_snapshot[2] != (REAL_TYPE)5 ||    \
        g_##SUFFIX##_solve_fortran_call.ldb != 3 ||                              \
        d[0] != (REAL_TYPE)1 || REAL_PART(e[0]) != (REAL_TYPE)4 ||               \
        IMAG_PART(e[0]) != (REAL_TYPE)40 ||                                      \
        REAL_PART(b[0]) != (REAL_TYPE)((BASE) + 1) ||                            \
        IMAG_PART(b[0]) != (REAL_TYPE)((BASE) + 11) ||                           \
        REAL_PART(b[1]) != (REAL_TYPE)((BASE) + 4) ||                            \
        IMAG_PART(b[1]) != (REAL_TYPE)((BASE) + 14) ||                           \
        REAL_PART(b[2]) != (REAL_TYPE)((BASE) + 2) ||                            \
        IMAG_PART(b[2]) != (REAL_TYPE)((BASE) + 12)) {                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factored-solve thunk did not route complex PTTRS inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS factored-solve thunk routes complex PTTRS inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_solve_cblas_to_fortran(void)                         \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char uplo = 'L';                                                             \
    int n = 2;                                                                   \
    int nrhs = 1;                                                                \
    REAL_TYPE d[2] = { (REAL_TYPE)6, (REAL_TYPE)7 };                            \
    CTYPE e[1] = { MAKE((REAL_TYPE)8, (REAL_TYPE)80) };                         \
    CTYPE b[2] = { MAKE((REAL_TYPE)9, (REAL_TYPE)90), MAKE((REAL_TYPE)10, (REAL_TYPE)100) }; \
    int ldb = 2;                                                                 \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_solve_cblas_call, 0, sizeof(g_##SUFFIX##_solve_cblas_call)); \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_solve_cblas;              \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factored-solve thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&uplo, &n, &nrhs, d, e, b, &ldb, &info);                               \
    if (g_##SUFFIX##_solve_cblas_call.called != 1 ||                             \
        g_##SUFFIX##_solve_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||           \
        g_##SUFFIX##_solve_cblas_call.uplo != FB_LOWER ||                        \
        g_##SUFFIX##_solve_cblas_call.n != 2 ||                                  \
        g_##SUFFIX##_solve_cblas_call.nrhs != 1 ||                               \
        g_##SUFFIX##_solve_cblas_call.d != d ||                                  \
        g_##SUFFIX##_solve_cblas_call.e != e ||                                  \
        g_##SUFFIX##_solve_cblas_call.b != b ||                                  \
        g_##SUFFIX##_solve_cblas_call.ldb != 2 ||                                \
        info != (BASE) + 9 || REAL_PART(b[0]) != (REAL_TYPE)((BASE) + 8) ||     \
        IMAG_PART(b[0]) != (REAL_TYPE)((BASE) + 18)) {                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factored-solve thunk did not propagate complex PTTRS info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran factored-solve thunk propagates complex PTTRS info\n"); \
    return 0;                                                                    \
}

DEFINE_PTTRF_REAL_TESTS(spttrf, float, FB_OP_SPTTRF, 100)
DEFINE_PTTRF_REAL_TESTS(dpttrf, double, FB_OP_DPTTRF, 300)
DEFINE_PTTRF_COMPLEX_TESTS(cpttrf, fb_complex_float_t, float, FB_OP_CPTTRF, 500, make_cfloat, cfloat_real, cfloat_imag)
DEFINE_PTTRF_COMPLEX_TESTS(zpttrf, fb_complex_double_t, double, FB_OP_ZPTTRF, 700, make_cdouble, cdouble_real, cdouble_imag)
DEFINE_PTTRS_REAL_TESTS(spttrs, float, FB_OP_SPTTRS, 900)
DEFINE_PTTRS_REAL_TESTS(dpttrs, double, FB_OP_DPTTRS, 1100)
DEFINE_PTTRS_COMPLEX_TESTS(cpttrs, fb_complex_float_t, float, FB_OP_CPTTRS, 1300, make_cfloat, cfloat_real, cfloat_imag)
DEFINE_PTTRS_COMPLEX_TESTS(zpttrs, fb_complex_double_t, double, FB_OP_ZPTTRS, 1500, make_cdouble, cdouble_real, cdouble_imag)

int main(void)
{
    int status = 0;

    status |= check_spttrf_fortran_to_cblas();
    status |= check_spttrf_cblas_to_fortran();
    status |= check_dpttrf_fortran_to_cblas();
    status |= check_dpttrf_cblas_to_fortran();
    status |= check_cpttrf_fortran_to_cblas();
    status |= check_cpttrf_cblas_to_fortran();
    status |= check_zpttrf_fortran_to_cblas();
    status |= check_zpttrf_cblas_to_fortran();
    status |= check_spttrs_solve_fortran_to_cblas();
    status |= check_spttrs_solve_cblas_to_fortran();
    status |= check_dpttrs_solve_fortran_to_cblas();
    status |= check_dpttrs_solve_cblas_to_fortran();
    status |= check_cpttrs_solve_fortran_to_cblas();
    status |= check_cpttrs_solve_cblas_to_fortran();
    status |= check_zpttrs_solve_fortran_to_cblas();
    status |= check_zpttrs_solve_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}