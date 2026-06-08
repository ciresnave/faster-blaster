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

#define DEFINE_PTCON_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                      \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int n,               \
                                      const TYPE *d, const TYPE *e, TYPE anorm,\
                                      TYPE *rcond);                            \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, const TYPE *d, const TYPE *e,\
                                         const TYPE *anorm, TYPE *rcond,       \
                                         TYPE *work, int *info);               \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    TYPE d_snapshot[3];                                                          \
    TYPE e_snapshot[2];                                                          \
    TYPE anorm;                                                                  \
    TYPE *rcond;                                                                  \
    int work_nonnull;                                                            \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    int n;                                                                       \
    const TYPE *d;                                                               \
    const TYPE *e;                                                               \
    TYPE anorm;                                                                  \
    TYPE *rcond;                                                                 \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *n, const TYPE *d, const TYPE *e,      \
                                    const TYPE *anorm, TYPE *rcond, TYPE *work,\
                                    int *info)                                  \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    memset(g_##SUFFIX##_fortran_call.d_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.d_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.e_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.e_snapshot)); \
    if (*n > 0) { memcpy(g_##SUFFIX##_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(TYPE)); } \
    if (*n > 1) { memcpy(g_##SUFFIX##_fortran_call.e_snapshot, e, (size_t)(*n - 1) * sizeof(TYPE)); } \
    g_##SUFFIX##_fortran_call.anorm = *anorm;                                    \
    g_##SUFFIX##_fortran_call.rcond = rcond;                                     \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                     \
    *rcond = (TYPE)((BASE) + 1);                                                 \
    *info = (BASE) + 2;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int n, const TYPE *d,     \
                                 const TYPE *e, TYPE anorm, TYPE *rcond)       \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.e = e;                                               \
    g_##SUFFIX##_cblas_call.anorm = anorm;                                       \
    g_##SUFFIX##_cblas_call.rcond = rcond;                                       \
    *rcond = (TYPE)((BASE) + 3);                                                 \
    return (BASE) + 4;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE e[2] = { (TYPE)4, (TYPE)5 };                                            \
    TYPE anorm = (TYPE)9;                                                        \
    TYPE rcond = 0;                                                              \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));    \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                  \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, d, e, anorm, &rcond);                  \
    if (info != (BASE) + 2 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        memcmp(g_##SUFFIX##_fortran_call.d_snapshot, d, sizeof(d)) != 0 ||      \
        memcmp(g_##SUFFIX##_fortran_call.e_snapshot, e, sizeof(e)) != 0 ||      \
        g_##SUFFIX##_fortran_call.anorm != anorm ||                              \
        g_##SUFFIX##_fortran_call.rcond != &rcond ||                             \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                               \
        rcond != (TYPE)((BASE) + 1)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route PTCON inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes PTCON inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int n = 2;                                                                   \
    TYPE d[2] = { (TYPE)6, (TYPE)7 };                                            \
    TYPE e[1] = { (TYPE)8 };                                                     \
    TYPE anorm = (TYPE)10;                                                       \
    TYPE rcond = 0;                                                              \
    TYPE work[2] = { 0, 0 };                                                     \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));        \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&n, d, e, &anorm, &rcond, work, &info);                                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.e != e ||                                        \
        g_##SUFFIX##_cblas_call.anorm != anorm ||                                \
        g_##SUFFIX##_cblas_call.rcond != &rcond ||                               \
        info != (BASE) + 4 || rcond != (TYPE)((BASE) + 3)) {                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate PTCON info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates PTCON info\n"); \
    return 0;                                                                    \
}

#define DEFINE_PTCON_COMPLEX_TESTS(SUFFIX, CTYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART, IMAG_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int n,               \
                                      const REAL_TYPE *d, const CTYPE *e,      \
                                      REAL_TYPE anorm, REAL_TYPE *rcond);       \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, const REAL_TYPE *d,           \
                                         const CTYPE *e, const REAL_TYPE *anorm,\
                                         REAL_TYPE *rcond, REAL_TYPE *work,     \
                                         int *info);                            \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    REAL_TYPE d_snapshot[3];                                                     \
    REAL_TYPE e_real_snapshot[2];                                                \
    REAL_TYPE e_imag_snapshot[2];                                                \
    REAL_TYPE anorm;                                                             \
    REAL_TYPE *rcond;                                                            \
    int work_nonnull;                                                            \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    int n;                                                                       \
    const REAL_TYPE *d;                                                          \
    const CTYPE *e;                                                              \
    REAL_TYPE anorm;                                                             \
    REAL_TYPE *rcond;                                                            \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *n, const REAL_TYPE *d, const CTYPE *e,\
                                    const REAL_TYPE *anorm, REAL_TYPE *rcond,  \
                                    REAL_TYPE *work, int *info)                 \
{                                                                                \
    size_t index = 0;                                                            \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    memset(g_##SUFFIX##_fortran_call.d_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.d_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.e_real_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.e_real_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.e_imag_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.e_imag_snapshot)); \
    if (*n > 0) { memcpy(g_##SUFFIX##_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(REAL_TYPE)); } \
    for (index = 0; index < (size_t)((*n > 1) ? (*n - 1) : 0); ++index) {       \
        g_##SUFFIX##_fortran_call.e_real_snapshot[index] = REAL_PART(e[index]);  \
        g_##SUFFIX##_fortran_call.e_imag_snapshot[index] = IMAG_PART(e[index]);  \
    }                                                                            \
    g_##SUFFIX##_fortran_call.anorm = *anorm;                                    \
    g_##SUFFIX##_fortran_call.rcond = rcond;                                     \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                     \
    *rcond = (REAL_TYPE)((BASE) + 1);                                            \
    *info = (BASE) + 2;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int n, const REAL_TYPE *d,\
                                 const CTYPE *e, REAL_TYPE anorm,              \
                                 REAL_TYPE *rcond)                             \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.e = e;                                               \
    g_##SUFFIX##_cblas_call.anorm = anorm;                                       \
    g_##SUFFIX##_cblas_call.rcond = rcond;                                       \
    *rcond = (REAL_TYPE)((BASE) + 3);                                            \
    return (BASE) + 4;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    REAL_TYPE d[3] = { (REAL_TYPE)1, (REAL_TYPE)2, (REAL_TYPE)3 };              \
    CTYPE e[2] = { MAKE((REAL_TYPE)4, (REAL_TYPE)40), MAKE((REAL_TYPE)5, (REAL_TYPE)50) }; \
    REAL_TYPE anorm = (REAL_TYPE)9;                                              \
    REAL_TYPE rcond = 0;                                                         \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));    \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                  \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, d, e, anorm, &rcond);                  \
    if (info != (BASE) + 2 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        memcmp(g_##SUFFIX##_fortran_call.d_snapshot, d, sizeof(d)) != 0 ||      \
        g_##SUFFIX##_fortran_call.e_real_snapshot[0] != REAL_PART(e[0]) ||      \
        g_##SUFFIX##_fortran_call.e_imag_snapshot[0] != IMAG_PART(e[0]) ||      \
        g_##SUFFIX##_fortran_call.e_real_snapshot[1] != REAL_PART(e[1]) ||      \
        g_##SUFFIX##_fortran_call.e_imag_snapshot[1] != IMAG_PART(e[1]) ||      \
        g_##SUFFIX##_fortran_call.anorm != anorm ||                              \
        g_##SUFFIX##_fortran_call.rcond != &rcond ||                             \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                               \
        rcond != (REAL_TYPE)((BASE) + 1)) {                                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route complex PTCON inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes complex PTCON inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int n = 2;                                                                   \
    REAL_TYPE d[2] = { (REAL_TYPE)6, (REAL_TYPE)7 };                            \
    CTYPE e[1] = { MAKE((REAL_TYPE)8, (REAL_TYPE)80) };                         \
    REAL_TYPE anorm = (REAL_TYPE)10;                                             \
    REAL_TYPE rcond = 0;                                                         \
    REAL_TYPE work[2] = { 0, 0 };                                                \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));        \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&n, d, e, &anorm, &rcond, work, &info);                                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.e != e ||                                        \
        g_##SUFFIX##_cblas_call.anorm != anorm ||                                \
        g_##SUFFIX##_cblas_call.rcond != &rcond ||                               \
        info != (BASE) + 4 || rcond != (REAL_TYPE)((BASE) + 3)) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate complex PTCON info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates complex PTCON info\n"); \
    return 0;                                                                    \
}

DEFINE_PTCON_REAL_TESTS(sptcon, float, FB_OP_SPTCON, 100)
DEFINE_PTCON_REAL_TESTS(dptcon, double, FB_OP_DPTCON, 300)
DEFINE_PTCON_COMPLEX_TESTS(cptcon, fb_complex_float_t, float, FB_OP_CPTCON, 500,
                           make_cfloat, cfloat_real, cfloat_imag)
DEFINE_PTCON_COMPLEX_TESTS(zptcon, fb_complex_double_t, double, FB_OP_ZPTCON, 700,
                           make_cdouble, cdouble_real, cdouble_imag)

int main(void)
{
    int status = 0;

    status |= check_sptcon_fortran_to_cblas();
    status |= check_sptcon_cblas_to_fortran();
    status |= check_dptcon_fortran_to_cblas();
    status |= check_dptcon_cblas_to_fortran();
    status |= check_cptcon_fortran_to_cblas();
    status |= check_cptcon_cblas_to_fortran();
    status |= check_zptcon_fortran_to_cblas();
    status |= check_zptcon_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}