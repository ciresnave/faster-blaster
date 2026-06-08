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

#define DEFINE_RSCL_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                        \
typedef void (*fb_##SUFFIX##_cblas_fn)(int n, TYPE sa, TYPE *sx, int incx);    \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *sa, TYPE *sx, int *incx); \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    TYPE sa;                                                                      \
    TYPE *sx;                                                                     \
    int incx;                                                                     \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    TYPE sa;                                                                      \
    TYPE *sx;                                                                     \
    int incx;                                                                     \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(int *n, TYPE *sa, TYPE *sx, int *incx)     \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.sa = *sa;                                           \
    g_##SUFFIX##_fortran_call.sx = sx;                                            \
    g_##SUFFIX##_fortran_call.incx = *incx;                                       \
    sx[0] = (TYPE)((BASE) + 1);                                                   \
    if (*n > 1) sx[*incx] = (TYPE)((BASE) + 2);                                  \
}                                                                                 \
static void stub_##SUFFIX##_cblas(int n, TYPE sa, TYPE *sx, int incx)           \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.sa = sa;                                              \
    g_##SUFFIX##_cblas_call.sx = sx;                                              \
    g_##SUFFIX##_cblas_call.incx = incx;                                          \
    sx[0] = (TYPE)((BASE) + 3);                                                   \
    if (n > 1) sx[incx] = (TYPE)((BASE) + 4);                                    \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    int n = 2;                                                                    \
    TYPE sa = (TYPE)5;                                                            \
    TYPE sx[3] = { (TYPE)10, (TYPE)99, (TYPE)20 };                               \
    int incx = 2;                                                                 \
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
    thunk(n, sa, sx, incx);                                                       \
    if (g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.n != n ||                                       \
        g_##SUFFIX##_fortran_call.sa != sa ||                                     \
        g_##SUFFIX##_fortran_call.sx != sx ||                                     \
        g_##SUFFIX##_fortran_call.incx != incx ||                                 \
        sx[0] != (TYPE)((BASE) + 1) || sx[2] != (TYPE)((BASE) + 2)) {            \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward RSCL arguments correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards RSCL arguments\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int n = 2;                                                                    \
    TYPE sa = (TYPE)5;                                                            \
    TYPE sx[3] = { (TYPE)10, (TYPE)99, (TYPE)20 };                               \
    int incx = 2;                                                                 \
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
    thunk(&n, &sa, sx, &incx);                                                    \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.n != n ||                                         \
        g_##SUFFIX##_cblas_call.sa != sa ||                                       \
        g_##SUFFIX##_cblas_call.sx != sx ||                                       \
        g_##SUFFIX##_cblas_call.incx != incx ||                                   \
        sx[0] != (TYPE)((BASE) + 3) || sx[2] != (TYPE)((BASE) + 4)) {            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward RSCL arguments correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards RSCL arguments\n"); \
    return 0;                                                                     \
}

#define DEFINE_RSCL_COMPLEX_TESTS(SUFFIX, CTYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART, IMAG_PART) \
typedef void (*fb_##SUFFIX##_cblas_fn)(int n, REAL_TYPE sa, CTYPE *sx, int incx); \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, REAL_TYPE *sa, CTYPE *sx, int *incx); \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    REAL_TYPE sa;                                                                 \
    CTYPE *sx;                                                                    \
    int incx;                                                                     \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    REAL_TYPE sa;                                                                 \
    CTYPE *sx;                                                                    \
    int incx;                                                                     \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(int *n, REAL_TYPE *sa, CTYPE *sx, int *incx) \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.sa = *sa;                                           \
    g_##SUFFIX##_fortran_call.sx = sx;                                            \
    g_##SUFFIX##_fortran_call.incx = *incx;                                       \
    sx[0] = MAKE((REAL_TYPE)((BASE) + 1), (REAL_TYPE)((BASE) + 2));              \
    if (*n > 1) sx[*incx] = MAKE((REAL_TYPE)((BASE) + 3), (REAL_TYPE)((BASE) + 4)); \
}                                                                                 \
static void stub_##SUFFIX##_cblas(int n, REAL_TYPE sa, CTYPE *sx, int incx)     \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.sa = sa;                                              \
    g_##SUFFIX##_cblas_call.sx = sx;                                              \
    g_##SUFFIX##_cblas_call.incx = incx;                                          \
    sx[0] = MAKE((REAL_TYPE)((BASE) + 5), (REAL_TYPE)((BASE) + 6));              \
    if (n > 1) sx[incx] = MAKE((REAL_TYPE)((BASE) + 7), (REAL_TYPE)((BASE) + 8)); \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    int n = 2;                                                                    \
    REAL_TYPE sa = (REAL_TYPE)5;                                                  \
    CTYPE sx[3] = { MAKE((REAL_TYPE)10, (REAL_TYPE)100), MAKE((REAL_TYPE)99, (REAL_TYPE)999), MAKE((REAL_TYPE)20, (REAL_TYPE)200) }; \
    int incx = 2;                                                                 \
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
    thunk(n, sa, sx, incx);                                                       \
    if (g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.n != n ||                                       \
        g_##SUFFIX##_fortran_call.sa != sa ||                                     \
        g_##SUFFIX##_fortran_call.sx != sx ||                                     \
        g_##SUFFIX##_fortran_call.incx != incx ||                                 \
        REAL_PART(sx[0]) != (REAL_TYPE)((BASE) + 1) ||                            \
        IMAG_PART(sx[0]) != (REAL_TYPE)((BASE) + 2) ||                            \
        REAL_PART(sx[2]) != (REAL_TYPE)((BASE) + 3) ||                            \
        IMAG_PART(sx[2]) != (REAL_TYPE)((BASE) + 4)) {                            \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex RSCL arguments correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex RSCL arguments\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int n = 2;                                                                    \
    REAL_TYPE sa = (REAL_TYPE)5;                                                  \
    CTYPE sx[3] = { MAKE((REAL_TYPE)10, (REAL_TYPE)100), MAKE((REAL_TYPE)99, (REAL_TYPE)999), MAKE((REAL_TYPE)20, (REAL_TYPE)200) }; \
    int incx = 2;                                                                 \
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
    thunk(&n, &sa, sx, &incx);                                                    \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.n != n ||                                         \
        g_##SUFFIX##_cblas_call.sa != sa ||                                       \
        g_##SUFFIX##_cblas_call.sx != sx ||                                       \
        g_##SUFFIX##_cblas_call.incx != incx ||                                   \
        REAL_PART(sx[0]) != (REAL_TYPE)((BASE) + 5) ||                            \
        IMAG_PART(sx[0]) != (REAL_TYPE)((BASE) + 6) ||                            \
        REAL_PART(sx[2]) != (REAL_TYPE)((BASE) + 7) ||                            \
        IMAG_PART(sx[2]) != (REAL_TYPE)((BASE) + 8)) {                            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward complex RSCL arguments correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards complex RSCL arguments\n"); \
    return 0;                                                                     \
}

DEFINE_RSCL_REAL_TESTS(srscl, float, FB_OP_SRSCL, 100)
DEFINE_RSCL_REAL_TESTS(drscl, double, FB_OP_DRSCL, 300)
DEFINE_RSCL_COMPLEX_TESTS(csrscl, fb_complex_float_t, float, FB_OP_CSRSCL, 500, make_cfloat, cfloat_real, cfloat_imag)
DEFINE_RSCL_COMPLEX_TESTS(zdrscl, fb_complex_double_t, double, FB_OP_ZDRSCL, 700, make_cdouble, cdouble_real, cdouble_imag)

int main(void)
{
    int status = 0;

    status |= check_srscl_fortran_to_cblas();
    status |= check_srscl_cblas_to_fortran();
    status |= check_drscl_fortran_to_cblas();
    status |= check_drscl_cblas_to_fortran();
    status |= check_csrscl_fortran_to_cblas();
    status |= check_csrscl_cblas_to_fortran();
    status |= check_zdrscl_fortran_to_cblas();
    status |= check_zdrscl_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}