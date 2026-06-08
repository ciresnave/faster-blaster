#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cf32(float real_value, float imag_value)
{
    fb_complex_float_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static fb_complex_double_t make_cf64(double real_value, double imag_value)
{
    fb_complex_double_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static int cf32_eq(fb_complex_float_t lhs, fb_complex_float_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

static int cf64_eq(fb_complex_double_t lhs, fb_complex_double_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

#define DEFINE_LAHQR_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(char job, char compz, int n, int ilo,    \
                                      int ihi, TYPE *h, int ldh, TYPE *wr,    \
                                      TYPE *wi, int iloz, int ihiz, TYPE *z,  \
                                      int ldz);                                \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *job, char *compz, int *n,       \
                                         int *ilo, int *ihi, TYPE *h,         \
                                         int *ldh, TYPE *wr, TYPE *wi,        \
                                         int *iloz, int *ihiz, TYPE *z,       \
                                         int *ldz, int *info);                \
static struct {                                                                 \
    int called;                                                                 \
    char job;                                                                   \
    char compz;                                                                 \
    int n;                                                                      \
    int ilo;                                                                    \
    int ihi;                                                                    \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    TYPE *wr;                                                                   \
    TYPE *wi;                                                                   \
    int iloz;                                                                   \
    int ihiz;                                                                   \
    TYPE *z;                                                                    \
    int ldz;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    char job;                                                                   \
    char compz;                                                                 \
    int n;                                                                      \
    int ilo;                                                                    \
    int ihi;                                                                    \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    TYPE *wr;                                                                   \
    TYPE *wi;                                                                   \
    int iloz;                                                                   \
    int ihiz;                                                                   \
    TYPE *z;                                                                    \
    int ldz;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *job, char *compz, int *n, int *ilo,  \
                                    int *ihi, TYPE *h, int *ldh, TYPE *wr,    \
                                    TYPE *wi, int *iloz, int *ihiz, TYPE *z,  \
                                    int *ldz, int *info)                      \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.job = *job;                                       \
    g_##SUFFIX##_fortran_call.compz = *compz;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.ilo = *ilo;                                       \
    g_##SUFFIX##_fortran_call.ihi = *ihi;                                       \
    g_##SUFFIX##_fortran_call.h = h;                                            \
    g_##SUFFIX##_fortran_call.ldh = *ldh;                                       \
    g_##SUFFIX##_fortran_call.wr = wr;                                          \
    g_##SUFFIX##_fortran_call.wi = wi;                                          \
    g_##SUFFIX##_fortran_call.iloz = *iloz;                                     \
    g_##SUFFIX##_fortran_call.ihiz = *ihiz;                                     \
    g_##SUFFIX##_fortran_call.z = z;                                            \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                       \
    wr[0] = (TYPE)((BASE) + 1);                                                 \
    wi[0] = (TYPE)((BASE) + 2);                                                 \
    z[0] = (TYPE)((BASE) + 3);                                                  \
    *info = (BASE) + 4;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(char job, char compz, int n, int ilo, int ihi,\
                                 TYPE *h, int ldh, TYPE *wr, TYPE *wi,        \
                                 int iloz, int ihiz, TYPE *z, int ldz)        \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.job = job;                                          \
    g_##SUFFIX##_cblas_call.compz = compz;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.ilo = ilo;                                          \
    g_##SUFFIX##_cblas_call.ihi = ihi;                                          \
    g_##SUFFIX##_cblas_call.h = h;                                              \
    g_##SUFFIX##_cblas_call.ldh = ldh;                                          \
    g_##SUFFIX##_cblas_call.wr = wr;                                            \
    g_##SUFFIX##_cblas_call.wi = wi;                                            \
    g_##SUFFIX##_cblas_call.iloz = iloz;                                        \
    g_##SUFFIX##_cblas_call.ihiz = ihiz;                                        \
    g_##SUFFIX##_cblas_call.z = z;                                              \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                          \
    wr[0] = (TYPE)((BASE) + 5);                                                 \
    wi[0] = (TYPE)((BASE) + 6);                                                 \
    z[0] = (TYPE)((BASE) + 7);                                                  \
    return (BASE) + 8;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE h[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE wr[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
    TYPE wi[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
    TYPE z[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                    \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));   \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                    \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                 \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];       \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    if (thunk('E', 'I', 2, 1, 2, h, 2, wr, wi, 1, 2, z, 2) != (BASE) + 4 ||    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.job != 'E' ||                                 \
        g_##SUFFIX##_fortran_call.compz != 'I' ||                               \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ilo != 1 ||                                   \
        g_##SUFFIX##_fortran_call.ihi != 2 ||                                   \
        g_##SUFFIX##_fortran_call.h != h ||                                     \
        g_##SUFFIX##_fortran_call.ldh != 2 ||                                   \
        g_##SUFFIX##_fortran_call.wr != wr ||                                   \
        g_##SUFFIX##_fortran_call.wi != wi ||                                   \
        g_##SUFFIX##_fortran_call.iloz != 1 ||                                  \
        g_##SUFFIX##_fortran_call.ihiz != 2 ||                                  \
        g_##SUFFIX##_fortran_call.z != z ||                                     \
        g_##SUFFIX##_fortran_call.ldz != 2 ||                                   \
        wr[0] != (TYPE)((BASE) + 1) ||                                          \
        wi[0] != (TYPE)((BASE) + 2) ||                                          \
        z[0] != (TYPE)((BASE) + 3)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAHQR real inputs or return info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards real LAHQR inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char job = 'S';                                                             \
    char compz = 'V';                                                           \
    int n = 2;                                                                  \
    int ilo = 1;                                                                \
    int ihi = 2;                                                                \
    int ldh = 2;                                                                \
    int iloz = 1;                                                               \
    int ihiz = 2;                                                               \
    int ldz = 2;                                                                \
    int info = -1;                                                              \
    TYPE h[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                    \
    TYPE wr[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
    TYPE wi[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
    TYPE z[4] = { (TYPE)31, (TYPE)32, (TYPE)33, (TYPE)34 };                    \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));       \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];   \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    thunk(&job, &compz, &n, &ilo, &ihi, h, &ldh, wr, wi, &iloz, &ihiz, z,      \
          &ldz, &info);                                                         \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.job != 'S' ||                                   \
        g_##SUFFIX##_cblas_call.compz != 'V' ||                                 \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.ilo != 1 ||                                     \
        g_##SUFFIX##_cblas_call.ihi != 2 ||                                     \
        g_##SUFFIX##_cblas_call.h != h ||                                       \
        g_##SUFFIX##_cblas_call.ldh != 2 ||                                     \
        g_##SUFFIX##_cblas_call.wr != wr ||                                     \
        g_##SUFFIX##_cblas_call.wi != wi ||                                     \
        g_##SUFFIX##_cblas_call.iloz != 1 ||                                    \
        g_##SUFFIX##_cblas_call.ihiz != 2 ||                                    \
        g_##SUFFIX##_cblas_call.z != z ||                                       \
        g_##SUFFIX##_cblas_call.ldz != 2 ||                                     \
        info != (BASE) + 8 ||                                                   \
        wr[0] != (TYPE)((BASE) + 5) ||                                          \
        wi[0] != (TYPE)((BASE) + 6) ||                                          \
        z[0] != (TYPE)((BASE) + 7)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAHQR real inputs or store info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences real LAHQR inputs and stores info\n"); \
    return 0;                                                                   \
}

#define DEFINE_LAHQR_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(char job, char compz, int n, int ilo,    \
                                      int ihi, CTYPE *h, int ldh, CTYPE *w,   \
                                      int iloz, int ihiz, CTYPE *z, int ldz); \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *job, char *compz, int *n,       \
                                         int *ilo, int *ihi, CTYPE *h,        \
                                         int *ldh, CTYPE *w, int *iloz,       \
                                         int *ihiz, CTYPE *z, int *ldz,       \
                                         int *info);                          \
static struct {                                                                 \
    int called;                                                                 \
    char job;                                                                   \
    char compz;                                                                 \
    int n;                                                                      \
    int ilo;                                                                    \
    int ihi;                                                                    \
    CTYPE *h;                                                                   \
    int ldh;                                                                    \
    CTYPE *w;                                                                   \
    int iloz;                                                                   \
    int ihiz;                                                                   \
    CTYPE *z;                                                                   \
    int ldz;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    char job;                                                                   \
    char compz;                                                                 \
    int n;                                                                      \
    int ilo;                                                                    \
    int ihi;                                                                    \
    CTYPE *h;                                                                   \
    int ldh;                                                                    \
    CTYPE *w;                                                                   \
    int iloz;                                                                   \
    int ihiz;                                                                   \
    CTYPE *z;                                                                   \
    int ldz;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *job, char *compz, int *n, int *ilo,  \
                                    int *ihi, CTYPE *h, int *ldh, CTYPE *w,   \
                                    int *iloz, int *ihiz, CTYPE *z, int *ldz, \
                                    int *info)                                 \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.job = *job;                                       \
    g_##SUFFIX##_fortran_call.compz = *compz;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.ilo = *ilo;                                       \
    g_##SUFFIX##_fortran_call.ihi = *ihi;                                       \
    g_##SUFFIX##_fortran_call.h = h;                                            \
    g_##SUFFIX##_fortran_call.ldh = *ldh;                                       \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.iloz = *iloz;                                     \
    g_##SUFFIX##_fortran_call.ihiz = *ihiz;                                     \
    g_##SUFFIX##_fortran_call.z = z;                                            \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                       \
    w[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                    \
    z[0] = MAKE_FN((BASE) + 2, (BASE) + 12);                                    \
    *info = (BASE) + 3;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(char job, char compz, int n, int ilo, int ihi,\
                                 CTYPE *h, int ldh, CTYPE *w, int iloz,       \
                                 int ihiz, CTYPE *z, int ldz)                 \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.job = job;                                          \
    g_##SUFFIX##_cblas_call.compz = compz;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.ilo = ilo;                                          \
    g_##SUFFIX##_cblas_call.ihi = ihi;                                          \
    g_##SUFFIX##_cblas_call.h = h;                                              \
    g_##SUFFIX##_cblas_call.ldh = ldh;                                          \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.iloz = iloz;                                        \
    g_##SUFFIX##_cblas_call.ihiz = ihiz;                                        \
    g_##SUFFIX##_cblas_call.z = z;                                              \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                          \
    w[0] = MAKE_FN((BASE) + 4, (BASE) + 14);                                    \
    z[0] = MAKE_FN((BASE) + 5, (BASE) + 15);                                    \
    return (BASE) + 6;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE h[4];                                                                 \
    CTYPE w[4];                                                                 \
    CTYPE z[4];                                                                 \
    h[0] = MAKE_FN(1, 21);                                                      \
    h[1] = MAKE_FN(2, 22);                                                      \
    h[2] = MAKE_FN(3, 23);                                                      \
    h[3] = MAKE_FN(4, 24);                                                      \
    w[0] = MAKE_FN(0, 0);                                                       \
    w[1] = MAKE_FN(0, 0);                                                       \
    w[2] = MAKE_FN(0, 0);                                                       \
    w[3] = MAKE_FN(0, 0);                                                       \
    z[0] = MAKE_FN(11, 31);                                                     \
    z[1] = MAKE_FN(12, 32);                                                     \
    z[2] = MAKE_FN(13, 33);                                                     \
    z[3] = MAKE_FN(14, 34);                                                     \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));   \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                    \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                 \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];       \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    if (thunk('E', 'I', 2, 1, 2, h, 2, w, 1, 2, z, 2) != (BASE) + 3 ||         \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.job != 'E' ||                                 \
        g_##SUFFIX##_fortran_call.compz != 'I' ||                               \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ilo != 1 ||                                   \
        g_##SUFFIX##_fortran_call.ihi != 2 ||                                   \
        g_##SUFFIX##_fortran_call.h != h ||                                     \
        g_##SUFFIX##_fortran_call.ldh != 2 ||                                   \
        g_##SUFFIX##_fortran_call.w != w ||                                     \
        g_##SUFFIX##_fortran_call.iloz != 1 ||                                  \
        g_##SUFFIX##_fortran_call.ihiz != 2 ||                                  \
        g_##SUFFIX##_fortran_call.z != z ||                                     \
        g_##SUFFIX##_fortran_call.ldz != 2 ||                                   \
        !EQ_FN(w[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                      \
        !EQ_FN(z[0], MAKE_FN((BASE) + 2, (BASE) + 12))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAHQR complex inputs or return info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LAHQR inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char job = 'S';                                                             \
    char compz = 'V';                                                           \
    int n = 2;                                                                  \
    int ilo = 1;                                                                \
    int ihi = 2;                                                                \
    int ldh = 2;                                                                \
    int iloz = 1;                                                               \
    int ihiz = 2;                                                               \
    int ldz = 2;                                                                \
    int info = -1;                                                              \
    CTYPE h[4];                                                                 \
    CTYPE w[4];                                                                 \
    CTYPE z[4];                                                                 \
    h[0] = MAKE_FN(21, 51);                                                     \
    h[1] = MAKE_FN(22, 52);                                                     \
    h[2] = MAKE_FN(23, 53);                                                     \
    h[3] = MAKE_FN(24, 54);                                                     \
    w[0] = MAKE_FN(0, 0);                                                       \
    w[1] = MAKE_FN(0, 0);                                                       \
    w[2] = MAKE_FN(0, 0);                                                       \
    w[3] = MAKE_FN(0, 0);                                                       \
    z[0] = MAKE_FN(31, 61);                                                     \
    z[1] = MAKE_FN(32, 62);                                                     \
    z[2] = MAKE_FN(33, 63);                                                     \
    z[3] = MAKE_FN(34, 64);                                                     \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));       \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];   \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    thunk(&job, &compz, &n, &ilo, &ihi, h, &ldh, w, &iloz, &ihiz, z, &ldz,     \
          &info);                                                               \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.job != 'S' ||                                   \
        g_##SUFFIX##_cblas_call.compz != 'V' ||                                 \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.ilo != 1 ||                                     \
        g_##SUFFIX##_cblas_call.ihi != 2 ||                                     \
        g_##SUFFIX##_cblas_call.h != h ||                                       \
        g_##SUFFIX##_cblas_call.ldh != 2 ||                                     \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        g_##SUFFIX##_cblas_call.iloz != 1 ||                                    \
        g_##SUFFIX##_cblas_call.ihiz != 2 ||                                    \
        g_##SUFFIX##_cblas_call.z != z ||                                       \
        g_##SUFFIX##_cblas_call.ldz != 2 ||                                     \
        info != (BASE) + 6 ||                                                   \
        !EQ_FN(w[0], MAKE_FN((BASE) + 4, (BASE) + 14)) ||                      \
        !EQ_FN(z[0], MAKE_FN((BASE) + 5, (BASE) + 15))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAHQR complex inputs or store info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LAHQR inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LAHQR_REAL_TESTS(slahqr, float, FB_OP_SLAHQR, 100)
DEFINE_LAHQR_REAL_TESTS(dlahqr, double, FB_OP_DLAHQR, 300)
DEFINE_LAHQR_COMPLEX_TESTS(clahqr, fb_complex_float_t, FB_OP_CLAHQR,
                           make_cf32, cf32_eq, 500)
DEFINE_LAHQR_COMPLEX_TESTS(zlahqr, fb_complex_double_t, FB_OP_ZLAHQR,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slahqr_fortran_to_cblas();
    status |= check_slahqr_cblas_to_fortran();
    status |= check_dlahqr_fortran_to_cblas();
    status |= check_dlahqr_cblas_to_fortran();
    status |= check_clahqr_fortran_to_cblas();
    status |= check_clahqr_cblas_to_fortran();
    status |= check_zlahqr_fortran_to_cblas();
    status |= check_zlahqr_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}