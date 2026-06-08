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

#define DEFINE_REAL_LATDF_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int ijob, int n,     \
                                      TYPE *z, int ldz, TYPE *rhs,             \
                                      TYPE *rdsum, TYPE *rdscal,               \
                                      int *ipiv, int *jpiv);                   \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *ijob, int *n, TYPE *z, int *ldz, \
                                         TYPE *rhs, TYPE *rdsum, TYPE *rdscal, \
                                         int *ipiv, int *jpiv);                \
static struct {                                                                 \
    int called;                                                                 \
    int ijob;                                                                   \
    int n;                                                                      \
    int ldz;                                                                    \
    TYPE *z;                                                                    \
    TYPE *rhs;                                                                  \
    TYPE *rdsum;                                                                \
    TYPE *rdscal;                                                               \
    int *ipiv;                                                                  \
    int *jpiv;                                                                  \
    TYPE z_snapshot[4];                                                         \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    int ijob;                                                                   \
    int n;                                                                      \
    int ldz;                                                                    \
    TYPE *z;                                                                    \
    TYPE *rhs;                                                                  \
    TYPE *rdsum;                                                                \
    TYPE *rdscal;                                                               \
    int *ipiv;                                                                  \
    int *jpiv;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *ijob, int *n, TYPE *z, int *ldz,      \
                                    TYPE *rhs, TYPE *rdsum, TYPE *rdscal,      \
                                    int *ipiv, int *jpiv)                      \
{                                                                               \
    int idx;                                                                    \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.ijob = *ijob;                                     \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                       \
    g_##SUFFIX##_fortran_call.z = z;                                            \
    g_##SUFFIX##_fortran_call.rhs = rhs;                                        \
    g_##SUFFIX##_fortran_call.rdsum = rdsum;                                    \
    g_##SUFFIX##_fortran_call.rdscal = rdscal;                                  \
    g_##SUFFIX##_fortran_call.ipiv = ipiv;                                      \
    g_##SUFFIX##_fortran_call.jpiv = jpiv;                                      \
    for (idx = 0; idx < 4; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.z_snapshot[idx] = z[idx];                     \
    }                                                                           \
    z[0] = (TYPE)((BASE) + 10);                                                 \
    z[1] = (TYPE)((BASE) + 11);                                                 \
    z[2] = (TYPE)((BASE) + 12);                                                 \
    z[3] = (TYPE)((BASE) + 13);                                                 \
    rhs[0] = (TYPE)((BASE) + 20);                                               \
    rhs[1] = (TYPE)((BASE) + 21);                                               \
    *rdsum = (TYPE)((BASE) + 30);                                               \
    *rdscal = (TYPE)((BASE) + 31);                                              \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int ijob, int n, TYPE *z, \
                                 int ldz, TYPE *rhs, TYPE *rdsum,              \
                                 TYPE *rdscal, int *ipiv, int *jpiv)           \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.ijob = ijob;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                          \
    g_##SUFFIX##_cblas_call.z = z;                                              \
    g_##SUFFIX##_cblas_call.rhs = rhs;                                          \
    g_##SUFFIX##_cblas_call.rdsum = rdsum;                                      \
    g_##SUFFIX##_cblas_call.rdscal = rdscal;                                    \
    g_##SUFFIX##_cblas_call.ipiv = ipiv;                                        \
    g_##SUFFIX##_cblas_call.jpiv = jpiv;                                        \
    z[0] = (TYPE)((BASE) + 300);                                                \
    rhs[0] = (TYPE)((BASE) + 320);                                              \
    *rdsum = (TYPE)((BASE) + 330);                                              \
    *rdscal = (TYPE)((BASE) + 331);                                             \
    return (BASE) + 500;                                                        \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE z[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE rhs[2] = { (TYPE)5, (TYPE)6 };                                         \
    TYPE rdsum = (TYPE)7;                                                       \
    TYPE rdscal = (TYPE)8;                                                      \
    int ipiv[2] = { 2, 1 };                                                     \
    int jpiv[2] = { 2, 1 };                                                     \
    int status = 0;                                                             \
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
    status = thunk(FB_LAYOUT_ROW_MAJOR, 2, 2, z, 2, rhs, &rdsum, &rdscal, ipiv, jpiv); \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.ijob != 2 ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ldz != 2 ||                                   \
        g_##SUFFIX##_fortran_call.z == z ||                                     \
        g_##SUFFIX##_fortran_call.rhs != rhs ||                                 \
        g_##SUFFIX##_fortran_call.rdsum != &rdsum ||                            \
        g_##SUFFIX##_fortran_call.rdscal != &rdscal ||                          \
        g_##SUFFIX##_fortran_call.ipiv != ipiv ||                               \
        g_##SUFFIX##_fortran_call.jpiv != jpiv ||                               \
        g_##SUFFIX##_fortran_call.z_snapshot[0] != (TYPE)1 ||                   \
        g_##SUFFIX##_fortran_call.z_snapshot[1] != (TYPE)3 ||                   \
        g_##SUFFIX##_fortran_call.z_snapshot[2] != (TYPE)2 ||                   \
        g_##SUFFIX##_fortran_call.z_snapshot[3] != (TYPE)4 ||                   \
        z[0] != (TYPE)((BASE) + 10) || z[1] != (TYPE)((BASE) + 12) ||           \
        z[2] != (TYPE)((BASE) + 11) || z[3] != (TYPE)((BASE) + 13) ||           \
        rhs[0] != (TYPE)((BASE) + 20) || rhs[1] != (TYPE)((BASE) + 21) ||       \
        rdsum != (TYPE)((BASE) + 30) || rdscal != (TYPE)((BASE) + 31)) {       \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not transpose row-major Z or propagate RHS/RDSUM/RDSCAL correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk uses a col-major Z copy and propagates RHS/RDSUM/RDSCAL updates\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int ijob = 2;                                                               \
    int n = 2;                                                                  \
    int ldz = 2;                                                                \
    TYPE z[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE rhs[2] = { (TYPE)0, (TYPE)0 };                                         \
    TYPE rdsum = (TYPE)0;                                                       \
    TYPE rdscal = (TYPE)0;                                                      \
    int ipiv[2] = { 2, 1 };                                                     \
    int jpiv[2] = { 2, 1 };                                                     \
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
    thunk(&ijob, &n, z, &ldz, rhs, &rdsum, &rdscal, ipiv, jpiv);               \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.ijob != 2 ||                                    \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.ldz != 2 ||                                     \
        g_##SUFFIX##_cblas_call.z != z ||                                       \
        g_##SUFFIX##_cblas_call.rhs != rhs ||                                   \
        g_##SUFFIX##_cblas_call.rdsum != &rdsum ||                              \
        g_##SUFFIX##_cblas_call.rdscal != &rdscal ||                            \
        g_##SUFFIX##_cblas_call.ipiv != ipiv ||                                 \
        g_##SUFFIX##_cblas_call.jpiv != jpiv ||                                 \
        z[0] != (TYPE)((BASE) + 300) || rhs[0] != (TYPE)((BASE) + 320) ||       \
        rdsum != (TYPE)((BASE) + 330) || rdscal != (TYPE)((BASE) + 331)) {     \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward the LAPACKE-style LATDF arguments correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards the all-pointer LATDF ABI into the LAPACKE-style entry\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LATDF_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int ijob, int n,     \
                                      CTYPE *z, int ldz, CTYPE *rhs,           \
                                      RTYPE *rdsum, RTYPE *rdscal,             \
                                      int *ipiv, int *jpiv);                   \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *ijob, int *n, CTYPE *z, int *ldz, \
                                         CTYPE *rhs, RTYPE *rdsum, RTYPE *rdscal, \
                                         int *ipiv, int *jpiv);                \
static struct {                                                                 \
    int called;                                                                 \
    int ijob;                                                                   \
    int n;                                                                      \
    int ldz;                                                                    \
    CTYPE *z;                                                                   \
    CTYPE *rhs;                                                                 \
    RTYPE *rdsum;                                                               \
    RTYPE *rdscal;                                                              \
    int *ipiv;                                                                  \
    int *jpiv;                                                                  \
    CTYPE z_snapshot[4];                                                        \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    int ijob;                                                                   \
    int n;                                                                      \
    int ldz;                                                                    \
    CTYPE *z;                                                                   \
    CTYPE *rhs;                                                                 \
    RTYPE *rdsum;                                                               \
    RTYPE *rdscal;                                                              \
    int *ipiv;                                                                  \
    int *jpiv;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *ijob, int *n, CTYPE *z, int *ldz,     \
                                    CTYPE *rhs, RTYPE *rdsum, RTYPE *rdscal,   \
                                    int *ipiv, int *jpiv)                      \
{                                                                               \
    int idx;                                                                    \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.ijob = *ijob;                                     \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                       \
    g_##SUFFIX##_fortran_call.z = z;                                            \
    g_##SUFFIX##_fortran_call.rhs = rhs;                                        \
    g_##SUFFIX##_fortran_call.rdsum = rdsum;                                    \
    g_##SUFFIX##_fortran_call.rdscal = rdscal;                                  \
    g_##SUFFIX##_fortran_call.ipiv = ipiv;                                      \
    g_##SUFFIX##_fortran_call.jpiv = jpiv;                                      \
    for (idx = 0; idx < 4; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.z_snapshot[idx] = z[idx];                     \
    }                                                                           \
    z[0] = MAKE_FN((RTYPE)((BASE) + 10), (RTYPE)((BASE) + 60));                \
    z[1] = MAKE_FN((RTYPE)((BASE) + 11), (RTYPE)((BASE) + 61));                \
    z[2] = MAKE_FN((RTYPE)((BASE) + 12), (RTYPE)((BASE) + 62));                \
    z[3] = MAKE_FN((RTYPE)((BASE) + 13), (RTYPE)((BASE) + 63));                \
    rhs[0] = MAKE_FN((RTYPE)((BASE) + 20), (RTYPE)((BASE) + 70));              \
    rhs[1] = MAKE_FN((RTYPE)((BASE) + 21), (RTYPE)((BASE) + 71));              \
    *rdsum = (RTYPE)((BASE) + 30);                                              \
    *rdscal = (RTYPE)((BASE) + 31);                                             \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int ijob, int n, CTYPE *z, \
                                 int ldz, CTYPE *rhs, RTYPE *rdsum,            \
                                 RTYPE *rdscal, int *ipiv, int *jpiv)         \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.ijob = ijob;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                          \
    g_##SUFFIX##_cblas_call.z = z;                                              \
    g_##SUFFIX##_cblas_call.rhs = rhs;                                          \
    g_##SUFFIX##_cblas_call.rdsum = rdsum;                                      \
    g_##SUFFIX##_cblas_call.rdscal = rdscal;                                    \
    g_##SUFFIX##_cblas_call.ipiv = ipiv;                                        \
    g_##SUFFIX##_cblas_call.jpiv = jpiv;                                        \
    z[0] = MAKE_FN((RTYPE)((BASE) + 300), (RTYPE)((BASE) + 350));              \
    rhs[0] = MAKE_FN((RTYPE)((BASE) + 320), (RTYPE)((BASE) + 370));            \
    *rdsum = (RTYPE)((BASE) + 330);                                             \
    *rdscal = (RTYPE)((BASE) + 331);                                            \
    return (BASE) + 500;                                                        \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE z[4];                                                                 \
    CTYPE rhs[2];                                                               \
    RTYPE rdsum = (RTYPE)7;                                                     \
    RTYPE rdscal = (RTYPE)8;                                                    \
    int ipiv[2] = { 2, 1 };                                                     \
    int jpiv[2] = { 2, 1 };                                                     \
    int status = 0;                                                             \
    z[0] = MAKE_FN((RTYPE)1, (RTYPE)11);                                        \
    z[1] = MAKE_FN((RTYPE)2, (RTYPE)12);                                        \
    z[2] = MAKE_FN((RTYPE)3, (RTYPE)13);                                        \
    z[3] = MAKE_FN((RTYPE)4, (RTYPE)14);                                        \
    rhs[0] = MAKE_FN((RTYPE)5, (RTYPE)15);                                      \
    rhs[1] = MAKE_FN((RTYPE)6, (RTYPE)16);                                      \
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
    status = thunk(FB_LAYOUT_ROW_MAJOR, 2, 2, z, 2, rhs, &rdsum, &rdscal, ipiv, jpiv); \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.ijob != 2 ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ldz != 2 ||                                   \
        g_##SUFFIX##_fortran_call.z == z ||                                     \
        g_##SUFFIX##_fortran_call.rhs != rhs ||                                 \
        g_##SUFFIX##_fortran_call.rdsum != &rdsum ||                            \
        g_##SUFFIX##_fortran_call.rdscal != &rdscal ||                          \
        g_##SUFFIX##_fortran_call.ipiv != ipiv ||                               \
        g_##SUFFIX##_fortran_call.jpiv != jpiv ||                               \
        !EQ_FN(g_##SUFFIX##_fortran_call.z_snapshot[0],                         \
               MAKE_FN((RTYPE)1, (RTYPE)11)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.z_snapshot[1],                         \
               MAKE_FN((RTYPE)3, (RTYPE)13)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.z_snapshot[2],                         \
               MAKE_FN((RTYPE)2, (RTYPE)12)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.z_snapshot[3],                         \
               MAKE_FN((RTYPE)4, (RTYPE)14)) ||                                 \
        !EQ_FN(z[0], MAKE_FN((RTYPE)((BASE) + 10), (RTYPE)((BASE) + 60))) ||    \
        !EQ_FN(z[1], MAKE_FN((RTYPE)((BASE) + 12), (RTYPE)((BASE) + 62))) ||    \
        !EQ_FN(z[2], MAKE_FN((RTYPE)((BASE) + 11), (RTYPE)((BASE) + 61))) ||    \
        !EQ_FN(z[3], MAKE_FN((RTYPE)((BASE) + 13), (RTYPE)((BASE) + 63))) ||    \
        !EQ_FN(rhs[0], MAKE_FN((RTYPE)((BASE) + 20), (RTYPE)((BASE) + 70))) ||  \
        !EQ_FN(rhs[1], MAKE_FN((RTYPE)((BASE) + 21), (RTYPE)((BASE) + 71))) ||  \
        rdsum != (RTYPE)((BASE) + 30) || rdscal != (RTYPE)((BASE) + 31)) {     \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not transpose row-major complex Z or propagate RHS/RDSUM/RDSCAL correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk uses a col-major Z copy and propagates complex RHS/RDSUM/RDSCAL updates\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int ijob = 2;                                                               \
    int n = 2;                                                                  \
    int ldz = 2;                                                                \
    CTYPE z[4];                                                                 \
    CTYPE rhs[2];                                                               \
    RTYPE rdsum = (RTYPE)0;                                                     \
    RTYPE rdscal = (RTYPE)0;                                                    \
    int ipiv[2] = { 2, 1 };                                                     \
    int jpiv[2] = { 2, 1 };                                                     \
    z[0] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    z[1] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    z[2] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    z[3] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    rhs[0] = MAKE_FN((RTYPE)0, (RTYPE)0);                                       \
    rhs[1] = MAKE_FN((RTYPE)0, (RTYPE)0);                                       \
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
    thunk(&ijob, &n, z, &ldz, rhs, &rdsum, &rdscal, ipiv, jpiv);               \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.ijob != 2 ||                                    \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.ldz != 2 ||                                     \
        g_##SUFFIX##_cblas_call.z != z ||                                       \
        g_##SUFFIX##_cblas_call.rhs != rhs ||                                   \
        g_##SUFFIX##_cblas_call.rdsum != &rdsum ||                              \
        g_##SUFFIX##_cblas_call.rdscal != &rdscal ||                            \
        g_##SUFFIX##_cblas_call.ipiv != ipiv ||                                 \
        g_##SUFFIX##_cblas_call.jpiv != jpiv ||                                 \
        !EQ_FN(z[0], MAKE_FN((RTYPE)((BASE) + 300), (RTYPE)((BASE) + 350))) ||  \
        !EQ_FN(rhs[0], MAKE_FN((RTYPE)((BASE) + 320), (RTYPE)((BASE) + 370))) ||\
        rdsum != (RTYPE)((BASE) + 330) || rdscal != (RTYPE)((BASE) + 331)) {   \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward the complex LAPACKE-style LATDF arguments correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards the complex all-pointer LATDF ABI into the LAPACKE-style entry\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_LATDF_TESTS(slatdf, float, FB_OP_SLATDF, 100)
DEFINE_REAL_LATDF_TESTS(dlatdf, double, FB_OP_DLATDF, 300)
DEFINE_COMPLEX_LATDF_TESTS(clatdf, fb_complex_float_t, float, FB_OP_CLATDF,
                           make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LATDF_TESTS(zlatdf, fb_complex_double_t, double, FB_OP_ZLATDF,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slatdf_fortran_to_cblas();
    status |= check_slatdf_cblas_to_fortran();
    status |= check_dlatdf_fortran_to_cblas();
    status |= check_dlatdf_cblas_to_fortran();
    status |= check_clatdf_fortran_to_cblas();
    status |= check_clatdf_cblas_to_fortran();
    status |= check_zlatdf_fortran_to_cblas();
    status |= check_zlatdf_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}