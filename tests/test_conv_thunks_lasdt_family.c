#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASDT_TESTS(SUFFIX, OP_ID, BASE)                                   \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, int *lvl, int *lnode, int *inode,   \
                                      int *ndiml, int *ndimr, int msub);         \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, int *lvl, int *lnode,           \
                                         int *inode, int *ndiml, int *ndimr,      \
                                         int *msub);                              \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    int *lvl;                                                                     \
    int *lnode;                                                                   \
    int *inode;                                                                   \
    int *ndiml;                                                                   \
    int *ndimr;                                                                   \
    int msub;                                                                     \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    int *lvl;                                                                     \
    int *lnode;                                                                   \
    int *inode;                                                                   \
    int *ndiml;                                                                   \
    int *ndimr;                                                                   \
    int msub;                                                                     \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(int *n, int *lvl, int *lnode, int *inode,    \
                                    int *ndiml, int *ndimr, int *msub) {         \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.lvl = lvl;                                          \
    g_##SUFFIX##_fortran_call.lnode = lnode;                                      \
    g_##SUFFIX##_fortran_call.inode = inode;                                      \
    g_##SUFFIX##_fortran_call.ndiml = ndiml;                                      \
    g_##SUFFIX##_fortran_call.ndimr = ndimr;                                      \
    g_##SUFFIX##_fortran_call.msub = *msub;                                       \
    *lvl = (BASE) + 1;                                                            \
    *lnode = (BASE) + 2;                                                          \
    inode[0] = (BASE) + 3;                                                        \
    ndiml[0] = (BASE) + 4;                                                        \
    ndimr[0] = (BASE) + 5;                                                        \
}                                                                                 \
static int stub_##SUFFIX##_cblas(int n, int *lvl, int *lnode, int *inode,        \
                                 int *ndiml, int *ndimr, int msub) {             \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.lvl = lvl;                                            \
    g_##SUFFIX##_cblas_call.lnode = lnode;                                        \
    g_##SUFFIX##_cblas_call.inode = inode;                                        \
    g_##SUFFIX##_cblas_call.ndiml = ndiml;                                        \
    g_##SUFFIX##_cblas_call.ndimr = ndimr;                                        \
    g_##SUFFIX##_cblas_call.msub = msub;                                          \
    *lvl = (BASE) + 6;                                                            \
    *lnode = (BASE) + 7;                                                          \
    inode[0] = (BASE) + 8;                                                        \
    ndiml[0] = (BASE) + 9;                                                        \
    ndimr[0] = (BASE) + 10;                                                       \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    int lvl = 0;                                                                  \
    int lnode = 0;                                                                \
    int inode[2] = { 1, 2 };                                                      \
    int ndiml[2] = { 3, 4 };                                                      \
    int ndimr[2] = { 5, 6 };                                                      \
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
    if (thunk(2, &lvl, &lnode, inode, ndiml, ndimr, 7) != 0 ||                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.lvl != &lvl ||                                  \
        g_##SUFFIX##_fortran_call.lnode != &lnode ||                              \
        g_##SUFFIX##_fortran_call.inode != inode ||                               \
        g_##SUFFIX##_fortran_call.ndiml != ndiml ||                               \
        g_##SUFFIX##_fortran_call.ndimr != ndimr ||                               \
        g_##SUFFIX##_fortran_call.msub != 7 ||                                    \
        lvl != (BASE) + 1 || lnode != (BASE) + 2 || inode[0] != (BASE) + 3 ||    \
        ndiml[0] != (BASE) + 4 || ndimr[0] != (BASE) + 5) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASDT inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASDT inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int n = 2;                                                                    \
    int lvl = 0;                                                                  \
    int lnode = 0;                                                                \
    int inode[2] = { 11, 12 };                                                    \
    int ndiml[2] = { 13, 14 };                                                    \
    int ndimr[2] = { 15, 16 };                                                    \
    int msub = 7;                                                                 \
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
    thunk(&n, &lvl, &lnode, inode, ndiml, ndimr, &msub);                         \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.lvl != &lvl ||                                    \
        g_##SUFFIX##_cblas_call.lnode != &lnode ||                                \
        g_##SUFFIX##_cblas_call.inode != inode ||                                 \
        g_##SUFFIX##_cblas_call.ndiml != ndiml ||                                 \
        g_##SUFFIX##_cblas_call.ndimr != ndimr ||                                 \
        g_##SUFFIX##_cblas_call.msub != 7 ||                                      \
        lvl != (BASE) + 6 || lnode != (BASE) + 7 || inode[0] != (BASE) + 8 ||    \
        ndiml[0] != (BASE) + 9 || ndimr[0] != (BASE) + 10) {                     \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASDT inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASDT inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LASDT_TESTS(slasdt, FB_OP_SLASDT, 100)
DEFINE_LASDT_TESTS(dlasdt, FB_OP_DLASDT, 300)

int main(void)
{
    int status = 0;

    status |= check_slasdt_fortran_to_cblas();
    status |= check_slasdt_cblas_to_fortran();
    status |= check_dlasdt_fortran_to_cblas();
    status |= check_dlasdt_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}