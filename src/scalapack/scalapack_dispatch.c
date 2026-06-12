/**
 * @file scalapack_dispatch.c
 * @brief ScaLAPACK dynamic loading, BLACS lifecycle, and C-to-Fortran shims.
 *
 * Architecture
 * ─────────────
 * ScaLAPACK + PBLAS are Fortran libraries.  Every argument — including
 * scalars and character flags — is passed by pointer.  Character arguments
 * additionally carry a hidden trailing length (gfortran / ifort ABI).
 *
 * This file owns that conversion:
 *   caller (C, scalars by value) → shim (takes C args, builds stack
 *   temporaries) → dlsym'd Fortran symbol (all pointers + char lengths).
 *
 * The dispatch entry point fb_scalapack_dispatch_op() is the hook the
 * future sequence executor calls when it encounters an op_id in the
 * ScaLAPACK range [FB_OP_SCALAPACK_BASE, FB_OP_SCALAPACK_END).
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "scalapack_dispatch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#  include <windows.h>
#  define SCALOAD(name)      ((void*)LoadLibraryA(name))
#  define SCAPROC(lib, sym)  ((void*)GetProcAddress((HMODULE)(lib), sym))
#else
#  include <dlfcn.h>
#  define SCALOAD(name)      dlopen(name, RTLD_LAZY | RTLD_GLOBAL)
#  define SCAPROC(lib, sym)  dlsym(lib, sym)
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * Global loader state
 * ───────────────────────────────────────────────────────────────────────── */

static struct {
    void *scalapack_handle;
    void *blacs_handle;

    /* BLACS lifecycle */
    blacs_pinfo_t    pinfo;
    blacs_get_t      get;
    blacs_gridinit_t gridinit;
    blacs_gridinfo_t gridinfo;
    blacs_gridexit_t gridexit;

    /* PBLAS Level 1 */
    psaxpy_t  Psaxpy;
    pdaxpy_t  Pdaxpy;

    /* PBLAS Level 3 */
    psgemm_t  Psgemm;
    pdgemm_t  Pdgemm;
    pstrsm_t  Pstrsm;
    pdtrsm_t  Pdtrsm;

    /* ScaLAPACK drivers */
    psgetrf_t Psgetrf;
    pdgetrf_t Pdgetrf;
    psgetrs_t Psgetrs;
    pdgetrs_t Pdgetrs;
    psgesv_t  Psgesv;
    pdgesv_t  Pdgesv;
    pspotrf_t Pspotrf;
    pdpotrf_t Pdpotrf;
    psgeqrf_t Psgeqrf;
    pdgeqrf_t Pdgeqrf;
    psgesvd_t Psgesvd;
    pdgesvd_t Pdgesvd;

    bool initialized;
} g_sca;

/* ─────────────────────────────────────────────────────────────────────────
 * Library loading
 * ───────────────────────────────────────────────────────────────────────── */

int fb_scalapack_load(void) {
    if (g_sca.initialized) return 0;

    /* Try platform-specific library names.  BLACS is often bundled inside
     * the ScaLAPACK lib on modern distributions, but may be separate. */
#ifdef _WIN32
    g_sca.scalapack_handle = SCALOAD("scalapack.dll");
    if (!g_sca.scalapack_handle)
        g_sca.scalapack_handle = SCALOAD("libscalapack.dll");
    g_sca.blacs_handle = g_sca.scalapack_handle; /* often same .dll */
    if (!g_sca.blacs_handle)
        g_sca.blacs_handle = SCALOAD("blacs.dll");
#else
    /* Try versioned sonames first, then unversioned fallbacks. */
    g_sca.scalapack_handle = SCALOAD("libscalapack-openmpi.so.2");
    if (!g_sca.scalapack_handle)
        g_sca.scalapack_handle = SCALOAD("libscalapack-mpich.so.2");
    if (!g_sca.scalapack_handle)
        g_sca.scalapack_handle = SCALOAD("libscalapack.so.2");
    if (!g_sca.scalapack_handle)
        g_sca.scalapack_handle = SCALOAD("libscalapack.so");

    /* BLACS is bundled in modern ScaLAPACK packages */
    g_sca.blacs_handle = g_sca.scalapack_handle;
    if (!g_sca.blacs_handle)
        g_sca.blacs_handle = SCALOAD("libblacs-openmpi.so");
    if (!g_sca.blacs_handle)
        g_sca.blacs_handle = SCALOAD("libblacs.so");
#endif

    if (!g_sca.scalapack_handle || !g_sca.blacs_handle) return -1;

    /* BLACS lifecycle — trailing underscore (Fortran symbol mangling) */
    g_sca.pinfo    = (blacs_pinfo_t)   SCAPROC(g_sca.blacs_handle, "blacs_pinfo_");
    g_sca.get      = (blacs_get_t)     SCAPROC(g_sca.blacs_handle, "blacs_get_");
    g_sca.gridinit = (blacs_gridinit_t)SCAPROC(g_sca.blacs_handle, "blacs_gridinit_");
    g_sca.gridinfo = (blacs_gridinfo_t)SCAPROC(g_sca.blacs_handle, "blacs_gridinfo_");
    g_sca.gridexit = (blacs_gridexit_t)SCAPROC(g_sca.blacs_handle, "blacs_gridexit_");

    /* PBLAS Level 1 */
    g_sca.Psaxpy = (psaxpy_t)SCAPROC(g_sca.scalapack_handle, "psaxpy_");
    g_sca.Pdaxpy = (pdaxpy_t)SCAPROC(g_sca.scalapack_handle, "pdaxpy_");

    /* PBLAS Level 3 */
    g_sca.Psgemm = (psgemm_t)SCAPROC(g_sca.scalapack_handle, "psgemm_");
    g_sca.Pdgemm = (pdgemm_t)SCAPROC(g_sca.scalapack_handle, "pdgemm_");
    g_sca.Pstrsm = (pstrsm_t)SCAPROC(g_sca.scalapack_handle, "pstrsm_");
    g_sca.Pdtrsm = (pdtrsm_t)SCAPROC(g_sca.scalapack_handle, "pdtrsm_");

    /* ScaLAPACK drivers */
    g_sca.Psgetrf = (psgetrf_t)SCAPROC(g_sca.scalapack_handle, "psgetrf_");
    g_sca.Pdgetrf = (pdgetrf_t)SCAPROC(g_sca.scalapack_handle, "pdgetrf_");
    g_sca.Psgetrs = (psgetrs_t)SCAPROC(g_sca.scalapack_handle, "psgetrs_");
    g_sca.Pdgetrs = (pdgetrs_t)SCAPROC(g_sca.scalapack_handle, "pdgetrs_");
    g_sca.Psgesv  = (psgesv_t) SCAPROC(g_sca.scalapack_handle, "psgesv_");
    g_sca.Pdgesv  = (pdgesv_t) SCAPROC(g_sca.scalapack_handle, "pdgesv_");
    g_sca.Pspotrf = (pspotrf_t)SCAPROC(g_sca.scalapack_handle, "pspotrf_");
    g_sca.Pdpotrf = (pdpotrf_t)SCAPROC(g_sca.scalapack_handle, "pdpotrf_");
    g_sca.Psgeqrf = (psgeqrf_t)SCAPROC(g_sca.scalapack_handle, "psgeqrf_");
    g_sca.Pdgeqrf = (pdgeqrf_t)SCAPROC(g_sca.scalapack_handle, "pdgeqrf_");
    g_sca.Psgesvd = (psgesvd_t)SCAPROC(g_sca.scalapack_handle, "psgesvd_");
    g_sca.Pdgesvd = (pdgesvd_t)SCAPROC(g_sca.scalapack_handle, "pdgesvd_");

    g_sca.initialized = true;
    return 0;
}

bool fb_scalapack_is_available(void) {
    return fb_scalapack_load() == 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * BLACS process-grid lifecycle
 * ───────────────────────────────────────────────────────────────────────── */

int fb_distributed_ctx_init(fb_distributed_ctx_t *ctx,
                             void *mpi_comm,
                             int   nprow,
                             int   npcol)
{
    if (!ctx) return -1;
    if (fb_scalapack_load() != 0) return -1;
    if (!g_sca.pinfo || !g_sca.get || !g_sca.gridinit || !g_sca.gridinfo)
        return -1;

    memset(ctx, 0, sizeof(*ctx));
    ctx->mpi_comm = mpi_comm;

    /* Query this process's rank and total size */
    g_sca.pinfo(&ctx->blacs.mypnum, &ctx->blacs.nprocs);

    /* Obtain a system context from BLACS.  Passing ictxt=-1, what=0 gives
     * the default system context which BLACS maps to mpi_comm internally. */
    int neg1 = -1, zero = 0;
    g_sca.get(&neg1, &zero, &ctx->blacs.ictxt);

    /* Initialize a row-major process grid */
    const char order = 'R';
    int nr = nprow, nc = npcol;
    g_sca.gridinit(&ctx->blacs.ictxt, &order, 1, &nr, &nc);

    /* Read back actual grid topology for this process */
    g_sca.gridinfo(&ctx->blacs.ictxt,
                   &ctx->blacs.nprow, &ctx->blacs.npcol,
                   &ctx->blacs.myrow, &ctx->blacs.mycol);

    ctx->initialized = true;
    return 0;
}

void fb_distributed_ctx_shutdown(fb_distributed_ctx_t *ctx) {
    if (!ctx || !ctx->initialized) return;
    if (g_sca.gridexit) g_sca.gridexit(&ctx->blacs.ictxt);
    ctx->initialized = false;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Array descriptor helper
 * ───────────────────────────────────────────────────────────────────────── */

void fb_scalapack_desc_init(fb_scalapack_desc_t desc,
                             const fb_distributed_ctx_t *ctx,
                             int m, int n,
                             int mb, int nb,
                             int lld)
{
    desc[0] = 1;                  /* dtype: block-cyclic 2D */
    desc[1] = ctx->blacs.ictxt;  /* BLACS context          */
    desc[2] = m;                  /* global rows            */
    desc[3] = n;                  /* global cols            */
    desc[4] = mb;                 /* row block size         */
    desc[5] = nb;                 /* col block size         */
    desc[6] = 0;                  /* first row process      */
    desc[7] = 0;                  /* first col process      */
    desc[8] = lld;                /* local leading dim      */
}

/* ─────────────────────────────────────────────────────────────────────────
 * PBLAS Level 1 shims
 * ───────────────────────────────────────────────────────────────────────── */

void fb_psaxpy(int n, float alpha,
               const float *X, int ix, int jx, const fb_scalapack_desc_t descx, int incx,
               float *Y, int iy, int jy, const fb_scalapack_desc_t descy, int incy,
               const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Psaxpy) return;
    int _n=n, _ix=ix, _jx=jx, _incx=incx, _iy=iy, _jy=jy, _incy=incy;
    float _alpha=alpha;
    g_sca.Psaxpy(&_n, &_alpha, X, &_ix, &_jx, descx, &_incx,
                               Y, &_iy, &_jy, descy, &_incy);
}

void fb_pdaxpy(int n, double alpha,
               const double *X, int ix, int jx, const fb_scalapack_desc_t descx, int incx,
               double *Y, int iy, int jy, const fb_scalapack_desc_t descy, int incy,
               const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pdaxpy) return;
    int _n=n, _ix=ix, _jx=jx, _incx=incx, _iy=iy, _jy=jy, _incy=incy;
    double _alpha=alpha;
    g_sca.Pdaxpy(&_n, &_alpha, X, &_ix, &_jx, descx, &_incx,
                               Y, &_iy, &_jy, descy, &_incy);
}

/* ─────────────────────────────────────────────────────────────────────────
 * PBLAS Level 3 shims
 * ───────────────────────────────────────────────────────────────────────── */

void fb_psgemm(char transa, char transb,
               int m, int n, int k,
               float alpha,
               const float *A, int ia, int ja, const fb_scalapack_desc_t desca,
               const float *B, int ib, int jb, const fb_scalapack_desc_t descb,
               float beta,
               float *C, int ic, int jc, const fb_scalapack_desc_t descc,
               const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Psgemm) return;
    int _m=m, _n=n, _k=k, _ia=ia, _ja=ja, _ib=ib, _jb=jb, _ic=ic, _jc=jc;
    float _alpha=alpha, _beta=beta;
    g_sca.Psgemm(&transa, &transb, &_m, &_n, &_k, &_alpha,
                 A, &_ia, &_ja, desca,
                 B, &_ib, &_jb, descb,
                 &_beta, C, &_ic, &_jc, descc,
                 1, 1);  /* hidden char lengths: gfortran ABI */
}

void fb_pdgemm(char transa, char transb,
               int m, int n, int k,
               double alpha,
               const double *A, int ia, int ja, const fb_scalapack_desc_t desca,
               const double *B, int ib, int jb, const fb_scalapack_desc_t descb,
               double beta,
               double *C, int ic, int jc, const fb_scalapack_desc_t descc,
               const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pdgemm) return;
    int _m=m, _n=n, _k=k, _ia=ia, _ja=ja, _ib=ib, _jb=jb, _ic=ic, _jc=jc;
    double _alpha=alpha, _beta=beta;
    g_sca.Pdgemm(&transa, &transb, &_m, &_n, &_k, &_alpha,
                 A, &_ia, &_ja, desca,
                 B, &_ib, &_jb, descb,
                 &_beta, C, &_ic, &_jc, descc,
                 1, 1);
}

void fb_pstrsm(char side, char uplo, char transa, char diag,
               int m, int n, float alpha,
               const float *A, int ia, int ja, const fb_scalapack_desc_t desca,
               float *B, int ib, int jb, const fb_scalapack_desc_t descb,
               const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pstrsm) return;
    int _m=m, _n=n, _ia=ia, _ja=ja, _ib=ib, _jb=jb;
    float _alpha=alpha;
    g_sca.Pstrsm(&side, &uplo, &transa, &diag,
                 &_m, &_n, &_alpha,
                 A, &_ia, &_ja, desca,
                 B, &_ib, &_jb, descb,
                 1, 1, 1, 1);
}

void fb_pdtrsm(char side, char uplo, char transa, char diag,
               int m, int n, double alpha,
               const double *A, int ia, int ja, const fb_scalapack_desc_t desca,
               double *B, int ib, int jb, const fb_scalapack_desc_t descb,
               const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pdtrsm) return;
    int _m=m, _n=n, _ia=ia, _ja=ja, _ib=ib, _jb=jb;
    double _alpha=alpha;
    g_sca.Pdtrsm(&side, &uplo, &transa, &diag,
                 &_m, &_n, &_alpha,
                 A, &_ia, &_ja, desca,
                 B, &_ib, &_jb, descb,
                 1, 1, 1, 1);
}

/* ─────────────────────────────────────────────────────────────────────────
 * ScaLAPACK driver shims
 * ───────────────────────────────────────────────────────────────────────── */

void fb_psgetrf(int m, int n,
                float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                int *ipiv, int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Psgetrf) { if (info) *info=-1; return; }
    int _m=m, _n=n, _ia=ia, _ja=ja;
    g_sca.Psgetrf(&_m, &_n, A, &_ia, &_ja, desca, ipiv, info);
}

void fb_pdgetrf(int m, int n,
                double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                int *ipiv, int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pdgetrf) { if (info) *info=-1; return; }
    int _m=m, _n=n, _ia=ia, _ja=ja;
    g_sca.Pdgetrf(&_m, &_n, A, &_ia, &_ja, desca, ipiv, info);
}

void fb_psgetrs(char trans, int n, int nrhs,
                const float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                const int *ipiv,
                float *B, int ib, int jb, const fb_scalapack_desc_t descb,
                int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Psgetrs) { if (info) *info=-1; return; }
    int _n=n, _nrhs=nrhs, _ia=ia, _ja=ja, _ib=ib, _jb=jb;
    g_sca.Psgetrs(&trans, &_n, &_nrhs,
                  A, &_ia, &_ja, desca, ipiv,
                  B, &_ib, &_jb, descb, info, 1);
}

void fb_pdgetrs(char trans, int n, int nrhs,
                const double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                const int *ipiv,
                double *B, int ib, int jb, const fb_scalapack_desc_t descb,
                int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pdgetrs) { if (info) *info=-1; return; }
    int _n=n, _nrhs=nrhs, _ia=ia, _ja=ja, _ib=ib, _jb=jb;
    g_sca.Pdgetrs(&trans, &_n, &_nrhs,
                  A, &_ia, &_ja, desca, ipiv,
                  B, &_ib, &_jb, descb, info, 1);
}

void fb_psgesv(int n, int nrhs,
               float *A, int ia, int ja, const fb_scalapack_desc_t desca,
               int *ipiv,
               float *B, int ib, int jb, const fb_scalapack_desc_t descb,
               int *info,
               const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Psgesv) { if (info) *info=-1; return; }
    int _n=n, _nrhs=nrhs, _ia=ia, _ja=ja, _ib=ib, _jb=jb;
    g_sca.Psgesv(&_n, &_nrhs, A, &_ia, &_ja, desca, ipiv,
                 B, &_ib, &_jb, descb, info);
}

void fb_pdgesv(int n, int nrhs,
               double *A, int ia, int ja, const fb_scalapack_desc_t desca,
               int *ipiv,
               double *B, int ib, int jb, const fb_scalapack_desc_t descb,
               int *info,
               const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pdgesv) { if (info) *info=-1; return; }
    int _n=n, _nrhs=nrhs, _ia=ia, _ja=ja, _ib=ib, _jb=jb;
    g_sca.Pdgesv(&_n, &_nrhs, A, &_ia, &_ja, desca, ipiv,
                 B, &_ib, &_jb, descb, info);
}

void fb_pspotrf(char uplo, int n,
                float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pspotrf) { if (info) *info=-1; return; }
    int _n=n, _ia=ia, _ja=ja;
    g_sca.Pspotrf(&uplo, &_n, A, &_ia, &_ja, desca, info, 1);
}

void fb_pdpotrf(char uplo, int n,
                double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pdpotrf) { if (info) *info=-1; return; }
    int _n=n, _ia=ia, _ja=ja;
    g_sca.Pdpotrf(&uplo, &_n, A, &_ia, &_ja, desca, info, 1);
}

void fb_psgeqrf(int m, int n,
                float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                float *tau, float *work, int lwork,
                int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Psgeqrf) { if (info) *info=-1; return; }
    int _m=m, _n=n, _ia=ia, _ja=ja, _lwork=lwork;
    g_sca.Psgeqrf(&_m, &_n, A, &_ia, &_ja, desca, tau, work, &_lwork, info);
}

void fb_pdgeqrf(int m, int n,
                double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                double *tau, double *work, int lwork,
                int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pdgeqrf) { if (info) *info=-1; return; }
    int _m=m, _n=n, _ia=ia, _ja=ja, _lwork=lwork;
    g_sca.Pdgeqrf(&_m, &_n, A, &_ia, &_ja, desca, tau, work, &_lwork, info);
}

void fb_psgesvd(char jobu, char jobvt,
                int m, int n,
                float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                float *S,
                float *U,  int iu,  int ju,  const fb_scalapack_desc_t descu,
                float *VT, int ivt, int jvt, const fb_scalapack_desc_t descvt,
                float *work, int lwork,
                int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Psgesvd) { if (info) *info=-1; return; }
    int _m=m, _n=n, _ia=ia, _ja=ja, _iu=iu, _ju=ju, _ivt=ivt, _jvt=jvt, _lwork=lwork;
    g_sca.Psgesvd(&jobu, &jobvt, &_m, &_n,
                  A,  &_ia,  &_ja,  desca,  S,
                  U,  &_iu,  &_ju,  descu,
                  VT, &_ivt, &_jvt, descvt,
                  work, &_lwork, info,
                  1, 1);
}

void fb_pdgesvd(char jobu, char jobvt,
                int m, int n,
                double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                double *S,
                double *U,  int iu,  int ju,  const fb_scalapack_desc_t descu,
                double *VT, int ivt, int jvt, const fb_scalapack_desc_t descvt,
                double *work, int lwork,
                int *info,
                const fb_distributed_ctx_t *ctx)
{
    (void)ctx;
    if (!g_sca.Pdgesvd) { if (info) *info=-1; return; }
    int _m=m, _n=n, _ia=ia, _ja=ja, _iu=iu, _ju=ju, _ivt=ivt, _jvt=jvt, _lwork=lwork;
    g_sca.Pdgesvd(&jobu, &jobvt, &_m, &_n,
                  A,  &_ia,  &_ja,  desca,  S,
                  U,  &_iu,  &_ju,  descu,
                  VT, &_ivt, &_jvt, descvt,
                  work, &_lwork, info,
                  1, 1);
}

/* ─────────────────────────────────────────────────────────────────────────
 * Sequence executor routing hook  (TODO — wire when sequence executor lands)
 *
 * When the sequence executor needs to dispatch an op_id in the range
 * [FB_OP_SCALAPACK_BASE, FB_OP_SCALAPACK_END), it will call a function here
 * that builds a tagged union from the op_chain node and forwards to the
 * appropriate Fortran shim above.
 *
 * Implementation deferred until:
 *   1. Individual ScaLAPACK op IDs (FB_OP_PSGEMM, FB_OP_PSGETRF, …) are
 *      assigned in the main op-ID header.
 *   2. The sequence executor's op_chain_node_t gains a dist_args member.
 *
 * At that point, add:
 *     #include "../../include/faster-blaster/op_ids.h"
 *     int fb_scalapack_dispatch_op(uint32_t op_id,
 *                                   fb_distributed_ctx_t *ctx,
 *                                   fb_scalapack_op_args_t *args);
 * and implement the switch on op_id.
 * ───────────────────────────────────────────────────────────────────────── */
