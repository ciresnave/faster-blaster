#
#include <string.h>
#include <blas_reference.h>

/**
 * DGESVD - Compute singular value decomposition (double precision)
 *
 * Computes the singular value decomposition (SVD) of an m-by-n matrix A:
 * A = U * SIGMA * V^T
 */
void dgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             double *A, const int *lda, double *s, double *u, const int *ldu,
             double *vt, const int *ldvt, double *work, const int *lwork, int *info)
{
    int i, j, minmn, maxmn, iinfo;
    int nru, ncvt, ncc;
    int lwork_gebrd, lwork_bdsqr, lwork_needed;
    double *work_ptr, *tauq, *taup, *d, *e;
    char uplo;
    
    *info = 0;
    
    /* Validate inputs */
    if (*m < 0) { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*lda < (*m > 1 ? *m : 1)) { *info = -5; return; }
    if (*ldu < (*m > 1 ? *m : 1)) { *info = -8; return; }
    if (*ldvt < (*n > 1 ? *n : 1)) { *info = -10; return; }
    if (*lwork < 1 && *lwork != -1) { *info = -12; return; }
    
    /* Quick return */
    if (*m == 0 || *n == 0) {
        if (*lwork == -1) {
            work[0] = 1.0;
        }
        return;
    }
    
    minmn = (*m < *n) ? *m : *n;
    maxmn = (*m > *n) ? *m : *n;
    
    /* Determine workspace needed */
    lwork_gebrd = minmn + maxmn;
    lwork_bdsqr = maxmn;
    lwork_needed = 5 * minmn + maxmn;
    
    if (*lwork == -1) {
        work[0] = (double)lwork_needed;
        return;
    }
    
    if (*lwork < lwork_needed) { *info = -12; return; }
    
    /* Allocate workspace from work array */
    work_ptr = work;
    tauq = work_ptr; work_ptr += minmn;
    taup = work_ptr; work_ptr += minmn;
    d = work_ptr; work_ptr += minmn;
    e = work_ptr; work_ptr += minmn - 1;
    
    /* Step 1: Reduce A to bidiagonal form B = U0^T * A * V0 */
    int lwork_bdr = *lwork - (int)(work_ptr - work);
    dgebrd_(m, n, A, lda, d, e, tauq, taup, work_ptr, &lwork_bdr, &iinfo);
    if (iinfo != 0) {
        *info = iinfo;
        return;
    }
    
    /* Step 2: Compute SVD of bidiagonal matrix B = USVT */
    uplo = (*m >= *n) ? 'U' : 'L';
    
    /* Determine number of rows/columns for bidiagonal SVD output */
    nru = 0;
    ncvt = 0;
    ncc = 0;
    
    /* For now, compute minimal singular vectors */
    if (jobu[0] != 'N') nru = minmn;
    if (jobvt[0] != 'N') ncvt = minmn;
    
    dbdsqr_(&uplo, &minmn, &ncvt, &nru, &ncc, d, e, vt, ldvt, u, ldu,
            (double*)NULL, &ncc, work_ptr, &iinfo);
    if (iinfo != 0) {
        *info = iinfo + 1;
        return;
    }
    
    /* Copy singular values to output */
    for (i = 0; i < minmn; i++) {
        s[i] = fabs(d[i]);
    }
}
