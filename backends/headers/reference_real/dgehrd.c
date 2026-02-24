#
#include <string.h>
#include <blas_reference.h>

/**
 * DGEHRD - Reduce general matrix to upper Hessenberg form (double precision)
 *
 * Reduces a general matrix A to upper Hessenberg form H by an orthogonal
 * similarity transformation: A = Q * H * Q^T
 * Only the submatrix A(ilo:ihi, ilo:ihi) is reduced to Hessenberg form.
 */
void dgehrd_(const int *n, const int *ilo, const int *ihi, double *A, const int *lda,
             double *tau, double *work, const int *lwork, int *info)
{
    int i, nv, ncol, nrow;
    double alpha, one = 1.0, zero = 0.0, neg_one = -1.0;
    int incx = 1;
    
    *info = 0;
    
    /* Input validation */
    if (*n < 0) {
        *info = -1;
        return;
    }
    if (*ilo < 1 || *ilo > *n) {
        *info = -2;
        return;
    }
    if (*ihi < *ilo || *ihi > *n) {
        *info = -3;
        return;
    }
    if (*lda < (*n > 1 ? *n : 1)) {
        *info = -5;
        return;
    }
    if (*lwork < 1 && *lwork != -1) {
        *info = -7;
        return;
    }
    
    /* Handle workspace query */
    if (*lwork == -1) {
        work[0] = (double)(*n > 2 ? (*n * 64) : 1);
        return;
    }
    
    /* Quick return */
    if (*ihi - *ilo + 1 <= 1) {
        return;
    }
    
    /* Reduce columns ilo to ihi-1 to Hessenberg form */
    for (i = *ilo; i < *ihi; i++) {
        /* Generate elementary reflector H(i) to annihilate A(i+2:ihi, i) */
        nv = *ihi - i;
        alpha = A[(i + 1) + (i - 1) * (*lda)];
        
        dlarfg_(&nv, &alpha, &A[(i + 2) + (i - 1) * (*lda)], &incx, &tau[i - *ilo]);
        
        A[(i + 1) + (i - 1) * (*lda)] = one;
        
        /* Apply H(i) to A(1:ihi, i+1:n) from the right */
        if (*ihi < *n) {
            ncol = *n - i;
            dgemv_("T", &nv, &ncol, &one, &A[(i + 1) + i * (*lda)], lda,
                    &A[(i + 1) + (i - 1) * (*lda)], &incx, &zero, work, &incx);
            
            alpha = -tau[i - *ilo];
            dger_(&nv, &ncol, &alpha, &A[(i + 1) + (i - 1) * (*lda)], &incx, work, &incx,
                  &A[(i + 1) + i * (*lda)], lda);
        }
        
        /* Apply H(i) to A(1:ihi, 1:i) from the left */
        if (i > 1) {
            nrow = i;
            dgemv_("N", &nrow, &nv, &one, &A[0 + i * (*lda)], lda,
                    &A[(i + 1) + (i - 1) * (*lda)], &incx, &zero, work, &incx);
            
            alpha = -tau[i - *ilo];
            dger_(&nrow, &nv, &alpha, work, &incx, &A[(i + 1) + (i - 1) * (*lda)], &incx,
                  &A[0 + i * (*lda)], lda);
        }
        
        A[(i + 1) + (i - 1) * (*lda)] = alpha;
    }
}
