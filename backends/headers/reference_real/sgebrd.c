#
#include <string.h>
#include <blas_reference.h>

/**
 * SGEBRD - Reduce general matrix to bidiagonal form
 *
 * Reduces a general m-by-n matrix A to upper bidiagonal form B by orthogonal
 * transformations: A = Q * B * P^T
 * The diagonal elements (d) and off-diagonal elements (e) are stored.
 */
void sgebrd_(const int *m, const int *n, float *A, const int *lda,
             float *d, float *e, float *tauq, float *taup, float *work,
             const int *lwork, int *info)
{
    int i, minmn, nv, nh, nq, np;
    float alpha, one = 1.0f, zero = 0.0f;
    int incx = 1;
    
    *info = 0;
    
    /* Input validation */
    if (*m < 0) {
        *info = -1;
        return;
    }
    if (*n < 0) {
        *info = -2;
        return;
    }
    if (*lda < (*m > 1 ? *m : 1)) {
        *info = -4;
        return;
    }
    if (*lwork < 1 && *lwork != -1) {
        *info = -9;
        return;
    }
    
    /* Workspace query */
    if (*lwork == -1) {
        work[0] = (float)((*m > *n) ? (*m * 64) : (*n * 64));
        return;
    }
    
    minmn = (*m < *n) ? *m : *n;
    
    /* Quick return */
    if (minmn == 0) {
        return;
    }
    
    /* Reduce matrix to bidiagonal form */
    for (i = 0; i < minmn; i++) {
        
        /* Reduce A(i:m-1, i) to a single element (column reduction) */
        if (i < *m - 1) {
            nv = *m - i;
            alpha = A[i + i * (*lda)];
            
            slarfg_(&nv, &alpha, &A[(i + 1) + i * (*lda)], &incx, &tauq[i]);
            d[i] = alpha;
            A[i + i * (*lda)] = one;
            
            /* Apply H(i) to A(i:m-1, i+1:n) from the left */
            if (i < *n - 1) {
                np = *n - i - 1;
                sgemv_("T", &nv, &np, &one, &A[i + (i + 1) * (*lda)], lda,
                        &A[i + i * (*lda)], &incx, &zero, work, &incx);
                
                alpha = -tauq[i];
                sger_(&nv, &np, &alpha, &A[i + i * (*lda)], &incx, work, &incx,
                      &A[i + (i + 1) * (*lda)], lda);
            }
            
            A[i + i * (*lda)] = d[i];
        } else {
            d[i] = A[i + i * (*lda)];
        }
        
        /* Reduce A(i, i+1:n) to a single element (row reduction) */
        if (i < *n - 1) {
            nh = *n - i - 1;
            alpha = A[i + (i + 1) * (*lda)];
            
            slarfg_(&nh, &alpha, &A[i + (i + 2) * (*lda)], lda, &taup[i]);
            e[i] = alpha;
            A[i + (i + 1) * (*lda)] = one;
            
            /* Apply G(i) to A(i+1:m-1, i+1:n) from the right */
            if (i < *m - 1) {
                nq = *m - i - 1;
                sgemv_("N", &nq, &nh, &one, &A[(i + 1) + (i + 1) * (*lda)], lda,
                        &A[i + (i + 1) * (*lda)], lda, &zero, work, &incx);
                
                alpha = -taup[i];
                sger_(&nq, &nh, &alpha, work, &incx, &A[i + (i + 1) * (*lda)], lda,
                      &A[(i + 1) + (i + 1) * (*lda)], lda);
            }
            
            A[i + (i + 1) * (*lda)] = e[i];
        } else {
            if (i < minmn - 1) {
                e[i] = A[i + (i + 1) * (*lda)];
            }
        }
    }
}
