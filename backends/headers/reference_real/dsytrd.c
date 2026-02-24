#
#include <string.h>
#include <blas_reference.h>

/**
 * DSYTRD - Reduce symmetric matrix to tridiagonal form (double precision)
 *
 * Reduces a real symmetric matrix A to symmetric tridiagonal form T
 * using orthogonal similarity transformations (Householder reflections).
 * A = Q * T * Q^T where Q is orthogonal and T is tridiagonal.
 */
void dsytrd_(const char *uplo, const int *n, double *A, const int *lda, double *d,
             double *e, double *tau, double *work, const int *lwork, int *info)
{
    int i, k, iinfo, nv;
    double alpha, one = 1.0, zero = 0.0, neg_one = -1.0;
    
    *info = 0;
    
    /* Input validation */
    if ((*uplo != 'U' && *uplo != 'u' && *uplo != 'L' && *uplo != 'l')) {
        *info = -1;
        return;
    }
    if (*n < 0) {
        *info = -2;
        return;
    }
    if (*lda < (*n > 1 ? *n : 1)) {
        *info = -4;
        return;
    }
    if (*lwork < 1 && *lwork != -1) {
        *info = -8;
        return;
    }
    
    /* Handle workspace query */
    if (*lwork == -1) {
        work[0] = (double)(*n > 2 ? (*n * 32) : 1);
        return;
    }
    
    /* Quick return */
    if (*n <= 1) {
        if (*n == 1) {
            d[0] = A[0];
        }
        return;
    }
    
    if ((*uplo == 'U' || *uplo == 'u')) {
        /* Reduce upper triangle */
        for (i = *n - 1; i >= 1; i--) {
            /* Generate elementary reflector H(i) */
            nv = i;
            alpha = A[(i - 1) + i * (*lda)];
            
            dlarfg_(&nv, &alpha, &A[0 + i * (*lda)], &one, &tau[i - 1]);
            
            e[i - 1] = alpha;
            A[(i - 1) + i * (*lda)] = one;
            
            /* Apply H(i) to A[0:i, 0:i] */
            if (i > 0) {
                /* y := A * v */
                dgemv_("N", &i, &i, &one, A, lda, &A[0 + i * (*lda)], &one, 
                        &zero, work, &one);
                
                /* y := y - (1/2) * tau * (v^T*y) * v */
                alpha = -0.5 * tau[i - 1] * ddot_(&i, &A[0 + i * (*lda)], &one, work, &one);
                daxpy_(&i, &alpha, &A[0 + i * (*lda)], &one, work, &one);
                
                /* A := A - v*y^T - y*v^T */
                dger_(&i, &i, &neg_one, &A[0 + i * (*lda)], &one, work, &one, A, lda);
                dger_(&i, &i, &neg_one, work, &one, &A[0 + i * (*lda)], &one, A, lda);
            }
            
            A[(i - 1) + i * (*lda)] = e[i - 1];
            d[i] = A[i + i * (*lda)];
        }
        d[0] = A[0];
        
    } else {
        /* Reduce lower triangle */
        for (i = 0; i < *n - 1; i++) {
            /* Generate elementary reflector H(i) */
            nv = *n - i - 1;
            alpha = A[(i + 1) + i * (*lda)];
            
            dlarfg_(&nv, &alpha, &A[(i + 2) + i * (*lda)], &one, &tau[i]);
            
            e[i] = alpha;
            A[(i + 1) + i * (*lda)] = one;
            
            /* Apply H(i) to A[i+1:n, i+1:n] */
            if (i < *n - 2) {
                /* y := A * v */
                dgemv_("N", &nv, &nv, &one, &A[(i + 1) + (i + 1) * (*lda)], lda,
                        &A[(i + 1) + i * (*lda)], &one, &zero, work, &one);
                
                /* y := y - (1/2) * tau * (v^T*y) * v */
                alpha = -0.5 * tau[i] * ddot_(&nv, &A[(i + 1) + i * (*lda)], &one, work, &one);
                daxpy_(&nv, &alpha, &A[(i + 1) + i * (*lda)], &one, work, &one);
                
                /* A := A - v*y^T - y*v^T */
                dger_(&nv, &nv, &neg_one, &A[(i + 1) + i * (*lda)], &one, work, &one,
                      &A[(i + 1) + (i + 1) * (*lda)], lda);
                dger_(&nv, &nv, &neg_one, work, &one, &A[(i + 1) + i * (*lda)], &one,
                      &A[(i + 1) + (i + 1) * (*lda)], lda);
            }
            
            A[(i + 1) + i * (*lda)] = e[i];
            d[i] = A[i + i * (*lda)];
        }
        d[*n - 1] = A[(*n - 1) + (*n - 1) * (*lda)];
    }
}
