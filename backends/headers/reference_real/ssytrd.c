#
#include <string.h>
#include <blas_reference.h>

/**
 * SSYTRD - Reduce symmetric matrix to tridiagonal form
 *
 * Reduces a real symmetric matrix A to symmetric tridiagonal form T
 * using orthogonal similarity transformations (Householder reflections).
 * A = Q * T * Q^T where Q is orthogonal and T is tridiagonal.
 */
void ssytrd_(const char *uplo, const int *n, float *A, const int *lda, float *d,
             float *e, float *tau, float *work, const int *lwork, int *info)
{
    int i, k, iinfo, nv;
    float alpha, one = 1.0f, zero = 0.0f, neg_one = -1.0f;
    
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
        work[0] = (float)(*n > 2 ? (*n * 32) : 1);
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
            
            slarfg_(&nv, &alpha, &A[0 + i * (*lda)], &one, &tau[i - 1]);
            
            e[i - 1] = alpha;
            A[(i - 1) + i * (*lda)] = one;
            
            /* Apply H(i) to A[0:i, 0:i] */
            if (i > 0) {
                /* y := A * v */
                sgemv_("N", &i, &i, &one, A, lda, &A[0 + i * (*lda)], &one, 
                        &zero, work, &one);
                
                /* y := y - (1/2) * tau * (v^T*y) * v */
                alpha = -0.5f * tau[i - 1] * sdot_(&i, &A[0 + i * (*lda)], &one, work, &one);
                saxpy_(&i, &alpha, &A[0 + i * (*lda)], &one, work, &one);
                
                /* A := A - v*y^T - y*v^T */
                sger_(&i, &i, &neg_one, &A[0 + i * (*lda)], &one, work, &one, A, lda);
                sger_(&i, &i, &neg_one, work, &one, &A[0 + i * (*lda)], &one, A, lda);
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
            
            slarfg_(&nv, &alpha, &A[(i + 2) + i * (*lda)], &one, &tau[i]);
            
            e[i] = alpha;
            A[(i + 1) + i * (*lda)] = one;
            
            /* Apply H(i) to A[i+1:n, i+1:n] */
            if (i < *n - 2) {
                /* y := A * v */
                sgemv_("N", &nv, &nv, &one, &A[(i + 1) + (i + 1) * (*lda)], lda,
                        &A[(i + 1) + i * (*lda)], &one, &zero, work, &one);
                
                /* y := y - (1/2) * tau * (v^T*y) * v */
                alpha = -0.5f * tau[i] * sdot_(&nv, &A[(i + 1) + i * (*lda)], &one, work, &one);
                saxpy_(&nv, &alpha, &A[(i + 1) + i * (*lda)], &one, work, &one);
                
                /* A := A - v*y^T - y*v^T */
                sger_(&nv, &nv, &neg_one, &A[(i + 1) + i * (*lda)], &one, work, &one,
                      &A[(i + 1) + (i + 1) * (*lda)], lda);
                sger_(&nv, &nv, &neg_one, work, &one, &A[(i + 1) + i * (*lda)], &one,
                      &A[(i + 1) + (i + 1) * (*lda)], lda);
            }
            
            A[(i + 1) + i * (*lda)] = e[i];
            d[i] = A[i + i * (*lda)];
        }
        d[*n - 1] = A[(*n - 1) + (*n - 1) * (*lda)];
    }
}
