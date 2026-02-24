#
#include <string.h>

/**
 * SORGLQ - Generate m-by-n matrix Q with orthonormal rows (single precision)
 * 
 * Generates m-by-n orthogonal matrix Q from LQ factorization via SGELQF.
 * Q consists of the first k Householder reflectors for row reduction.
 * 
 * Parameters:
 *   m, n  : Dimensions of Q (m <= n)
 *   k     : Number of elementary reflectors (k <= m)
 *   A     : LQ factorization from SGELQF (input/output)
 *   lda   : Leading dimension of A
 *   tau   : Scalar factors from SGELQF
 *   work  : Workspace
 *   lwork : Size of workspace
 *   info  : Status (0=success, <0=invalid param)
 */
void sorglq_(const int *m, const int *n, const int *k, float *A, const int *lda,
             const float *tau, float *work, const int *lwork, int *info)
{
    int i, j, l;
    float aii;
    
    *info = 0;
    
    /* Validate inputs */
    if (*m < 0) {
        *info = -1;
        return;
    }
    if (*n < 0 || *n < *m) {
        *info = -2;
        return;
    }
    if (*k < 0 || *k > *m) {
        *info = -3;
        return;
    }
    if (*lda < (*m > 1 ? *m : 1)) {
        *info = -5;
        return;
    }
    
    if (*m == 0 || *n == 0) {
        return;
    }
    
    /* Initialize Q to identity */
    for (i = 0; i < *m; i++) {
        for (j = 0; j < *n; j++) {
            A[i + j * *lda] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    /* Generate Q from elementary reflectors (row-wise) */
    /* Apply reflectors in reverse order */
    for (l = *k - 1; l >= 0; l--) {
        if (tau[l] == 0.0f) continue;
        
        /* Set A[l,l] = 1 */
        aii = A[l + l * *lda];
        A[l + l * *lda] = 1.0f;
        
        /* Apply H_l to rows l:m-1 and columns l:n-1 */
        for (i = l + 1; i < *m; i++) {
            float vnorm = 0.0f;
            for (j = l; j < *n; j++) {
                vnorm += A[l + j * *lda] * A[i + j * *lda];
            }
            vnorm *= tau[l];
            for (j = l; j < *n; j++) {
                A[i + j * *lda] -= vnorm * A[l + j * *lda];
            }
        }
        
        A[l + l * *lda] = aii;
    }
}
