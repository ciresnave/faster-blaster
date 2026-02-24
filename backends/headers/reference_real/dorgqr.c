#
#include <string.h>

/**
 * DORGQR - Generate an m-by-n matrix Q with orthonormal columns (double precision)
 * 
 * Generates the m-by-n orthogonal matrix Q from QR factorization via DGEQRF.
 * Q consists of the first k Householder reflectors applied sequentially.
 * 
 * Parameters:
 *   m, n  : Dimensions of Q (m >= n)
 *   k     : Number of elementary reflectors (k <= min(m,n))
 *   A     : QR factorization from DGEQRF (input/output)
 *   lda   : Leading dimension of A
 *   tau   : Scalar factors from DGEQRF
 *   work  : Workspace (at least max(1, n) for lwork=-1)
 *   lwork : Size of workspace
 *   info  : Status (0=success, <0=invalid param)
 */
void dorgqr_(const int *m, const int *n, const int *k, double *A, const int *lda,
             const double *tau, double *work, const int *lwork, int *info)
{
    int i, j, l, ib, iws;
    double aii;
    int nb = 64;  /* Block size for efficiency */
    
    *info = 0;
    
    /* Validate inputs */
    if (*m < 0) {
        *info = -1;
        return;
    }
    if (*n < 0 || *n > *m) {
        *info = -2;
        return;
    }
    if (*k < 0 || *k > *n) {
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
    
    /* Initialize Q to identity, then apply reflectors */
    for (j = 0; j < *n; j++) {
        for (i = 0; i < *m; i++) {
            A[i + j * *lda] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    /* Generate Q from elementary reflectors */
    /* Apply Householder reflectors in reverse order */
    for (l = *k - 1; l >= 0; l--) {
        /* Skip if tau = 0 */
        if (tau[l] == 0.0) continue;
        
        /* Set A[l,l] = 1 temporarily */
        aii = A[l + l * *lda];
        A[l + l * *lda] = 1.0;
        
        /* Apply H_l = I - tau_l * v * v^T to columns l:n-1 */
        for (j = l; j < *n; j++) {
            /* Compute projection onto v */
            double vnorm = 0.0;
            for (i = l; i < *m; i++) {
                vnorm += A[i + l * *lda] * A[i + j * *lda];
            }
            vnorm *= tau[l];
            
            /* Update column j */
            for (i = l; i < *m; i++) {
                A[i + j * *lda] -= vnorm * A[i + l * *lda];
            }
        }
        
        /* Restore A[l,l] */
        A[l + l * *lda] = aii;
    }
}
