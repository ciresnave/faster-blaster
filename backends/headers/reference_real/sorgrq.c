#

/**
 * SORGRQ - Generate m-by-n matrix Q with orthonormal rows from RQ factorization
 * 
 * Generates m-by-n orthogonal matrix Q from RQ factorization via SGERQF.
 * Q consists of the first k Householder reflectors for row reduction.
 * 
 * Parameters:
 *   m, n  : Dimensions of Q (m <= n)
 *   k     : Number of elementary reflectors (k <= m)
 *   A     : RQ factorization from SGERQF (input/output)
 *   lda   : Leading dimension of A
 *   tau   : Scalar factors from SGERQF
 *   work  : Workspace
 *   lwork : Size of workspace
 *   info  : Status (0=success, <0=invalid param)
 */
void sorgrq_(const int *m, const int *n, const int *k, float *A, const int *lda,
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
    
    /* Generate Q from elementary reflectors (row-wise, RQ form) */
    /* RQ factorization stores reflectors in reverse (from bottom-right) */
    for (l = *k - 1; l >= 0; l--) {
        if (tau[l] == 0.0f) continue;
        
        int col = *n - *k + l;  /* Column position of reflector */
        if (col < 0 || col >= *n) continue;
        
        aii = A[l + col * *lda];
        A[l + col * *lda] = 1.0f;
        
        /* Apply H_l to rows l:m-1 and columns 0:col */
        for (i = l + 1; i < *m; i++) {
            float vnorm = 0.0f;
            for (j = 0; j <= col; j++) {
                vnorm += A[l + j * *lda] * A[i + j * *lda];
            }
            vnorm *= tau[l];
            for (j = 0; j <= col; j++) {
                A[i + j * *lda] -= vnorm * A[l + j * *lda];
            }
        }
        
        A[l + col * *lda] = aii;
    }
}
