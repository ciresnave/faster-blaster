#

/**
 * SORGQL - Generate m-by-n matrix Q with orthonormal columns from QL factorization
 * 
 * Generates m-by-n orthogonal matrix Q from QL factorization via SGEQLF.
 * Q is the product of k Householder reflectors applied in reverse order.
 * 
 * Parameters:
 *   m, n  : Dimensions of Q (m >= n)
 *   k     : Number of elementary reflectors (k <= min(m,n))
 *   A     : QL factorization from SGEQLF (input/output)
 *   lda   : Leading dimension of A
 *   tau   : Scalar factors from SGEQLF
 *   work  : Workspace
 *   lwork : Size of workspace
 *   info  : Status (0=success, <0=invalid param)
 */
void sorgql_(const int *m, const int *n, const int *k, float *A, const int *lda,
             const float *tau, float *work, const int *lwork, int *info)
{
    int i, j, l, irows;
    float aii;
    
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
    
    /* Initialize Q to identity */
    for (j = 0; j < *n; j++) {
        for (i = 0; i < *m; i++) {
            A[i + j * *lda] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    /* Generate Q from elementary reflectors in QL form */
    /* QL is like QR but from the bottom-right */
    /* Apply reflectors in forward order starting from last */
    for (l = 0; l < *k; l++) {
        if (tau[l] == 0.0f) continue;
        
        int row = *m - *k + l;  /* Position of reflector */
        if (row < 0 || row >= *m) continue;
        
        aii = A[row + l * *lda];
        A[row + l * *lda] = 1.0f;
        
        /* Apply H_l to rows 0:row and columns l:n-1 */
        for (j = l; j < *n; j++) {
            float vnorm = 0.0f;
            for (i = 0; i <= row; i++) {
                vnorm += A[i + l * *lda] * A[i + j * *lda];
            }
            vnorm *= tau[l];
            for (i = 0; i <= row; i++) {
                A[i + j * *lda] -= vnorm * A[i + l * *lda];
            }
        }
        
        A[row + l * *lda] = aii;
    }
}
