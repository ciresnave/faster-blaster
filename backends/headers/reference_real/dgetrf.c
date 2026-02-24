#

void dgetrf_(const int *m, const int *n, double *A, const int *lda,
             int *ipiv, int *info)
{
    int i, j, k, minmn, ipiv_k;
    double pivot_val, abs_val, inv_diag, temp, akj;
    
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
    
    minmn = (*m < *n) ? *m : *n;
    
    /* LU factorization with partial pivoting */
    for (k = 0; k < minmn; k++) {
        /* Find pivot */
        ipiv_k = k;
        pivot_val = fabs(A[k + k * *lda]);
        
        for (i = k + 1; i < *m; i++) {
            abs_val = fabs(A[i + k * *lda]);
            if (abs_val > pivot_val) {
                ipiv_k = i;
                pivot_val = abs_val;
            }
        }
        
        ipiv[k] = ipiv_k + 1;  /* Store 1-based pivot index */
        
        /* Check for singular matrix */
        if (pivot_val == 0.0) {
            if (*info == 0) {
                *info = k + 1;
            }
            continue;
        }
        
        /* Swap rows k and ipiv_k */
        if (ipiv_k != k) {
            for (j = 0; j < *n; j++) {
                temp = A[k + j * *lda];
                A[k + j * *lda] = A[ipiv_k + j * *lda];
                A[ipiv_k + j * *lda] = temp;
            }
        }
        
        /* Scale column k (compute L) */
        inv_diag = 1.0 / A[k + k * *lda];
        for (i = k + 1; i < *m; i++) {
            A[i + k * *lda] *= inv_diag;
        }
        
        /* Update trailing submatrix */
        for (j = k + 1; j < *n; j++) {
            akj = A[k + j * *lda];
            for (i = k + 1; i < *m; i++) {
                A[i + j * *lda] -= A[i + k * *lda] * akj;
            }
        }
    }
}
