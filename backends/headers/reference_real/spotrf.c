#

void spotrf_(const char *uplo, const int *n, float *A, const int *lda, int *info)
{
    int i, j, k;
    float ajj, inv_diag, aij;
    int is_lower = (uplo[0] == 'L' || uplo[0] == 'l');
    
    *info = 0;
    if (*n <= 0) return;
    
    if (is_lower) {
        /* Cholesky factorization of lower triangular */
        for (j = 0; j < *n; j++) {
            ajj = A[j + j * *lda];
            
            /* Compute L(j,j) = sqrt(A(j,j) - sum_{k=0}^{j-1} L(j,k)^2) */
            for (k = 0; k < j; k++) {
                ajj -= A[j + k * *lda] * A[j + k * *lda];
            }
            
            if (ajj <= 0.0f) {
                *info = j + 1;
                return;
            }
            
            A[j + j * *lda] = sqrtf(ajj);
            
            /* Update column j+1:n */
            if (j < *n - 1) {
                inv_diag = 1.0f / A[j + j * *lda];
                for (i = j + 1; i < *n; i++) {
                    aij = A[i + j * *lda];
                    for (k = 0; k < j; k++) {
                        aij -= A[i + k * *lda] * A[j + k * *lda];
                    }
                    A[i + j * *lda] = aij * inv_diag;
                }
            }
        }
    } else {
        /* Cholesky factorization of upper triangular */
        for (j = 0; j < *n; j++) {
            ajj = A[j + j * *lda];
            
            /* Compute U(j,j) = sqrt(A(j,j) - sum_{k=0}^{j-1} U(k,j)^2) */
            for (k = 0; k < j; k++) {
                ajj -= A[k + j * *lda] * A[k + j * *lda];
            }
            
            if (ajj <= 0.0f) {
                *info = j + 1;
                return;
            }
            
            A[j + j * *lda] = sqrtf(ajj);
            
            /* Update row j, columns j+1:n */
            if (j < *n - 1) {
                inv_diag = 1.0f / A[j + j * *lda];
                for (i = j + 1; i < *n; i++) {
                    aij = A[j + i * *lda];
                    for (k = 0; k < j; k++) {
                        aij -= A[k + j * *lda] * A[k + i * *lda];
                    }
                    A[j + i * *lda] = aij * inv_diag;
                }
            }
        }
    }
}
