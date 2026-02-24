#

void dgetrs_(const char *trans, const int *n, const int *nrhs,
             const double *A, const int *lda, const int *ipiv,
             double *B, const int *ldb, int *info)
{
    int i, j, k, ipiv_i;
    double sum, temp;
    int is_notrans = (trans[0] == 'N' || trans[0] == 'n');
    
    *info = 0;
    if (*n <= 0 || *nrhs <= 0) return;
    
    for (j = 0; j < *nrhs; j++) {
        double *b = &B[j * *ldb];
        
        /* Apply row permutations */
        for (i = 0; i < *n; i++) {
            ipiv_i = ipiv[i] - 1;  /* Convert to 0-based */
            if (ipiv_i != i) {
                temp = b[i];
                b[i] = b[ipiv_i];
                b[ipiv_i] = temp;
            }
        }
    }
    
    if (is_notrans) {
        /* Solve A*X = B: L*U*X = B */
        for (j = 0; j < *nrhs; j++) {
            double *b = &B[j * *ldb];
            
            /* Solve L*y = P*b (forward substitution) */
            for (i = 0; i < *n; i++) {
                sum = b[i];
                for (k = 0; k < i; k++) {
                    sum -= A[i + k * *lda] * b[k];
                }
                b[i] = sum;
            }
            
            /* Solve U*x = y (backward substitution) */
            for (i = *n - 1; i >= 0; i--) {
                sum = b[i];
                for (k = i + 1; k < *n; k++) {
                    sum -= A[i + k * *lda] * b[k];
                }
                b[i] = sum / A[i + i * *lda];
            }
        }
    } else {
        /* Solve A^T*X = B: U^T*L^T*X = B */
        for (j = 0; j < *nrhs; j++) {
            double *b = &B[j * *ldb];
            
            /* Solve U^T*y = P*b (backward substitution) */
            for (i = *n - 1; i >= 0; i--) {
                sum = b[i];
                for (k = i + 1; k < *n; k++) {
                    sum -= A[k + i * *lda] * b[k];
                }
                b[i] = sum / A[i + i * *lda];
            }
            
            /* Solve L^T*x = y (forward substitution) */
            for (i = 0; i < *n; i++) {
                sum = b[i];
                for (k = 0; k < i; k++) {
                    sum -= A[k + i * *lda] * b[k];
                }
                b[i] = sum;
            }
        }
    }
}
