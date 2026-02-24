#

void dpotrs_(const char *uplo, const int *n, const int *nrhs,
             const double *A, const int *lda, double *B, const int *ldb, int *info)
{
    int i, j, k;
    double sum;
    int is_lower = (uplo[0] == 'L' || uplo[0] == 'l');
    
    *info = 0;
    if (*n <= 0 || *nrhs <= 0) return;
    
    if (is_lower) {
        /* Solve L*Y = B (forward substitution) */
        for (j = 0; j < *nrhs; j++) {
            double *b = &B[j * *ldb];
            
            for (i = 0; i < *n; i++) {
                sum = b[i];
                for (k = 0; k < i; k++) {
                    sum -= A[i + k * *lda] * b[k];
                }
                b[i] = sum / A[i + i * *lda];
            }
        }
        
        /* Solve L^T*X = Y (backward substitution) */
        for (j = 0; j < *nrhs; j++) {
            double *b = &B[j * *ldb];
            
            for (i = *n - 1; i >= 0; i--) {
                sum = b[i];
                for (k = i + 1; k < *n; k++) {
                    sum -= A[k + i * *lda] * b[k];
                }
                b[i] = sum / A[i + i * *lda];
            }
        }
    } else {
        /* Solve U^T*Y = B (backward substitution) */
        for (j = 0; j < *nrhs; j++) {
            double *b = &B[j * *ldb];
            
            for (i = *n - 1; i >= 0; i--) {
                sum = b[i];
                for (k = i + 1; k < *n; k++) {
                    sum -= A[i + k * *lda] * b[k];
                }
                b[i] = sum / A[i + i * *lda];
            }
        }
        
        /* Solve U*X = Y (forward substitution) */
        for (j = 0; j < *nrhs; j++) {
            double *b = &B[j * *ldb];
            
            for (i = 0; i < *n; i++) {
                sum = b[i];
                for (k = 0; k < i; k++) {
                    sum -= A[k + i * *lda] * b[k];
                }
                b[i] = sum / A[i + i * *lda];
            }
        }
    }
}
