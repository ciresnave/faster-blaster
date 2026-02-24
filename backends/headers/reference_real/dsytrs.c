/* dsytrs - solve symmetric indefinite system using Bunch-Kaufman factorization */
void dsytrs_(const char *uplo, const int *n, const int *nrhs, const double *A, const int *lda,
             const int *ipiv, double *B, const int *ldb, int *info)
{
    int nn = *n;
    int nrhs_val = *nrhs;
    int lda_val = *lda;
    int ldb_val = *ldb;
    char uplo_val = *uplo;
    
    *info = 0;
    
    // Simplified: apply row permutations and solve with stored factors
    // This is a simplified version assuming 1x1 pivots
    
    if (uplo_val == 'U' || uplo_val == 'u') {
        // Apply permutations: P*B = B
        for (int i = 0; i < nn; i++) {
            int kp = ipiv[i] - 1;
            if (kp != i) {
                for (int j = 0; j < nrhs_val; j++) {
                    double temp = B[i + j * ldb_val];
                    B[i + j * ldb_val] = B[kp + j * ldb_val];
                    B[kp + j * ldb_val] = temp;
                }
            }
        }
        
        // Solve with stored D and multipliers (stored in upper triangle)
        for (int i = nn - 1; i >= 0; i--) {
            for (int j = 0; j < nrhs_val; j++) {
                for (int k = 0; k < i; k++) {
                    B[i + j * ldb_val] -= A[k + i * lda_val] * B[k + j * ldb_val];
                }
                double diag = A[i + i * lda_val];
                if (diag != 0.0) {
                    B[i + j * ldb_val] /= diag;
                }
            }
        }
        
        // Solve with stored multipliers (backward)
        for (int i = 0; i < nn; i++) {
            for (int j = 0; j < nrhs_val; j++) {
                for (int k = i + 1; k < nn; k++) {
                    B[i + j * ldb_val] -= A[i + k * lda_val] * B[k + j * ldb_val];
                }
            }
        }
        
        // Apply inverse permutations: B = P^{-1}*B
        for (int i = nn - 1; i >= 0; i--) {
            int kp = ipiv[i] - 1;
            if (kp != i) {
                for (int j = 0; j < nrhs_val; j++) {
                    double temp = B[i + j * ldb_val];
                    B[i + j * ldb_val] = B[kp + j * ldb_val];
                    B[kp + j * ldb_val] = temp;
                }
            }
        }
    } else {
        // Apply permutations: P*B = B
        for (int i = nn - 1; i >= 0; i--) {
            int kp = ipiv[i] - 1;
            if (kp != i) {
                for (int j = 0; j < nrhs_val; j++) {
                    double temp = B[i + j * ldb_val];
                    B[i + j * ldb_val] = B[kp + j * ldb_val];
                    B[kp + j * ldb_val] = temp;
                }
            }
        }
        
        // Solve with stored multipliers (forward)
        for (int i = 0; i < nn; i++) {
            for (int j = 0; j < nrhs_val; j++) {
                for (int k = 0; k < i; k++) {
                    B[i + j * ldb_val] -= A[i + k * lda_val] * B[k + j * ldb_val];
                }
                double diag = A[i + i * lda_val];
                if (diag != 0.0) {
                    B[i + j * ldb_val] /= diag;
                }
            }
        }
        
        // Solve with stored multipliers (backward)
        for (int i = nn - 1; i >= 0; i--) {
            for (int j = 0; j < nrhs_val; j++) {
                for (int k = i + 1; k < nn; k++) {
                    B[i + j * ldb_val] -= A[i + k * lda_val] * B[k + j * ldb_val];
                }
            }
        }
        
        // Apply inverse permutations: B = P^{-1}*B
        for (int i = 0; i < nn; i++) {
            int kp = ipiv[i] - 1;
            if (kp != i) {
                for (int j = 0; j < nrhs_val; j++) {
                    double temp = B[i + j * ldb_val];
                    B[i + j * ldb_val] = B[kp + j * ldb_val];
                    B[kp + j * ldb_val] = temp;
                }
            }
        }
    }
}
