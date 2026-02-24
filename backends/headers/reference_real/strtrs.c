#include <blas_reference.h>

void strtrs_(const char *uplo, const char *trans, const char *diag, const int *n, const int *nrhs,
                const float *A, const int *lda, float *B, const int *ldb, int *info) {
    
    int nn = *n;
    int nrhs_val = *nrhs;
    int lda_val = *lda;
    int ldb_val = *ldb;
    char uplo_val = *uplo;
    char trans_val = *trans;
    char diag_val = *diag;
    
    if (nn <= 0 || nrhs_val <= 0) {
        *info = 0;
        return;
    }
    
    *info = 0;
    
    int is_lower = (uplo_val == 'L' || uplo_val == 'l');
    int is_notrans = (trans_val == 'N' || trans_val == 'n');
    int is_unit = (diag_val == 'U' || diag_val == 'u');
    
    /* Solve A*X = B or A^T*X = B */
    for (int j = 0; j < nrhs_val; j++) {
        float *b = &B[j * ldb_val];
        
        if (is_notrans) {
            if (is_lower) {
                /* Lower triangular: L*x = b (forward substitution) */
                for (int i = 0; i < nn; i++) {
                    float sum = b[i];
                    for (int k = 0; k < i; k++) {
                        sum -= A[i + k * lda_val] * b[k];
                    }
                    if (!is_unit) {
                        if (A[i + i * lda_val] == 0.0f) {
                            *info = i + 1;
                            return;
                        }
                        sum /= A[i + i * lda_val];
                    }
                    b[i] = sum;
                }
            } else {
                /* Upper triangular: U*x = b (backward substitution) */
                for (int i = nn - 1; i >= 0; i--) {
                    float sum = b[i];
                    for (int k = i + 1; k < nn; k++) {
                        sum -= A[i + k * lda_val] * b[k];
                    }
                    if (!is_unit) {
                        if (A[i + i * lda_val] == 0.0f) {
                            *info = i + 1;
                            return;
                        }
                        sum /= A[i + i * lda_val];
                    }
                    b[i] = sum;
                }
            }
        } else {
            if (is_lower) {
                /* Lower triangular transpose: L^T*x = b (backward substitution) */
                for (int i = nn - 1; i >= 0; i--) {
                    float sum = b[i];
                    for (int k = i + 1; k < nn; k++) {
                        sum -= A[k + i * lda_val] * b[k];
                    }
                    if (!is_unit) {
                        if (A[i + i * lda_val] == 0.0f) {
                            *info = i + 1;
                            return;
                        }
                        sum /= A[i + i * lda_val];
                    }
                    b[i] = sum;
                }
            } else {
                /* Upper triangular transpose: U^T*x = b (forward substitution) */
                for (int i = 0; i < nn; i++) {
                    float sum = b[i];
                    for (int k = 0; k < i; k++) {
                        sum -= A[k + i * lda_val] * b[k];
                    }
                    if (!is_unit) {
                        if (A[i + i * lda_val] == 0.0f) {
                            *info = i + 1;
                            return;
                        }
                        sum /= A[i + i * lda_val];
                    }
                    b[i] = sum;
                }
            }
        }
    }
}
