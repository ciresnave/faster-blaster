/**
 * @file dtrtrs.c
 * @brief Reference implementation of DTRTRS (Double precision triangular solve)
 */

#include <blas_reference.h>

void dtrtrs_(const char *uplo, const char *trans, const char *diag, const int *n, const int *nrhs,
                const double *A, const int *lda, double *B, const int *ldb, int *info) {
    
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
    
    for (int j = 0; j < nrhs_val; j++) {
        double *b = &B[j * ldb_val];
        
        if (is_notrans) {
            if (is_lower) {
                for (int i = 0; i < nn; i++) {
                    double sum = b[i];
                    for (int k = 0; k < i; k++) {
                        sum -= A[i + k * lda_val] * b[k];
                    }
                    if (!is_unit) {
                        if (A[i + i * lda_val] == 0.0) {
                            *info = i + 1;
                            return;
                        }
                        sum /= A[i + i * lda_val];
                    }
                    b[i] = sum;
                }
            } else {
                for (int i = nn - 1; i >= 0; i--) {
                    double sum = b[i];
                    for (int k = i + 1; k < nn; k++) {
                        sum -= A[i + k * lda_val] * b[k];
                    }
                    if (!is_unit) {
                        if (A[i + i * lda_val] == 0.0) {
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
                for (int i = nn - 1; i >= 0; i--) {
                    double sum = b[i];
                    for (int k = i + 1; k < nn; k++) {
                        sum -= A[k + i * lda_val] * b[k];
                    }
                    if (!is_unit) {
                        if (A[i + i * lda_val] == 0.0) {
                            *info = i + 1;
                            return;
                        }
                        sum /= A[i + i * lda_val];
                    }
                    b[i] = sum;
                }
            } else {
                for (int i = 0; i < nn; i++) {
                    double sum = b[i];
                    for (int k = 0; k < i; k++) {
                        sum -= A[k + i * lda_val] * b[k];
                    }
                    if (!is_unit) {
                        if (A[i + i * lda_val] == 0.0) {
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
