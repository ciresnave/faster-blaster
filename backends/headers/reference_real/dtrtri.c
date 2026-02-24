/**
 * @file dtrtri.c
 * @brief Reference implementation of DTRTRI (Double precision triangular inversion)
 */

#include <blas_reference.h>

void dtrtri_(const char *uplo, const char *diag, const int *n, double *A, const int *lda, int *info) {
    
    int nn = *n;
    int lda_val = *lda;
    char uplo_val = *uplo;
    char diag_val = *diag;
    
    if (nn <= 0) {
        *info = 0;
        return;
    }
    
    *info = 0;
    
    int is_lower = (uplo_val == 'L' || uplo_val == 'l');
    int is_unit = (diag_val == 'U' || diag_val == 'u');
    
    if (is_lower) {
        for (int j = 0; j < nn; j++) {
            if (!is_unit) {
                if (A[j + j * lda_val] == 0.0) {
                    *info = j + 1;
                    return;
                }
                A[j + j * lda_val] = 1.0 / A[j + j * lda_val];
            }
            double ajj = A[j + j * lda_val];
            
            for (int i = j + 1; i < nn; i++) {
                double sum = 0.0;
                for (int k = j; k < i; k++) {
                    sum -= A[i + k * lda_val] * A[k + j * lda_val];
                }
                A[i + j * lda_val] = sum / ajj;
            }
        }
    } else {
        for (int j = nn - 1; j >= 0; j--) {
            if (!is_unit) {
                if (A[j + j * lda_val] == 0.0) {
                    *info = j + 1;
                    return;
                }
                A[j + j * lda_val] = 1.0 / A[j + j * lda_val];
            }
            double ajj = A[j + j * lda_val];
            
            for (int i = j - 1; i >= 0; i--) {
                double sum = 0.0;
                for (int k = i + 1; k <= j; k++) {
                    sum -= A[i + k * lda_val] * A[k + j * lda_val];
                }
                A[i + j * lda_val] = sum / ajj;
            }
        }
    }
}
