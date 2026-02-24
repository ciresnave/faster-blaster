/**
 * @file strtri.c
 * @brief Reference implementation of STRTRI (Triangular matrix inversion)
 *
 * STRTRI computes the inverse of a triangular matrix A
 */

#include <blas_reference.h>

void strtri_(const char *uplo, const char *diag, const int *n, float *A, const int *lda, int *info) {
    
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
        /* Invert lower triangular */
        for (int j = 0; j < nn; j++) {
            if (!is_unit) {
                if (A[j + j * lda_val] == 0.0f) {
                    *info = j + 1;
                    return;
                }
                A[j + j * lda_val] = 1.0f / A[j + j * lda_val];
            }
            float ajj = A[j + j * lda_val];
            
            for (int i = j + 1; i < nn; i++) {
                float sum = 0.0f;
                for (int k = j; k < i; k++) {
                    sum -= A[i + k * lda_val] * A[k + j * lda_val];
                }
                A[i + j * lda_val] = sum / ajj;
            }
        }
    } else {
        /* Invert upper triangular */
        for (int j = nn - 1; j >= 0; j--) {
            if (!is_unit) {
                if (A[j + j * lda_val] == 0.0f) {
                    *info = j + 1;
                    return;
                }
                A[j + j * lda_val] = 1.0f / A[j + j * lda_val];
            }
            float ajj = A[j + j * lda_val];
            
            for (int i = j - 1; i >= 0; i--) {
                float sum = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    sum -= A[i + k * lda_val] * A[k + j * lda_val];
                }
                A[i + j * lda_val] = sum / ajj;
            }
        }
    }
}
