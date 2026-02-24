/**
 * @file dgetri.c
 * @brief Reference implementation of DGETRI (Double precision LU-based matrix inversion)
 */

#include <blas_reference.h>
#include <string.h>

/* Static helper: Invert triangular matrix in-place */
static void dtrtri_helper(char uplo, char diag, int n, double *A, int lda, int *info) {
    
    if (n <= 0) {
        *info = 0;
        return;
    }
    
    *info = 0;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_unit = (diag == 'U' || diag == 'u');
    
    if (is_lower) {
        for (int j = 0; j < n; j++) {
            if (!is_unit) {
                if (A[j + j * lda] == 0.0) {
                    *info = j + 1;
                    return;
                }
                A[j + j * lda] = 1.0 / A[j + j * lda];
            }
            double ajj = A[j + j * lda];
            
            for (int i = j + 1; i < n; i++) {
                double sum = 0.0;
                for (int k = j; k < i; k++) {
                    sum -= A[i + k * lda] * A[k + j * lda];
                }
                A[i + j * lda] = sum / ajj;
            }
        }
    } else {
        for (int j = n - 1; j >= 0; j--) {
            if (!is_unit) {
                if (A[j + j * lda] == 0.0) {
                    *info = j + 1;
                    return;
                }
                A[j + j * lda] = 1.0 / A[j + j * lda];
            }
            double ajj = A[j + j * lda];
            
            for (int i = j - 1; i >= 0; i--) {
                double sum = 0.0;
                for (int k = i + 1; k <= j; k++) {
                    sum -= A[i + k * lda] * A[k + j * lda];
                }
                A[i + j * lda] = sum / ajj;
            }
        }
    }
}

void dgetri_(const int *n, double *A, const int *lda, const int *ipiv, int *info) {
    
    int nn = *n;
    int lda_val = *lda;
    
    if (nn <= 0) {
        *info = 0;
        return;
    }
    
    *info = 0;
    
    dtrtri_helper('U', 'N', nn, A, lda_val, info);
    
    if (*info != 0) return;
    
    for (int k = nn - 2; k >= 0; k--) {
        for (int i = k + 1; i < nn; i++) {
            for (int j = 0; j < nn; j++) {
                A[i + j * lda_val] -= A[k + j * lda_val] * A[i + k * lda_val];
            }
        }
    }
    
    for (int i = nn - 1; i >= 0; i--) {
        int ipiv_i = ipiv[i] - 1;
        if (ipiv_i != i) {
            for (int j = 0; j < nn; j++) {
                double temp = A[i + j * lda_val];
                A[i + j * lda_val] = A[ipiv_i + j * lda_val];
                A[ipiv_i + j * lda_val] = temp;
            }
        }
    }
}
