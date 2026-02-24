/**
 * @file dtrsm.c
 * @brief Reference implementation of DTRSM (Double precision triangular solve with matrix RHS)
 *
 * DTRSM: B := alpha*A^(-1)*B (double precision)
 */

#include <blas_reference.h>

/**
 * Double precision triangular solve with matrix RHS
 */
void dtrsm_ref(char side, char uplo, char trans, char diag, int m, int n, double alpha,
               const double *A, int lda, double *B, int ldb) {
    
    if (m <= 0 || n <= 0) return;
    
    if (alpha != 1.0) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                B[i + j * ldb] *= alpha;
            }
        }
    }
    
    int is_left = (side == 'L' || side == 'l');
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_unit = (diag == 'U' || diag == 'u');
    
    if (is_left) {
        for (int j = 0; j < n; j++) {
            if (is_lower) {
                for (int i = 0; i < m; i++) {
                    double sum = B[i + j * ldb];
                    
                    for (int k = 0; k < i; k++) {
                        sum -= A[i + k * lda] * B[k + j * ldb];
                    }
                    
                    if (!is_unit) {
                        double aii = A[i + i * lda];
                        if (aii != 0.0) sum /= aii;
                    }
                    
                    B[i + j * ldb] = sum;
                }
            } else {
                for (int i = m - 1; i >= 0; i--) {
                    double sum = B[i + j * ldb];
                    
                    for (int k = i + 1; k < m; k++) {
                        sum -= A[i + k * lda] * B[k + j * ldb];
                    }
                    
                    if (!is_unit) {
                        double aii = A[i + i * lda];
                        if (aii != 0.0) sum /= aii;
                    }
                    
                    B[i + j * ldb] = sum;
                }
            }
        }
    } else {
        if (is_lower) {
            for (int j = n - 1; j >= 0; j--) {
                for (int i = 0; i < m; i++) {
                    double sum = B[i + j * ldb];
                    
                    for (int k = j + 1; k < n; k++) {
                        sum -= B[i + k * ldb] * A[j + k * lda];
                    }
                    
                    if (!is_unit) {
                        double ajj = A[j + j * lda];
                        if (ajj != 0.0) sum /= ajj;
                    }
                    
                    B[i + j * ldb] = sum;
                }
            }
        } else {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    double sum = B[i + j * ldb];
                    
                    for (int k = 0; k < j; k++) {
                        sum -= B[i + k * ldb] * A[k + j * lda];
                    }
                    
                    if (!is_unit) {
                        double ajj = A[j + j * lda];
                        if (ajj != 0.0) sum /= ajj;
                    }
                    
                    B[i + j * ldb] = sum;
                }
            }
        }
    }
}
