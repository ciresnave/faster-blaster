/**
 * @file dtrmm.c
 * @brief Reference implementation of DTRMM (Double precision triangular matrix multiply)
 *
 * DTRMM: B := alpha*A*B (double precision)
 */

#include <blas_reference.h>

/**
 * Double precision triangular matrix multiply
 */
void dtrmm_ref(char side, char uplo, char trans, char diag, int m, int n, double alpha,
               const double *A, int lda, double *B, int ldb) {
    
    if (m <= 0 || n <= 0) return;
    if (alpha == 0.0) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                B[i + j * ldb] = 0.0;
            }
        }
        return;
    }
    
    int is_left = (side == 'L' || side == 'l');
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_unit = (diag == 'U' || diag == 'u');
    int is_notrans = (trans == 'N' || trans == 'n');
    
    if (is_left) {
        if (is_notrans) {
            for (int j = 0; j < n; j++) {
                for (int i = m - 1; i >= 0; i--) {
                    double sum = alpha * B[i + j * ldb];
                    
                    if (is_lower) {
                        for (int k = i + 1; k < m; k++) {
                            sum += A[i + k * lda] * B[k + j * ldb];
                        }
                    } else {
                        for (int k = 0; k < i; k++) {
                            sum += A[i + k * lda] * B[k + j * ldb];
                        }
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
        for (int j = 0; j < n; j++) {
            if (alpha != 1.0) {
                for (int i = 0; i < m; i++) {
                    B[i + j * ldb] *= alpha;
                }
            }
            
            if (is_lower) {
                for (int k = 0; k < j; k++) {
                    double akj = A[k + j * lda];
                    if (akj != 0.0) {
                        for (int i = 0; i < m; i++) {
                            B[i + j * ldb] += akj * B[i + k * ldb];
                        }
                    }
                }
            } else {
                for (int k = j + 1; k < n; k++) {
                    double akj = A[k + j * lda];
                    if (akj != 0.0) {
                        for (int i = 0; i < m; i++) {
                            B[i + j * ldb] += akj * B[i + k * ldb];
                        }
                    }
                }
            }
        }
    }
}
