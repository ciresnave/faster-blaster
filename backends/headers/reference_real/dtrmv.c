/**
 * @file dtrmv.c
 * @brief Reference implementation of DTRMV (Double precision triangular matrix-vector multiply)
 *
 * DTRMV: x := A*x (double precision)
 */

#include <blas_reference.h>

/**
 * Double precision triangular matrix-vector multiply
 */
void dtrmv_ref(char uplo, char trans, char diag, int n, const double *A, int lda,
               double *x, int incx) {
    
    if (n <= 0) return;
    
    int is_unit = (diag == 'U' || diag == 'u');
    int is_lower = (uplo == 'L' || uplo == 'l');
    int is_notrans = (trans == 'N' || trans == 'n');
    
    double temp[1000];
    
    if (is_lower && is_notrans) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            if (is_unit) sum = x[i * incx];
            
            for (int j = 0; j <= i; j++) {
                if (!is_unit || j < i) {
                    sum += A[i + j * lda] * x[j * incx];
                }
            }
            temp[i] = sum;
        }
    } else if (!is_lower && is_notrans) {
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            if (is_unit) sum = x[i * incx];
            
            for (int j = i; j < n; j++) {
                if (!is_unit || j > i) {
                    sum += A[i + j * lda] * x[j * incx];
                }
            }
            temp[i] = sum;
        }
    } else if (is_lower && !is_notrans) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            if (is_unit) sum = x[j * incx];
            
            for (int i = j; i < n; i++) {
                if (!is_unit || i > j) {
                    sum += A[i + j * lda] * x[i * incx];
                }
            }
            temp[j] = sum;
        }
    } else {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            if (is_unit) sum = x[j * incx];
            
            for (int i = 0; i <= j; i++) {
                if (!is_unit || i < j) {
                    sum += A[i + j * lda] * x[i * incx];
                }
            }
            temp[j] = sum;
        }
    }
    
    for (int i = 0; i < n; i++) {
        x[i * incx] = temp[i];
    }
}
