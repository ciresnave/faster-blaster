/**
 * @file dsyr.c
 * @brief Reference implementation of DSYR (Double precision symmetric rank-1 update)
 *
 * DSYR: A := alpha*x*x^T + A where A is symmetric
 */

#include <blas_reference.h>

void dsyr_ref(char uplo, int n, double alpha, const double *x, int incx,
              double *A, int lda) {
    
    if (n <= 0) return;
    if (alpha == 0.0) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    
    if (is_lower) {
        int jx = 0;
        for (int j = 0; j < n; j++) {
            double xj = alpha * x[jx];
            int ix = jx;
            for (int i = j; i < n; i++) {
                A[i + j * lda] += x[ix] * xj;
                ix += incx;
            }
            jx += incx;
        }
    } else {
        int jx = 0;
        for (int j = 0; j < n; j++) {
            double xj = alpha * x[jx];
            int ix = 0;
            for (int i = 0; i <= j; i++) {
                A[i + j * lda] += x[ix] * xj;
                ix += incx;
            }
            jx += incx;
        }
    }
}
