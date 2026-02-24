/**
 * @file dger.c
 * @brief Reference implementation of DGER (Double precision outer product)
 *
 * DGER: A := alpha*x*y^T + A
 */

#include <blas_reference.h>

/**
 * Double precision outer product (rank-1 update)
 */
void dger_ref(int m, int n, double alpha, const double *x, int incx, const double *y, int incy,
              double *A, int lda) {
    
    if (m <= 0 || n <= 0) return;
    if (alpha == 0.0) return;
    
    int jy = 0;
    for (int j = 0; j < n; j++) {
        double temp = alpha * y[jy];
        int ix = 0;
        for (int i = 0; i < m; i++) {
            A[i + j * lda] += x[ix] * temp;
            ix += incx;
        }
        jy += incy;
    }
}
