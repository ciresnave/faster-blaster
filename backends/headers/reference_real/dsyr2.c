/**
 * @file dsyr2.c
 * @brief Reference implementation of DSYR2 (Double precision symmetric rank-2 update)
 */

#include <blas_reference.h>

void dsyr2_ref(char uplo, int n, double alpha, const double *x, int incx,
               const double *y, int incy, double *A, int lda) {
    
    if (n <= 0 || alpha == 0.0) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    
    if (is_lower) {
        int jx = 0, jy = 0;
        for (int j = 0; j < n; j++) {
            double xj = alpha * x[jx];
            double yj = alpha * y[jy];
            
            int ix = jx, iy = jy;
            for (int i = j; i < n; i++) {
                A[i + j * lda] += x[ix] * yj + y[iy] * xj;
                ix += incx;
                iy += incy;
            }
            
            jx += incx;
            jy += incy;
        }
    } else {
        int jx = 0, jy = 0;
        for (int j = 0; j < n; j++) {
            double xj = alpha * x[jx];
            double yj = alpha * y[jy];
            
            int ix = 0, iy = 0;
            for (int i = 0; i <= j; i++) {
                A[i + j * lda] += x[ix] * yj + y[iy] * xj;
                ix += incx;
                iy += incy;
            }
            
            jx += incx;
            jy += incy;
        }
    }
}
