/**
 * @file ssyr2.c
 * @brief Reference implementation of SSYR2 (Symmetric rank-2 update)
 *
 * SSYR2: A := alpha*x*y^T + alpha*y*x^T + A where A is symmetric
 */

#include <blas_reference.h>

void ssyr2_ref(char uplo, int n, float alpha, const float *x, int incx,
               const float *y, int incy, float *A, int lda) {
    
    if (n <= 0 || alpha == 0.0f) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    
    if (is_lower) {
        /* Lower triangle: update A[i,j] for i >= j */
        int jx = 0, jy = 0;
        for (int j = 0; j < n; j++) {
            float xj = alpha * x[jx];
            float yj = alpha * y[jy];
            
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
        /* Upper triangle: update A[i,j] for i <= j */
        int jx = 0, jy = 0;
        for (int j = 0; j < n; j++) {
            float xj = alpha * x[jx];
            float yj = alpha * y[jy];
            
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
