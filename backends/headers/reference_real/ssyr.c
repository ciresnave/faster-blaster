/**
 * @file ssyr.c
 * @brief Reference implementation of SSYR (Single precision symmetric rank-1 update)
 *
 * SSYR: A := alpha*x*x^T + A where A is symmetric
 *
 * Only the upper or lower triangle is used/updated
 */

#include <blas_reference.h>

/**
 * Single precision symmetric rank-1 update
 *
 * @param uplo 'U'/'L' - upper or lower triangle
 * @param n Dimension of A
 * @param alpha Scalar
 * @param x Vector x (n elements)
 * @param incx Stride for x
 * @param A Symmetric matrix (n x n, column-major)
 * @param lda Leading dimension
 */
void ssyr_ref(char uplo, int n, float alpha, const float *x, int incx,
              float *A, int lda) {
    
    if (n <= 0) return;
    if (alpha == 0.0f) return;
    
    int is_lower = (uplo == 'L' || uplo == 'l');
    
    if (is_lower) {
        /* Lower triangle: update A[i,j] for i >= j */
        int jx = 0;
        for (int j = 0; j < n; j++) {
            float xj = alpha * x[jx];
            
            int ix = jx;
            for (int i = j; i < n; i++) {
                A[i + j * lda] += x[ix] * xj;
                ix += incx;
            }
            
            jx += incx;
        }
    } else {
        /* Upper triangle: update A[i,j] for i <= j */
        int jx = 0;
        for (int j = 0; j < n; j++) {
            float xj = alpha * x[jx];
            
            int ix = 0;
            for (int i = 0; i <= j; i++) {
                A[i + j * lda] += x[ix] * xj;
                ix += incx;
            }
            
            jx += incx;
        }
    }
}
