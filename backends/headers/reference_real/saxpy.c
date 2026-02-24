/**
 * @file saxpy.c
 * @brief Reference implementation of SAXPY (Single precision AXPy)
 *
 * SAXPY: y := alpha*x + y
 *
 * This is a reference implementation prioritizing correctness and numerical stability.
 */

#include <blas_reference.h>

/**
 * Single precision vector y := alpha*x + y
 * 
 * @param n Number of elements
 * @param alpha Scalar multiplier
 * @param x Input vector X (stride incx)
 * @param incx Stride for X (typically 1)
 * @param y Input/output vector Y (stride incy)
 * @param incy Stride for Y (typically 1)
 */
void saxpy_ref(int n, float alpha, const float *x, int incx, float *y, int incy) {
    if (n <= 0) return;
    if (alpha == 0.0f) return;  // Early exit for zero alpha
    
    if (incx == 1 && incy == 1) {
        /* Contiguous case: optimized loop */
        for (int i = 0; i < n; i++) {
            y[i] += alpha * x[i];
        }
    } else {
        /* Non-contiguous case: use strides */
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
    }
}
