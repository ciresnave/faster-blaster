/**
 * @file sdot.c
 * @brief Reference implementation of SDOT (Single precision dot product)
 *
 * SDOT: sum = x^T * y (single precision)
 *
 * Uses Kahan summation for numerical stability.
 */

#include <blas_reference.h>

/**
 * Single precision dot product using Kahan summation
 *
 * @param n Number of elements
 * @param x Input vector X
 * @param incx Stride for X
 * @param y Input vector Y
 * @param incy Stride for Y
 * @return Dot product result
 */
float sdot_ref(int n, const float *x, int incx, const float *y, int incy) {
    if (n <= 0) return 0.0f;
    
    float sum = 0.0f;
    float correction = 0.0f;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            float product = x[i] * y[i];
            float y_corrected = product - correction;
            float t = sum + y_corrected;
            correction = (t - sum) - y_corrected;
            sum = t;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            float product = x[ix] * y[iy];
            float y_corrected = product - correction;
            float t = sum + y_corrected;
            correction = (t - sum) - y_corrected;
            sum = t;
            ix += incx;
            iy += incy;
        }
    }
    
    return sum;
}
