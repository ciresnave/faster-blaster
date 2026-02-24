/**
 * @file ddot.c
 * @brief Reference implementation of DDOT (Double precision dot product)
 *
 * DDOT: sum = x^T * y (double precision)
 *
 * Uses Kahan summation for numerical stability.
 */

#include <blas_reference.h>

/**
 * Double precision dot product using Kahan summation
 */
double ddot_ref(int n, const double *x, int incx, const double *y, int incy) {
    if (n <= 0) return 0.0;
    
    double sum = 0.0;
    double correction = 0.0;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            double product = x[i] * y[i];
            double y_corrected = product - correction;
            double t = sum + y_corrected;
            correction = (t - sum) - y_corrected;
            sum = t;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            double product = x[ix] * y[iy];
            double y_corrected = product - correction;
            double t = sum + y_corrected;
            correction = (t - sum) - y_corrected;
            sum = t;
            ix += incx;
            iy += incy;
        }
    }
    
    return sum;
}
