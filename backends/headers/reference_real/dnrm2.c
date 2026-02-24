/**
 * @file dnrm2.c
 * @brief Reference implementation of DNRM2 (Double precision Euclidean norm)
 *
 * DNRM2: result = sqrt(x^T * x) = ||x||_2
 *
 * Uses scaled accumulation to avoid overflow/underflow.
 */

#include <blas_reference.h>
#

/**
 * Double precision Euclidean norm
 */
double dnrm2_ref(int n, const double *x, int incx) {
    if (n <= 0) return 0.0;
    if (incx <= 0) return 0.0;
    
    double scale = 0.0;
    double ssq = 0.0;
    
    int ix = 0;
    for (int i = 0; i < n; i++) {
        double absxi = fabs(x[ix]);
        
        if (absxi > 0.0) {
            if (scale < absxi) {
                double ratio = scale / absxi;
                ssq = 1.0 + ssq * ratio * ratio;
                scale = absxi;
            } else {
                double ratio = absxi / scale;
                ssq += ratio * ratio;
            }
        }
        
        ix += incx;
    }
    
    return scale * sqrt(ssq);
}
