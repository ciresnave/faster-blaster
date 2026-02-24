/**
 * @file znrm2.c
 * @brief Reference implementation of ZNRM2 (Complex double precision Euclidean norm)
 *
 * ZNRM2: result = sqrt(real(x^H * x)) = ||x||_2 for complex double
 *
 * Uses scaled accumulation.
 */

#include <blas_reference.h>
#

typedef struct {
    double real;
    double imag;
} complex_d;

/**
 * Complex double precision Euclidean norm
 */
double znrm2_ref(int n, const complex_d *x, int incx) {
    if (n <= 0) return 0.0;
    if (incx <= 0) return 0.0;
    
    double scale = 0.0;
    double ssq = 0.0;
    
    int ix = 0;
    for (int i = 0; i < n; i++) {
        double absreal = fabs(x[ix].real);
        double absimag = fabs(x[ix].imag);
        
        if (absreal > 0.0 || absimag > 0.0) {
            double absval = sqrt(absreal * absreal + absimag * absimag);
            
            if (scale < absval) {
                double ratio = scale / absval;
                ssq = 1.0 + ssq * ratio * ratio;
                scale = absval;
            } else {
                double ratio = absval / scale;
                ssq += ratio * ratio;
            }
        }
        
        ix += incx;
    }
    
    return scale * sqrt(ssq);
}
