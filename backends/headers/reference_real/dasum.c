/**
 * @file dasum.c
 * @brief Reference implementation of DASUM (Double precision absolute value sum)
 *
 * DASUM: result = sum(|x_i|)
 */

#include <blas_reference.h>
#

/**
 * Double precision sum of absolute values
 */
double dasum_ref(int n, const double *x, int incx) {
    if (n <= 0) return 0.0;
    if (incx <= 0) return 0.0;
    
    double sum = 0.0;
    int ix = 0;
    
    for (int i = 0; i < n; i++) {
        sum += fabs(x[ix]);
        ix += incx;
    }
    
    return sum;
}
