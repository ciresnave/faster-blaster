/**
 * @file sasum.c
 * @brief Reference implementation of SASUM (Single precision absolute value sum)
 *
 * SASUM: result = sum(|x_i|)
 */

#include <blas_reference.h>
#

/**
 * Single precision sum of absolute values
 */
float sasum_ref(int n, const float *x, int incx) {
    if (n <= 0) return 0.0f;
    if (incx <= 0) return 0.0f;
    
    float sum = 0.0f;
    int ix = 0;
    
    for (int i = 0; i < n; i++) {
        sum += fabsf(x[ix]);
        ix += incx;
    }
    
    return sum;
}
