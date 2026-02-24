/**
 * @file snrm2.c
 * @brief Reference implementation of SNRM2 (Single precision Euclidean norm)
 *
 * SNRM2: result = sqrt(x^T * x) = ||x||_2
 *
 * Uses scaled accumulation to avoid overflow/underflow.
 */

#include <blas_reference.h>
#

/**
 * Single precision Euclidean norm
 *
 * @param n Number of elements
 * @param x Input vector
 * @param incx Stride for X
 * @return Euclidean norm ||x||_2
 */
float snrm2_ref(int n, const float *x, int incx) {
    if (n <= 0) return 0.0f;
    if (incx <= 0) return 0.0f;
    
    float scale = 0.0f;
    float ssq = 0.0f;  /* Sum of squares (scaled) */
    
    int ix = 0;
    for (int i = 0; i < n; i++) {
        float absxi = fabsf(x[ix]);
        
        if (absxi > 0.0f) {
            if (scale < absxi) {
                /* Rescale: ssq = (scale/absxi)^2 * ssq + (absxi/absxi)^2 */
                float ratio = scale / absxi;
                ssq = 1.0f + ssq * ratio * ratio;
                scale = absxi;
            } else {
                float ratio = absxi / scale;
                ssq += ratio * ratio;
            }
        }
        
        ix += incx;
    }
    
    return scale * sqrtf(ssq);
}
