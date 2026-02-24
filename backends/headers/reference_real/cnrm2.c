/**
 * @file cnrm2.c
 * @brief Reference implementation of CNRM2 (Complex single precision Euclidean norm)
 *
 * CNRM2: result = sqrt(real(x^H * x)) = ||x||_2 for complex
 *
 * Uses scaled accumulation.
 */

#include <blas_reference.h>
#

typedef struct {
    float real;
    float imag;
} complex_f;

/**
 * Complex single precision Euclidean norm
 */
float cnrm2_ref(int n, const complex_f *x, int incx) {
    if (n <= 0) return 0.0f;
    if (incx <= 0) return 0.0f;
    
    float scale = 0.0f;
    float ssq = 0.0f;
    
    int ix = 0;
    for (int i = 0; i < n; i++) {
        float absreal = fabsf(x[ix].real);
        float absimag = fabsf(x[ix].imag);
        
        if (absreal > 0.0f || absimag > 0.0f) {
            float absval = sqrtf(absreal * absreal + absimag * absimag);
            
            if (scale < absval) {
                float ratio = scale / absval;
                ssq = 1.0f + ssq * ratio * ratio;
                scale = absval;
            } else {
                float ratio = absval / scale;
                ssq += ratio * ratio;
            }
        }
        
        ix += incx;
    }
    
    return scale * sqrtf(ssq);
}
