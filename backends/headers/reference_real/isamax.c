/**
 * @file isamax.c
 * @brief Reference implementation of ISAMAX (Single precision index of maximum absolute value)
 *
 * ISAMAX: result = index i where |x_i| is maximum
 *
 * Returns 1-based index for BLAS compatibility (not 0-based).
 */

#include <blas_reference.h>
#

/**
 * Single precision index of maximum absolute value (1-based indexing)
 */
int isamax_ref(int n, const float *x, int incx) {
    if (n <= 0) return 0;
    if (incx <= 0) return 0;
    
    int max_idx = 1;  /* 1-based index */
    float max_val = fabsf(x[0]);
    
    int ix = incx;
    for (int i = 2; i <= n; i++) {  /* Start from 2 (1-based) */
        float absval = fabsf(x[ix]);
        if (absval > max_val) {
            max_val = absval;
            max_idx = i;
        }
        ix += incx;
    }
    
    return max_idx;
}
