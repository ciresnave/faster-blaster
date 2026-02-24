/**
 * @file idamax.c
 * @brief Reference implementation of IDAMAX (Double precision index of maximum absolute value)
 *
 * IDAMAX: result = index i where |x_i| is maximum
 *
 * Returns 1-based index for BLAS compatibility.
 */

#include <blas_reference.h>
#

/**
 * Double precision index of maximum absolute value (1-based indexing)
 */
int idamax_ref(int n, const double *x, int incx) {
    if (n <= 0) return 0;
    if (incx <= 0) return 0;
    
    int max_idx = 1;
    double max_val = fabs(x[0]);
    
    int ix = incx;
    for (int i = 2; i <= n; i++) {
        double absval = fabs(x[ix]);
        if (absval > max_val) {
            max_val = absval;
            max_idx = i;
        }
        ix += incx;
    }
    
    return max_idx;
}
