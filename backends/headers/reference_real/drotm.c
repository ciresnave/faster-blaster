/**
 * @file drotm.c
 * @brief Reference implementation of DROTM (Double precision modified Givens rotation)
 *
 * DROTM: apply modified plane rotation (double precision)
 */

#include <blas_reference.h>

/**
 * Double precision apply modified Givens rotation
 */
void drotm_ref(int n, double *x, int incx, double *y, int incy, const double *param) {
    if (n <= 0 || param == NULL) return;
    
    int flag = (int)param[0];
    double h11, h21, h12, h22;
    
    if (flag == -1) {
        h11 = param[1]; h21 = param[2];
        h12 = param[3]; h22 = param[4];
    } else if (flag == 0) {
        h11 = param[1]; h12 = param[3];
        h21 = param[2];
        h22 = 1.0;
    } else if (flag == 1) {
        h11 = 1.0; h12 = param[3];
        h21 = param[2]; h22 = param[4];
    } else {
        return;
    }
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            double temp = h11 * x[i] + h12 * y[i];
            y[i] = h21 * x[i] + h22 * y[i];
            x[i] = temp;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            double temp = h11 * x[ix] + h12 * y[iy];
            y[iy] = h21 * x[ix] + h22 * y[iy];
            x[ix] = temp;
            ix += incx;
            iy += incy;
        }
    }
}
