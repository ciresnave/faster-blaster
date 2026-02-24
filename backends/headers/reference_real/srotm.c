/**
 * @file srotm.c
 * @brief Reference implementation of SROTM (Single precision modified Givens rotation)
 *
 * SROTM: apply modified plane rotation
 *
 * Uses a modified rotation representation (more efficient for sparse operations)
 * The param array encodes the rotation parameters:
 *   param[0] = flag (-2, -1, 0, 1)
 *   param[1] = H11
 *   param[2] = H21
 *   param[3] = H12
 *   param[4] = H22
 */

#include <blas_reference.h>

/**
 * Single precision apply modified Givens rotation
 */
void srotm_ref(int n, float *x, int incx, float *y, int incy, const float *param) {
    if (n <= 0 || param == NULL) return;
    
    int flag = (int)param[0];
    float h11, h21, h12, h22;
    
    /* Extract rotation parameters based on flag */
    if (flag == -1) {
        /* Both H matrices stored */
        h11 = param[1]; h21 = param[2];
        h12 = param[3]; h22 = param[4];
    } else if (flag == 0) {
        /* H = [[H11, H12], [H21, 1]] */
        h11 = param[1]; h12 = param[3];
        h21 = param[2];
        h22 = 1.0f;
    } else if (flag == 1) {
        /* H = [[1, H12], [H21, H22]] */
        h11 = 1.0f; h12 = param[3];
        h21 = param[2]; h22 = param[4];
    } else {
        /* flag == -2: identity (no rotation) */
        return;
    }
    
    /* Apply rotation */
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            float temp = h11 * x[i] + h12 * y[i];
            y[i] = h21 * x[i] + h22 * y[i];
            x[i] = temp;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            float temp = h11 * x[ix] + h12 * y[iy];
            y[iy] = h21 * x[ix] + h22 * y[iy];
            x[ix] = temp;
            ix += incx;
            iy += incy;
        }
    }
}
