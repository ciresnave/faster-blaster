/**
 * @file drotmg.c
 * @brief Reference implementation of DROTMG (Double precision modified Givens rotation generator)
 *
 * DROTMG: compute modified Givens rotation parameters (double precision)
 */

#include <blas_reference.h>
#

/**
 * Double precision modified Givens rotation generator
 */
void drotmg_ref(double *d1, double *d2, double *x1, double y1, double *param) {
    if (d1 == NULL || d2 == NULL || x1 == NULL || param == NULL) return;
    
    const double SAFMIN = 1e-15;
    const double SAFMAX = 1e15;
    
    double gam = *d2 * y1;
    
    if (fabs(*d1) < SAFMIN || fabs(*d2) < SAFMIN) {
        param[0] = -2.0;  /* Identity */
        return;
    }
    
    double alpha = *d1 * (*x1);
    double beta = gam / (*d2);
    
    double gamma_sq = alpha * alpha / (*d1) + beta * beta / (*d2);
    
    if (gamma_sq > SAFMAX || gamma_sq < SAFMIN) {
        param[0] = -2.0;
        return;
    }
    
    double gamma = sqrt(gamma_sq);
    
    double h11 = *d1 / gamma;
    double h12 = -(*x1) / gamma;
    double h21 = y1 / gamma;
    double h22 = *d2 / gamma;
    
    *d1 = gamma;
    *d2 = gamma;
    *x1 = h11 * alpha + h12 * beta;
    
    param[0] = -1.0;  /* Full 2x2 matrix */
    param[1] = h11;
    param[2] = h21;
    param[3] = h12;
    param[4] = h22;
}
