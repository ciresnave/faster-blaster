/**
 * @file srotmg.c
 * @brief Reference implementation of SROTMG (Single precision modified Givens rotation generator)
 *
 * SROTMG: compute modified Givens rotation parameters
 *
 * Given d1, d2, x1, y1, compute the modified Givens rotation
 * and store parameters in param array.
 */

#include <blas_reference.h>
#

/**
 * Single precision modified Givens rotation generator
 */
void srotmg_ref(float *d1, float *d2, float *x1, float y1, float *param) {
    if (d1 == NULL || d2 == NULL || x1 == NULL || param == NULL) return;
    
    const float SAFMIN = 1e-10f;
    const float SAFMAX = 1e10f;
    
    float gam = *d2 * y1;
    
    if (fabsf(*d1) < SAFMIN || fabsf(*d2) < SAFMIN) {
        /* Handle underflow */
        param[0] = -2.0f;  /* Identity */
        return;
    }
    
    float alpha = *d1 * (*x1);
    float beta = gam / (*d2);
    
    float gamma_sq = alpha * alpha / (*d1) + beta * beta / (*d2);
    
    if (gamma_sq > SAFMAX || gamma_sq < SAFMIN) {
        /* Handle overflow/underflow */
        param[0] = -2.0f;
        return;
    }
    
    float gamma = sqrtf(gamma_sq);
    
    float delta = *d1 / gamma;
    float h11 = *d1 / gamma;
    float h12 = -(*x1) / gamma;
    float h21 = y1 / gamma;
    float h22 = *d2 / gamma;
    
    *d1 = gamma;
    *d2 = gamma;
    *x1 = h11 * alpha + h12 * beta;
    
    /* Encode rotation in param array */
    param[0] = -1.0f;  /* Full 2x2 matrix */
    param[1] = h11;
    param[2] = h21;
    param[3] = h12;
    param[4] = h22;
}
