/* Generate Householder F */

#

/* SLARFP - Generate Householder reflector
   N: Vector length
   ALPHA: Alpha element (input/output)
   X: Vector
   INCX: Increment
   TAU: Scalar factor (output)
*/
void slarfp_(const int *n, float *alpha, float *x, const int *incx, float *tau)
{
    int n_val = *n;
    int incx_val = *incx;

    if (n_val <= 1) {
        *tau = 0.0f;
        return;
    }

    /* Compute norm of x */
    float xnorm = 0.0f;
    for (int i = 0; i < n_val - 1; i++) {
        float xi = x[i * incx_val];
        xnorm += xi * xi;
    }
    xnorm = sqrtf(xnorm);

    float sigma = (*alpha >= 0.0f) ? 1.0f : -1.0f;
    float beta = -sigma * sqrtf((*alpha) * (*alpha) + xnorm * xnorm);

    if (xnorm != 0.0f) {
        *tau = (beta - *alpha) / beta;
    } else {
        *tau = 2.0f;
    }

    *alpha = beta;

    /* Scale x */
    if (xnorm != 0.0f) {
        for (int i = 0; i < n_val - 1; i++) {
            x[i * incx_val] = x[i * incx_val] / ((*alpha) - beta);
        }
    }
}
