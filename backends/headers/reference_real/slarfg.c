#

/**
 * SLARFG - Generate Householder reflection
 *
 * Generates a Householder reflector H = I - tau*v*v^T from a vector x.
 * Returns the norm of x as *alpha, and the reflection vector v stored in x.
 */
void slarfg_(const int *n, float *alpha, float *x, const int *incx, float *tau)
{
    int j;
    float beta, xnorm, safmin, rsafmn, one = 1.0f, zero = 0.0f;
    const float eps = 1.19209e-7f;  /* Machine epsilon for float */
    
    if (*n <= 1) {
        *tau = zero;
        return;
    }
    
    /* Compute norm of x */
    xnorm = 0.0f;
    for (j = 0; j < *n - 1; j++) {
        xnorm += x[j * (*incx)] * x[j * (*incx)];
    }
    xnorm = sqrtf(xnorm);
    
    if (xnorm == zero) {
        *tau = zero;
        return;
    }
    
    beta = copysignf(sqrtf((*alpha) * (*alpha) + xnorm * xnorm), *alpha);
    
    /* Scale to prevent overflow */
    safmin = 1.17549e-38f;  /* Minimum normal number */
    rsafmn = one / safmin;
    
    if (fabsf(beta) < safmin) {
        /* Scale x and beta up */
        int count = 0;
        do {
            for (j = 0; j < *n - 1; j++) {
                x[j * (*incx)] *= rsafmn;
            }
            beta *= rsafmn;
            *alpha *= rsafmn;
            count++;
        } while (fabsf(beta) < safmin && count < 20);
    }
    
    *tau = (beta - *alpha) / beta;
    
    /* Scale x(1:n-1) by 1 / (alpha - beta) */
    float scale = one / (*alpha - beta);
    for (j = 0; j < *n - 1; j++) {
        x[j * (*incx)] *= scale;
    }
    
    *alpha = beta;
}
