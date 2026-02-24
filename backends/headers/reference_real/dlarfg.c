#

/**
 * DLARFG - Generate Householder reflection (double precision)
 *
 * Generates a Householder reflector H = I - tau*v*v^T from a vector x.
 * Returns the norm of x as *alpha, and the reflection vector v stored in x.
 */
void dlarfg_(const int *n, double *alpha, double *x, const int *incx, double *tau)
{
    int j;
    double beta, xnorm, safmin, rsafmn, one = 1.0, zero = 0.0;
    const double eps = 2.22045e-16;  /* Machine epsilon for double */
    
    if (*n <= 1) {
        *tau = zero;
        return;
    }
    
    /* Compute norm of x */
    xnorm = 0.0;
    for (j = 0; j < *n - 1; j++) {
        xnorm += x[j * (*incx)] * x[j * (*incx)];
    }
    xnorm = sqrt(xnorm);
    
    if (xnorm == zero) {
        *tau = zero;
        return;
    }
    
    beta = copysign(sqrt((*alpha) * (*alpha) + xnorm * xnorm), *alpha);
    
    /* Scale to prevent overflow */
    safmin = 2.22507e-308;  /* Minimum normal number */
    rsafmn = one / safmin;
    
    if (fabs(beta) < safmin) {
        /* Scale x and beta up */
        int count = 0;
        do {
            for (j = 0; j < *n - 1; j++) {
                x[j * (*incx)] *= rsafmn;
            }
            beta *= rsafmn;
            *alpha *= rsafmn;
            count++;
        } while (fabs(beta) < safmin && count < 20);
    }
    
    *tau = (beta - *alpha) / beta;
    
    /* Scale x(1:n-1) by 1 / (alpha - beta) */
    double scale = one / (*alpha - beta);
    for (j = 0; j < *n - 1; j++) {
        x[j * (*incx)] *= scale;
    }
    
    *alpha = beta;
}
