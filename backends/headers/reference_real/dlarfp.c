/* Generate Householder F */
void dlarfp_(int* n, double* alpha, double* x, int* incx, double* tau) {
    int n_val = *n;
    int incx_val = *incx;
    int i;
    double sigma, beta;
    
    if (n_val <= 0) {
        *tau = 0.0;
        return;
    }
    
    /* Compute norm of x */
    sigma = 0.0;
    for (i = 0; i < n_val - 1; i++) {
        sigma += x[i*incx_val] * x[i*incx_val];
    }
    sigma = sqrt(sigma);
    
    /* Compute Householder reflection */
    if (sigma == 0.0) {
        *tau = 0.0;
    } else {
        beta = *alpha + sigma;
        *tau = beta / sigma;
        *alpha = -sigma;
    }
}
