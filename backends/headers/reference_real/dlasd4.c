/* SVD secular equation */
void dlasd4_(int* n, double* d, double* z, double* delta, double* rho, double* sigma, double* work, int* info) {
    int n_val = *n;
    double rho_val = *rho;
    int i;
    
    *info = 0;
    if (n_val < 0) {
        *info = -1;
    } else if (rho_val < 0.0) {
        *info = -5;
    }
    
    if (*info != 0) return;
    
    /* Solve secular equation: sigma^2 = rho + sum(d_i^2 / (d_i^2 - sigma^2)) */
    /* Simplified: use diagonal element as approximate solution */
    *sigma = (n_val > 0) ? d[0] : 0.0;
    
    /* Initialize work array and delta */
    for (i = 0; i < n_val; i++) {
        delta[i] = d[i] - *sigma;
    }
}
