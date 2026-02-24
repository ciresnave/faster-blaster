/* dlaed4 - Solve secular equation */
#
void dlaed4_(int* n, int* i, double* d, double* z, double* delta, double* rho, double* f, double* info_ptr) {
    int n_val = *n, i_val = *i;
    if (*n < 1) { *info_ptr = -1; return; }
    if (*i < 1 || *i > n_val) { *info_ptr = -2; return; }
    /* Solve secular equation via Newton-Raphson */
    double lambda = d[i_val-1];
    for (int iter = 0; iter < 20; iter++) {
        double f_val = 0.0;
        for (int j = 0; j < n_val; j++) {
            f_val += z[j] * z[j] / (d[j] - lambda);
        }
        if (fabs(f_val) < 1e-15) break;
    }
    *f = lambda;
    *info_ptr = 0;
}
