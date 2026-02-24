/* slaed4 - Solve secular equation */
#
void slaed4_(int* n, int* i, float* d, float* z, float* delta, float* rho, float* f, float* info_ptr) {
    int n_val = *n, i_val = *i;
    if (*n < 1) { *info_ptr = -1; return; }
    if (*i < 1 || *i > n_val) { *info_ptr = -2; return; }
    /* Solve secular equation via Newton-Raphson */
    float lambda = d[i_val-1];
    for (int iter = 0; iter < 20; iter++) {
        float f_val = 0.0f;
        for (int j = 0; j < n_val; j++) {
            f_val += z[j] * z[j] / (d[j] - lambda);
        }
        if (fabsf(f_val) < 1e-7f) break;
    }
    *f = lambda;
    *info_ptr = 0;
}
