/* Equilibrate tridiagonal matrix */
#
void dtiequ_(int* n, double* d, double* e, double* s, double* scond, double* amax, int* info) {
    int n_val = *n;
    if (*n < 0) { *info = -1; return; }
    if (*n == 0) { *info = 0; return; }
    /* Initialize scaling factors to 1.0 */
    *amax = 0.0;
    for (int i = 0; i < n_val; i++) {
        s[i] = 1.0;
        if (d[i] > *amax) *amax = d[i];
        else if (-d[i] > *amax) *amax = -d[i];
    }
    for (int i = 0; i < n_val-1; i++) {
        if (e[i] > *amax) *amax = e[i];
        else if (-e[i] > *amax) *amax = -e[i];
    }
    *scond = 1.0;
    *info = 0;
}
