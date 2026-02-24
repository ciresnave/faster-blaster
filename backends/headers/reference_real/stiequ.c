/* Equilibrate tridiagonal matrix */
#
void stiequ_(int* n, float* d, float* e, float* s, float* scond, float* amax, int* info) {
    int n_val = *n;
    if (*n < 0) { *info = -1; return; }
    if (*n == 0) { *info = 0; return; }
    /* Initialize scaling factors to 1.0 */
    *amax = 0.0f;
    for (int i = 0; i < n_val; i++) {
        s[i] = 1.0f;
        if (d[i] > *amax) *amax = d[i];
        else if (-d[i] > *amax) *amax = -d[i];
    }
    for (int i = 0; i < n_val-1; i++) {
        if (e[i] > *amax) *amax = e[i];
        else if (-e[i] > *amax) *amax = -e[i];
    }
    *scond = 1.0f;
    *info = 0;
}
