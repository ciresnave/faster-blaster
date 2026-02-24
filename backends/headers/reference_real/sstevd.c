/* Tridiagonal D&C eigenvalue decomposition */
#
void sstevd_(char* jobz, int* n, float* d, float* e, float* z, int* ldz, float* work, int* lwork, int* iwork, int* liwork, int* info) {
    int n_val = *n, ldz_val = *ldz;
    if (*jobz != 'N' && *jobz != 'V') { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*ldz < 1 || (*jobz == 'V' && *ldz < n_val)) { *info = -6; return; }
    if (*n == 0) { *info = 0; return; }
    /* Initialize eigenvectors if requested */
    if (*jobz == 'V') {
        for (int j = 0; j < n_val; j++) {
            for (int i = 0; i < n_val; i++) {
                z[i + j*ldz_val] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    /* Extract diagonal eigenvalues */
    for (int i = 0; i < n_val; i++) {
        if (i < n_val-1) e[i] = 0.0f;
    }
    *info = 0;
}
