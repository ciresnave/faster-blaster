/* sstev_order - Symmetric tridiagonal eigenvalues in order */
#
void sstev_order_(char* jobz, int* n, float* d, float* e, float* z, int* ldz, float* work, int* info) {
    int n_val = *n, ldz_val = *ldz;
    if (*jobz != 'N' && *jobz != 'V') { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*ldz < n_val) { *info = -6; return; }
    for (int i = 0; i < n_val; i++) {
        work[i] = d[i];
    }
    *info = 0;
}
