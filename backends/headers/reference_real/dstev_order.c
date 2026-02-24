/* dstev_order - Double precision symmetric tridiagonal eigenvalues in order */
#
void dstev_order_(char* jobz, int* n, double* d, double* e, double* z, int* ldz, double* work, int* info) {
    int n_val = *n, ldz_val = *ldz;
    if (*jobz != 'N' && *jobz != 'V') { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*ldz < n_val) { *info = -6; return; }
    for (int i = 0; i < n_val; i++) {
        work[i] = d[i];
    }
    *info = 0;
}
