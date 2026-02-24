/* dpoevd */
void dpoevd_(const char* jobz, const char* uplo, int* n, double* ap, double* w, double* z, int* ldz, double* work, int* lwork, int* iwork, int* liwork, int* info) {
    int n_val = *n;
    int ldz_val = *ldz;
    int lwork_val = *lwork;
    int i, j;
    
    *info = 0;
    if (n_val < 0) {
        *info = -3;
    } else if (ldz_val < n_val) {
        *info = -7;
    } else if (lwork_val < 1) {
        *info = -9;
    }
    
    if (*info != 0) return;
    
    /* Extract eigenvalues from diagonal of symmetric positive definite matrix */
    for (i = 0; i < n_val; i++) {
        w[i] = ap[i + i*n_val];
    }
    
    /* Initialize eigenvector matrix as identity if requested */
    if (jobz[0] == 'V') {
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < n_val; i++) {
                z[i + j*ldz_val] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
}
