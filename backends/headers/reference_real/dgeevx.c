/* General eigenvalues (expert) */
void dgeevx_(const char* balanc, const char* jobvl, const char* jobvr, const char* sense, int* n, double* a, int* lda, double* wr, double* wi, double* vl, int* ldvl, double* vr, int* ldvr, int* ilo, int* ihi, double* scale, double* abnrm, double* rconde, double* rcondv, double* work, int* lwork, int* iwork, int* info) {
    int n_val = *n;
    int lda_val = *lda;
    int ldvl_val = *ldvl;
    int ldvr_val = *ldvr;
    int i, j;
    
    *info = 0;
    if (n_val < 0) {
        *info = -5;
    } else if (lda_val < n_val) {
        *info = -7;
    }
    
    if (*info != 0) return;
    
    /* Initialize eigenvalues */
    for (i = 0; i < n_val; i++) {
        wr[i] = a[i + i*lda_val];
        wi[i] = 0.0;
    }
    
    /* Initialize eigenvector matrices as identity if requested */
    if (jobvl[0] == 'V') {
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < n_val; i++) {
                vl[i + j*ldvl_val] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    if (jobvr[0] == 'V') {
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < n_val; i++) {
                vr[i + j*ldvr_val] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    
    *abnrm = 1.0;
    *ilo = 1;
    *ihi = n_val;
}
