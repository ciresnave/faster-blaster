/* Bidiagonal SVD */
void slasdq_(const char* uplo, int* sqre, int* n, int* ncvt, int* nru, int* ncc, float* d, float* e, float* vt, int* ldvt, float* u, int* ldu, float* c, int* ldc, float* work, int* info) {
    int n_val = *n;
    int sqre_val = *sqre;
    int ncvt_val = *ncvt;
    int nru_val = *nru;
    int ncc_val = *ncc;
    int ldu_val = *ldu;
    int ldvt_val = *ldvt;
    int ldc_val = *ldc;
    int i;
    
    *info = 0;
    if (n_val < 0) {
        *info = -3;
    } else if (sqre_val < 0 || sqre_val > 1) {
        *info = -4;
    } else if (ncvt_val < 0) {
        *info = -5;
    } else if (nru_val < 0) {
        *info = -6;
    } else if (ncc_val < 0) {
        *info = -7;
    } else if (ldu_val < nru_val) {
        *info = -10;
    } else if (ldvt_val < n_val) {
        *info = -8;
    }
    
    if (*info != 0) return;
    
    /* Initialize singular values and vectors */
    for (i = 0; i < n_val; i++) {
        if (d[i] < 0.0f) d[i] = -d[i];  /* Ensure non-negative */
    }
}
