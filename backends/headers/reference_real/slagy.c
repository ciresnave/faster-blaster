/* slagy */
void slagy_(const char* uplo, int* n, float* a, int* lda, int* kd, float* af, int* ldaf, int* ipiv, float* w, float* b, int* ldb, int* info) {
    int n_val = *n;
    int kd_val = *kd;
    int lda_val = *lda;
    int ldaf_val = *ldaf;
    int i, j;
    
    *info = 0;
    if (n_val < 0) {
        *info = -2;
    } else if (lda_val < n_val) {
        *info = -4;
    } else if (ldaf_val < n_val) {
        *info = -7;
    }
    
    if (*info != 0) return;
    
    /* Initialize pivot array and copy matrix */
    for (j = 0; j < n_val; j++) {
        for (i = 0; i < n_val; i++) {
            af[i + j*ldaf_val] = a[i + j*lda_val];
        }
        ipiv[j] = j + 1;
    }
}
