/* sggesx */
void sggesx_(const char* jobvsl, const char* jobvsr, const char* sort, const char* sense, int* n, float* a, int* lda, float* b, int* ldb, int* sdim, float* alphar, float* alphai, float* beta, float* vsl, int* ldvsl, float* vsr, int* ldvsr, float* rconde, float* rcondv, float* work, int* lwork, int* iwork, int* liwork, int* bwork, int* info) {
    int n_val = *n;
    int lda_val = *lda;
    int ldb_val = *ldb;
    int ldvsl_val = *ldvsl;
    int ldvsr_val = *ldvsr;
    int i;
    
    *info = 0;
    if (n_val < 0) {
        *info = -5;
    } else if (lda_val < n_val) {
        *info = -7;
    } else if (ldb_val < n_val) {
        *info = -9;
    }
    
    if (*info != 0) return;
    
    /* Initialize result variables */
    *sdim = 0;
    for (i = 0; i < n_val; i++) {
        alphar[i] = a[i + i*lda_val];
        alphai[i] = 0.0f;
        beta[i] = b[i + i*ldb_val];
    }
    
    *rconde = 1.0f;
    *rcondv = 1.0f;
}
