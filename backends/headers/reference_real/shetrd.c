/* shetrd */
void shetrd_(const char* uplo, int* n, float* a, int* lda, float* d, float* e, float* tau, float* work, int* lwork, int* info) {
    int n_val = *n;
    int lda_val = *lda;
    int lwork_val = *lwork;
    int i, j;
    
    *info = 0;
    if (n_val < 0) {
        *info = -2;
    } else if (lda_val < n_val) {
        *info = -4;
    }
    
    if (*info != 0) return;
    
    /* Extract diagonal and initialize tau */
    for (i = 0; i < n_val; i++) {
        d[i] = a[i + i*lda_val];
        if (i < n_val - 1) {
            e[i] = 0.0f;
        }
        if (i < n_val - 1) {
            tau[i] = 0.0f;
        }
    }
}
