/* slahrd */
void slahrd_(int* n, int* k, int* nb, float* a, int* lda, float* tau, float* t, int* ldt, float* y, int* ldy, int* info) {
    int n_val = *n;
    int k_val = *k;
    int nb_val = *nb;
    int lda_val = *lda;
    int ldt_val = *ldt;
    int ldy_val = *ldy;
    int i, j;
    
    *info = 0;
    if (n_val < 0) {
        *info = -1;
    } else if (k_val < 1) {
        *info = -2;
    } else if (lda_val < n_val) {
        *info = -5;
    }
    
    if (*info != 0) return;
    
    /* Initialize T matrix for Householder reflections */
    for (j = 0; j < nb_val; j++) {
        for (i = 0; i < nb_val; i++) {
            t[i + j*ldt_val] = (i <= j) ? tau[j] : 0.0f;
        }
    }
}
