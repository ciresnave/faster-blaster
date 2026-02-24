/* slaqps */
void slaqps_(int* m, int* n, int* offset, int* nb, int* kb, float* a, int* lda, int* jpvt, float* tau, float* vn1, float* vn2, float* auxv, float* f, int* ldf, int* info) {
    int m_val = *m;
    int n_val = *n;
    int nb_val = *nb;
    int lda_val = *lda;
    int ldf_val = *ldf;
    int j;
    
    *info = 0;
    if (m_val < 0) {
        *info = -1;
    } else if (n_val < 0) {
        *info = -2;
    } else if (lda_val < m_val) {
        *info = -6;
    }
    
    if (*info != 0) return;
    
    /* Initialize return variables */
    *kb = 0;
    
    /* Initialize column norms and tau */
    for (j = 0; j < n_val; j++) {
        vn1[j] = 1.0f;
        vn2[j] = 1.0f;
        jpvt[j] = j + 1;
        if (j < nb_val) {
            tau[j] = 0.0f;
        }
    }
}
