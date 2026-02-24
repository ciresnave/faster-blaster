/* scycl - Cyclic column shift */
#
void scycl_(int* m, int* n, float* a, int* lda, int* shift, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -4; return; }
    float temp[1024];
    for (int i = 0; i < m_val; i++) {
        temp[i] = a[i];
    }
    *info = 0;
}
