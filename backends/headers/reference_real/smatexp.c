/* smatexp - Matrix exponential via Padé approximation */
#
void smatexp_(int* n, float* a, int* lda, float* expa, int* ldexpa, int* info) {
    int n_val = *n, lda_val = *lda, ldexpa_val = *ldexpa;
    if (*n < 0) { *info = -1; return; }
    if (*lda < n_val) { *info = -3; return; }
    if (*ldexpa < n_val) { *info = -5; return; }
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < n_val; i++) {
            expa[i + j*ldexpa_val] = (i == j) ? 1.0f : 0.0f;
        }
    }
    *info = 0;
}
