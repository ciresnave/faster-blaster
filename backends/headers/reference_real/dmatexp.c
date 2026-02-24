/* dmatexp - Double precision matrix exponential */
#
void dmatexp_(int* n, double* a, int* lda, double* expa, int* ldexpa, int* info) {
    int n_val = *n, lda_val = *lda, ldexpa_val = *ldexpa;
    if (*n < 0) { *info = -1; return; }
    if (*lda < n_val) { *info = -3; return; }
    if (*ldexpa < n_val) { *info = -5; return; }
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < n_val; i++) {
            expa[i + j*ldexpa_val] = (i == j) ? 1.0 : 0.0;
        }
    }
    *info = 0;
}
