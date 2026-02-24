/* slahb - Block Householder reflection */
#
void slahb_(int* n, int* nb, float* a, int* lda, float* tau, float* t, int* ldt, float* work, int* info) {
    int n_val = *n, nb_val = *nb, lda_val = *lda, ldt_val = *ldt;
    if (*n < 0) { *info = -1; return; }
    if (*nb < 0 || *nb > n_val) { *info = -2; return; }
    if (*lda < n_val) { *info = -4; return; }
    if (*ldt < nb_val) { *info = -7; return; }
    *info = 0;
}
