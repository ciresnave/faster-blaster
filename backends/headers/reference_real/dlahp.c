/* dlahp - Double precision Householder panel */
#
void dlahp_(int* n, int* k, double* a, int* lda, double* tau, double* y, int* ldy, int* info) {
    int n_val = *n, lda_val = *lda, ldy_val = *ldy;
    if (*n < 0) { *info = -1; return; }
    if (*k < 0 || *k > n_val) { *info = -2; return; }
    if (*lda < n_val) { *info = -4; return; }
    if (*ldy < n_val) { *info = -7; return; }
    *info = 0;
}
