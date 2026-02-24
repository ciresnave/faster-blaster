/* dlasyf - Double precision Bunch-Kaufman factorization */
#
void dlasyf_(char* uplo, int* n, int* nb, int* kb, double* a, int* lda, int* ipiv, double* w, int* ldw, int* info) {
    if (*n < 0 || *nb < 0) { *info = -2; return; }
    *info = 0;
}
