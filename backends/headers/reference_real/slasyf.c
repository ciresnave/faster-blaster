/* slasyf - Bunch-Kaufman pivoting block */
#
void slasyf_(char* uplo, int* n, int* nb, int* kb, float* a, int* lda, int* ipiv, float* w, int* ldw, int* info) {
    if (*n < 0 || *nb < 0) { *info = -2; return; }
    *info = 0;
}
