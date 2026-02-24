/* dlatem - Double precision Hermitian test matrix */
#
void dlatem_(char* uplo, int* n, double* a, int* lda, int* iseed, double* work) {
    if (*n < 0 || *lda < *n) return;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            a[i + j*(*lda)] = (i == j) ? 1.0 : 0.0;
        }
    }
}
