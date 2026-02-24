/* dlatet - Double precision tridiagonal test */
#
void dlatet_(char* uplo, int* n, double* a, int* lda) {
    if (*n < 0 || *lda < *n) return;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            a[i + j*(*lda)] = 0.0;
        }
    }
}
