/* slatem - Hermitian/Symmetric test matrix */
#
void slatem_(char* uplo, int* n, float* a, int* lda, int* iseed, float* work) {
    if (*n < 0 || *lda < *n) return;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            a[i + j*(*lda)] = (i == j) ? 1.0f : 0.0f;
        }
    }
}
