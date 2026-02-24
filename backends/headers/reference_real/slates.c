/* slates - Triangular test matrix */
#
void slates_(char* uplo, int* n, float* a, int* lda) {
    if (*n < 0 || *lda < *n) return;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            a[i + j*(*lda)] = 0.0f;
        }
    }
}
