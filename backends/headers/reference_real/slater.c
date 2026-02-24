/* slater - Generate Lehmer test matrix */
#
void slater_(int* n, float* a, int* lda) {
    if (*n < 0 || *lda < *n) return;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            a[i + j*(*lda)] = 1.0f / (float)(i + j + 2);
        }
    }
}
