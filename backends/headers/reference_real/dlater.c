/* dlater - Double precision Lehmer test matrix */
#
void dlater_(int* n, double* a, int* lda) {
    if (*n < 0 || *lda < *n) return;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            a[i + j*(*lda)] = 1.0 / (double)(i + j + 2);
        }
    }
}
