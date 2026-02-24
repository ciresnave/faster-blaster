/* slantg - Triangular matrix with gap */
#
void slantg_(int* n, float* d, int* offd, float* x, int* lda, float* g, float* tol, float* work) {
    if (*n < 0 || *lda < *n) return;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            x[i + j*(*lda)] = (i == j) ? d[i] : 0.0f;
        }
    }
}
