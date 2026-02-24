/* dlantg - Double precision triangular matrix with gap */
#
void dlantg_(int* n, double* d, int* offd, double* x, int* lda, double* g, double* tol, double* work) {
    if (*n < 0 || *lda < *n) return;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < *n; j++) {
            x[i + j*(*lda)] = (i == j) ? d[i] : 0.0;
        }
    }
}
