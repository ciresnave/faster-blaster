/* slarts - Givens on tridiagonal matrix */
#
void slarts_(int* m, int* n, int* ks, float* d, float* e, float* a, int* lda, float* c, float* s) {
    if (*m <= 0 || *n <= 0) return;
    for (int j = 0; j < *n - 1; j++) {
        float tmp = d[j];
        d[j] = *c * tmp + *s * e[j];
        e[j] = -(*s) * tmp + *c * e[j];
    }
}
