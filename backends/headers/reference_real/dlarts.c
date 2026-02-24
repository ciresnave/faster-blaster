/* dlarts - Double precision Givens on tridiagonal */
#
void dlarts_(int* m, int* n, int* ks, double* d, double* e, double* a, int* lda, double* c, double* s) {
    if (*m <= 0 || *n <= 0) return;
    for (int j = 0; j < *n - 1; j++) {
        double tmp = d[j];
        d[j] = *c * tmp + *s * e[j];
        e[j] = -(*s) * tmp + *c * e[j];
    }
}
