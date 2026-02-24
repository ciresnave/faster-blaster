/* dlart - Double precision Givens multiplication */
#
void dlart_(int* m, int* n, double* c, double* s, double* a, int* lda) {
    if (*m <= 0 || *n <= 0) return;
    for (int i = 0; i < *m; i++) {
        double tmp = a[i];
        a[i] = *c * tmp + *s * a[i + *n*(*lda)];
        a[i + *n*(*lda)] = -(*s) * tmp + *c * a[i + *n*(*lda)];
    }
}
