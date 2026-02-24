/* slart - Givens row/column multiplication */
#
void slart_(int* m, int* n, float* c, float* s, float* a, int* lda) {
    if (*m <= 0 || *n <= 0) return;
    for (int i = 0; i < *m; i++) {
        float tmp = a[i];
        a[i] = *c * tmp + *s * a[i + *n*(*lda)];
        a[i + *n*(*lda)] = -(*s) * tmp + *c * a[i + *n*(*lda)];
    }
}
