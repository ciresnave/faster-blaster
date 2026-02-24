/* slarow - Givens rotation on matrix row */
#
void slarow_(int* m, float* a, int* lda, int* j1, float* f, float* c, float* s) {
    if (*m <= 0 || *j1 <= 0) return;
    for (int i = 0; i < *m; i++) {
        float tmp = *c * a[i + (*j1-1)*(*lda)] + *s * (*f);
        *f = -(*s) * a[i + (*j1-1)*(*lda)] + *c * (*f);
        a[i + (*j1-1)*(*lda)] = tmp;
    }
}
