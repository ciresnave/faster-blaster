/* dlarow - Double precision Givens row rotation */
#
void dlarow_(int* m, double* a, int* lda, int* j1, double* f, double* c, double* s) {
    if (*m <= 0 || *j1 <= 0) return;
    for (int i = 0; i < *m; i++) {
        double tmp = *c * a[i + (*j1-1)*(*lda)] + *s * (*f);
        *f = -(*s) * a[i + (*j1-1)*(*lda)] + *c * (*f);
        a[i + (*j1-1)*(*lda)] = tmp;
    }
}
