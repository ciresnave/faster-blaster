/* dconst - Double precision constant matrix */
#
void dconst_(int* n, double* val, double* a, int* lda) {
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            a[i + j*lda_val] = *val;
        }
    }
}
