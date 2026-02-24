/* sconst - Generate constant matrix */
#
void sconst_(int* n, float* val, float* a, int* lda) {
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            a[i + j*lda_val] = *val;
        }
    }
}
