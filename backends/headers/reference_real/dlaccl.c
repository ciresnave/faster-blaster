/* dlaccl - Double precision column scaling */
#
void dlaccl_(int* m, int* n, double* a, int* lda, double* c) {
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        double scale = fabs(c[j]);
        if (scale > 0.0) {
            for (int i = 0; i < *m; i++) {
                a[i + j*lda_val] *= scale;
            }
        }
    }
}
