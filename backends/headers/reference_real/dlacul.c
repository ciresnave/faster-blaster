/* dlacul - Double precision row scaling */
#
void dlacul_(int* m, int* n, double* a, int* lda, double* r) {
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *m; i++) {
            double scale = fabs(r[i]);
            if (scale > 0.0) {
                a[i + j*lda_val] *= scale;
            }
        }
    }
}
