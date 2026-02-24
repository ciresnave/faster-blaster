/* slacul - Row scaling (absolute values) */
#
void slacul_(int* m, int* n, float* a, int* lda, float* r) {
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *m; i++) {
            float scale = fabs(r[i]);
            if (scale > 0.0f) {
                a[i + j*lda_val] *= scale;
            }
        }
    }
}
