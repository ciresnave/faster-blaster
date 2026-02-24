/* slaccl - Column scaling (absolute values) */
#
void slaccl_(int* m, int* n, float* a, int* lda, float* c) {
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        float scale = fabs(c[j]);
        if (scale > 0.0f) {
            for (int i = 0; i < *m; i++) {
                a[i + j*lda_val] *= scale;
            }
        }
    }
}
