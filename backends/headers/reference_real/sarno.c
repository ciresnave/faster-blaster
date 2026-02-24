/* sarno */
#
void sarno_(int* n, float* a, int* lda, float* arn, float* arnorm, int* info) {
    int n_val = *n, lda_val = *lda;
    if (*n < 0) { *info = -1; return; }
    if (*lda < n_val) { *info = -3; return; }
    float maxval = 0.0f;
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < n_val; i++) {
            float absval = fabs(a[i + j*lda_val]);
            if (absval > maxval) maxval = absval;
        }
    }
    *arnorm = maxval;
    *info = 0;
}
