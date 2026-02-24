/* sgrow - Growth estimation component */
#
void sgrow_(int* n, float* a, int* lda, float* afb, int* ldafb, int* ipiv, float* grow) {
    if (*n < 0) return;
    float max_orig = 0.0f, max_fact = 0.0f;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            float val = fabs(a[i + j*(*lda)]);
            if (val > max_orig) max_orig = val;
            val = fabs(afb[i + j*(*ldafb)]);
            if (val > max_fact) max_fact = val;
        }
    }
    *grow = (max_fact > 0.0f) ? max_orig / max_fact : 1.0f;
}
