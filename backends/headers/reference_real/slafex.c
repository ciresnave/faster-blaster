/* slafex - Find index of maximum element */
#
void slafex_(int* n, float* a, int* lda, int* idx) {
    int max_idx = 0;
    float max_val = 0.0f;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            float absval = fabs(a[i + j*(*lda)]);
            if (absval > max_val) { max_val = absval; max_idx = i + j*(*lda); }
        }
    }
    *idx = max_idx + 1;
}
