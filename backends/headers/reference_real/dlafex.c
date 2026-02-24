/* dlafex - Double precision find maximum index */
#
void dlafex_(int* n, double* a, int* lda, int* idx) {
    int max_idx = 0;
    double max_val = 0.0;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            double absval = fabs(a[i + j*(*lda)]);
            if (absval > max_val) { max_val = absval; max_idx = i + j*(*lda); }
        }
    }
    *idx = max_idx + 1;
}
