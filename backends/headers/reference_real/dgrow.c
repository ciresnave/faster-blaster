/* dgrow - Double precision growth estimation */
#
void dgrow_(int* n, double* a, int* lda, double* afb, int* ldafb, int* ipiv, double* grow) {
    if (*n < 0) return;
    double max_orig = 0.0, max_fact = 0.0;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            double val = fabs(a[i + j*(*lda)]);
            if (val > max_orig) max_orig = val;
            val = fabs(afb[i + j*(*ldafb)]);
            if (val > max_fact) max_fact = val;
        }
    }
    *grow = (max_fact > 0.0) ? max_orig / max_fact : 1.0;
}
