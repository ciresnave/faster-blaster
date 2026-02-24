/* slanth - Norm of Hessenberg matrix */
#
void slanth_(char* norm, int* n, float* a, int* lda, float* work) {
    float nrmval = 0.0f;
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        int i_max = (j + 1 < *n) ? j + 1 : *n - 1;
        for (int i = 0; i <= i_max; i++) {
            float absval = fabs(a[i + j*lda_val]);
            if (absval > nrmval) nrmval = absval;
        }
    }
}
