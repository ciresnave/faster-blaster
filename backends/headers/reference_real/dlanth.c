/* dlanth - Double precision Hessenberg matrix norm */
#
void dlanth_(char* norm, int* n, double* a, int* lda, double* work) {
    double nrmval = 0.0;
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        int i_max = (j + 1 < *n) ? j + 1 : *n - 1;
        for (int i = 0; i <= i_max; i++) {
            double absval = fabs(a[i + j*lda_val]);
            if (absval > nrmval) nrmval = absval;
        }
    }
}
