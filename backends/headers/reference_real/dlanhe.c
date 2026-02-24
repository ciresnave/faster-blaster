/* dlanhe - Double precision Hermitian matrix norm */
#
void dlanhe_(char* norm, char* uplo, int* n, double* a, int* lda, double* work) {
    double nrmval = 0.0;
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            double absval = fabs(a[i + j*lda_val]);
            if (absval > nrmval) nrmval = absval;
        }
    }
}
