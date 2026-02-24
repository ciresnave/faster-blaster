/* dlantb - Double precision banded triangular norm */
#
void dlantb_(char* norm, char* uplo, char* diag, int* n, int* k, double* ab, int* ldab, double* work) {
    double nrmval = 0.0;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i <= *k && i < *n; i++) {
            double absval = fabs(ab[i + j*(*ldab)]);
            if (absval > nrmval) nrmval = absval;
        }
    }
}
