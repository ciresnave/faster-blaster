/* dlannb - Double precision symmetric banded norm */
#
void dlannb_(char* norm, int* n, int* k, double* ab, int* ldab, double* work) {
    double nrmval = 0.0;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j <= *k && i + j < *n; j++) {
            double absval = fabs(ab[j + i*(*ldab)]);
            if (absval > nrmval) nrmval = absval;
        }
    }
}
