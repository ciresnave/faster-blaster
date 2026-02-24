/* dlantp - Double precision packed triangular norm */
#
void dlantp_(char* norm, char* uplo, char* diag, int* n, double* ap, double* work, double* nrmval) {
    if (*n < 0) return;
    *nrmval = 0.0;
    for (int i = 0; i < *n; i++) {
        if (fabs(ap[i]) > *nrmval) *nrmval = fabs(ap[i]);
    }
}
