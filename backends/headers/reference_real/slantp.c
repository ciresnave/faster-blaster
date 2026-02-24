/* slantp - Packed triangular matrix norm */
#
void slantp_(char* norm, char* uplo, char* diag, int* n, float* ap, float* work, float* nrmval) {
    if (*n < 0) return;
    *nrmval = 0.0f;
    for (int i = 0; i < *n; i++) {
        if (fabs(ap[i]) > *nrmval) *nrmval = fabs(ap[i]);
    }
}
