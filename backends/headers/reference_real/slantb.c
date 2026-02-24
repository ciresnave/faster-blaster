/* slantb - Norm of banded triangular matrix */
#
void slantb_(char* norm, char* uplo, char* diag, int* n, int* k, float* ab, int* ldab, float* work) {
    float nrmval = 0.0f;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i <= *k && i < *n; i++) {
            float absval = fabs(ab[i + j*(*ldab)]);
            if (absval > nrmval) nrmval = absval;
        }
    }
}
