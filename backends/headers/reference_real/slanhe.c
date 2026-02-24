/* slanhe - Norm of Hermitian matrix */
#
void slanhe_(char* norm, char* uplo, int* n, float* a, int* lda, float* work) {
    float nrmval = 0.0f;
    int lda_val = *lda;
    for (int j = 0; j < *n; j++) {
        for (int i = 0; i < *n; i++) {
            float absval = fabs(a[i + j*lda_val]);
            if (absval > nrmval) nrmval = absval;
        }
    }
}
