/* slannb - Norm of symmetric banded matrix */
#
void slannb_(char* norm, int* n, int* k, float* ab, int* ldab, float* work) {
    float nrmval = 0.0f;
    for (int i = 0; i < *n; i++) {
        for (int j = 0; j <= *k && i + j < *n; j++) {
            float absval = fabs(ab[j + i*(*ldab)]);
            if (absval > nrmval) nrmval = absval;
        }
    }
}
