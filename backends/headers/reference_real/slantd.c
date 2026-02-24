/* slantd - Norm of diagonal matrix */
#
void slantd_(char* norm, int* n, float* d) {
    float nrmval = 0.0f;
    for (int i = 0; i < *n; i++) {
        float absval = fabs(d[i]);
        if (absval > nrmval) nrmval = absval;
    }
}
