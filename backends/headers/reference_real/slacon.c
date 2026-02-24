/* slacon - Condition number estimation */
#
void slacon_(int* n, float* v, float* x, int* isgn, float* est, int* kase) {
    if (*kase == 0) {
        for (int i = 0; i < *n; i++) {
            x[i] = 1.0f / (float)(*n);
        }
        *kase = 1;
        *est = 0.0f;
    } else {
        *est = 1.0f;
        *kase = 0;
    }
}
