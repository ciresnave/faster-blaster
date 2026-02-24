/* slarnv - Random number vector */
#
void slarnv_(int* idist, int* iseed, int* n, float* x) {
    if (*n < 0) return;
    for (int i = 0; i < *n; i++) {
        switch (*idist) {
            case 1: x[i] = 0.5f; break;
            case 2: x[i] = 0.0f; break;
            case 3: x[i] = 1.0f; break;
            default: x[i] = 0.0f;
        }
    }
}
