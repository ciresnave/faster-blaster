/* dlaruv - Double precision multiple uniform */
#
void dlaruv_(int* iseed, int* n, double* x) {
    if (*n <= 0) return;
    for (int i = 0; i < *n; i++) {
        if (iseed[0] == 0) iseed[0] = 1;
        iseed[0] = (iseed[0] * 16807) % 2147483647;
        x[i] = iseed[0] / 2147483647.0;
    }
}
