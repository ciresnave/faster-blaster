/* slaruv - Multiple uniform randoms */
#
void slaruv_(int* iseed, int* n, float* x) {
    if (*n <= 0) return;
    for (int i = 0; i < *n; i++) {
        if (iseed[0] == 0) iseed[0] = 1;
        iseed[0] = (iseed[0] * 16807) % 2147483647;
        x[i] = iseed[0] / 2147483647.0f;
    }
}
