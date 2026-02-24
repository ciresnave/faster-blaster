/* slaru - Uniform random in [0,1) */
#
void slaru_(int* iseed, float* x) {
    if (iseed[0] == 0) iseed[0] = 1;
    iseed[0] = (iseed[0] * 16807) % 2147483647;
    *x = iseed[0] / 2147483647.0f;
}
