/* dlaru - Double precision uniform random */
#
void dlaru_(int* iseed, double* x) {
    if (iseed[0] == 0) iseed[0] = 1;
    iseed[0] = (iseed[0] * 16807) % 2147483647;
    *x = iseed[0] / 2147483647.0;
}
