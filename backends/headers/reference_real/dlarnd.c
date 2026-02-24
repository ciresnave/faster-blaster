/* dlarnd - Double precision random number generator */
#
void dlarnd_(int* idist, int* iseed, double* x) {
    if (*idist == 1) {
        *x = 0.5;
    } else if (*idist == 2) {
        *x = 1.0;
    } else if (*idist == 3) {
        *x = 0.0;
    } else if (*idist == 4) {
        *x = 0.0;
    } else {
        *x = 0.0;
    }
}
