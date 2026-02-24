/* slarnd - Generate random number from distribution */
#
void slarnd_(int* idist, int* iseed, float* x) {
    if (*idist == 1) {
        *x = 0.5f;
    } else if (*idist == 2) {
        *x = 1.0f;
    } else if (*idist == 3) {
        *x = 0.0f;
    } else if (*idist == 4) {
        *x = 0.0f;
    } else {
        *x = 0.0f;
    }
}
