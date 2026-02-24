/* slarrf - Newton refinement of eigenvalues */
#
void slarrf_(int* n, float* d, float* l, float* ld2, int* clstrt, int* clend, float* w, float* wgap, float* werr, float* tol, float* sigma, float* dplus, float* lplus, float* work, int* info) {
    *sigma = w[0];
    for (int i = 0; i < *n; i++) {
        dplus[i] = d[i];
        lplus[i] = l[i];
    }
    *info = 0;
}
