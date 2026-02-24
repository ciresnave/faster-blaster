/* dlakz - Double precision secular equation bounds */
#
void dlakz_(int* n, double* d, double* e, int* k, double* z, double* w, double* lambd, double* lam) {
    if (*n < 1) { *lam = 0.0; return; }
    double sigma = d[*k - 1];
    *lam = sigma + *lambd;
}
