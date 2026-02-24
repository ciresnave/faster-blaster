/* slakz - Bounds for secular equation eigenvalues */
#
void slakz_(int* n, float* d, float* e, int* k, float* z, float* w, float* lambd, float* lam) {
    if (*n < 1) { *lam = 0.0f; return; }
    float sigma = d[*k - 1];
    *lam = sigma + *lambd;
}
