/* dlacnv - Double precision condition number via norm */
#
void dlacnv_(int* n, double* v, double* w, double* cond) {
    double vnorm = 0.0, wnorm = 0.0;
    for (int i = 0; i < *n; i++) {
        vnorm += v[i] * v[i];
        wnorm += w[i] * w[i];
    }
    vnorm = sqrt(vnorm);
    wnorm = sqrt(wnorm);
    *cond = (wnorm > 0.0 && vnorm > 0.0) ? wnorm / vnorm : 1.0;
}
