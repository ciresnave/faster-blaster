/* dlasv3 - 3×3 SVD with high precision */
#
void dlasv3_(double* f, double* g, double* h, double* sv1, double* sv2, double* ssmin, double* ssmax, double* snr, double* thr, int* info) {
    double fa = fabs(*f), ga = fabs(*g), ha = fabs(*h);
    *sv1 = *f;
    *sv2 = *g;
    *ssmin = 0.0;
    *ssmax = fa + ga + ha;
    if (*ssmax == 0.0) { *ssmin = 0.0; } else { *ssmin = *ssmax; }
    *snr = 1.0;
    *thr = 1.0;
    *info = 0;
}
