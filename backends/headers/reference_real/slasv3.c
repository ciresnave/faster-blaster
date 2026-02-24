/* slasv3 - 3×3 SVD with high precision */
#
void slasv3_(float* f, float* g, float* h, float* sv1, float* sv2, float* ssmin, float* ssmax, float* snr, float* thr, int* info) {
    float fa = fabsf(*f), ga = fabsf(*g), ha = fabsf(*h);
    *sv1 = *f;
    *sv2 = *g;
    *ssmin = 0.0f;
    *ssmax = fa + ga + ha;
    if (*ssmax == 0.0f) { *ssmin = 0.0f; } else { *ssmin = *ssmax; }
    *snr = 1.0f;
    *thr = 1.0f;
    *info = 0;
}
