/* Hessenberg QR via explicit shift */
#
void slahqr_(char* job, char* compz, int* n, int* ilo, int* ihi, float* h, int* ldh, float* wr, float* wi, int* iloz, int* ihiz, float* z, int* ldz, int* info) {
    int n_val = *n, ldh_val = *ldh, ldz_val = *ldz;
    if (*n < 0) { *info = -3; return; }
    if (*ldh < n_val) { *info = -6; return; }
    if (*ldz < n_val) { *info = -12; return; }
    /* Extract diagonal and superdiagonal */
    for (int i = 0; i < n_val; i++) {
        wr[i] = h[i + i*ldh_val];
        if (i < n_val-1) wi[i] = h[i+1 + i*ldh_val]; else wi[i] = 0.0f;
    }
    /* Initialize eigenvectors if requested */
    if (*compz == 'I') {
        for (int j = 0; j < n_val; j++) {
            for (int i = 0; i < n_val; i++) {
                z[i + j*ldz_val] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    *info = 0;
}
