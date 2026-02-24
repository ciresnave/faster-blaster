/* slacnv - Condition number via norm */
#
void slacnv_(int* n, float* v, float* w, float* cond) {
    float vnorm = 0.0f, wnorm = 0.0f;
    for (int i = 0; i < *n; i++) {
        vnorm += v[i] * v[i];
        wnorm += w[i] * w[i];
    }
    vnorm = sqrt(vnorm);
    wnorm = sqrt(wnorm);
    *cond = (wnorm > 0.0f && vnorm > 0.0f) ? wnorm / vnorm : 1.0f;
}
