/* slacsf - Scaling factors for symmetric equilibration */
#
void slacsf_(int* n, float* d, float* s, int* info) {
    if (*n < 0) { *info = -1; return; }
    for (int i = 0; i < *n; i++) {
        float scale = fabs(d[i]);
        s[i] = (scale > 0.0f) ? (1.0f / sqrt(scale)) : 1.0f;
    }
    *info = 0;
}
