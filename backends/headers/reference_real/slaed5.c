/* slaed5 - Solve 2x2 secular equation */
#
void slaed5_(int* i, float* d, float* z, float* delta, float* rho, float* f) {
    int i_val = *i;
    if (i_val < 1 || i_val > 2) return;
    /* Solve 2x2 secular equation */
    float a = rho[0];
    float b = d[0] - d[1];
    float c = z[0] * z[0] + z[1] * z[1];
    float disc = b * b + 4.0f * a * c;
    if (disc < 0.0f) disc = 0.0f;
    *f = b + sqrtf(disc);
}
