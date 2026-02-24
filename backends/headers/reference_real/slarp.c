/* slarp - Givens rotation elimination */
#
void slarp_(int* j, float* v, int* incv, float* x, int* incx, float* c, float* s) {
    if (*j <= 0) return;
    float x0 = x[0];
    float v0 = v[0];
    x[0] = *c * x0 + *s * v0;
    v[0] = -(*s) * x0 + *c * v0;
}
