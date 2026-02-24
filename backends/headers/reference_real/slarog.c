/* slarog - Givens rotation generator */
#
void slarog_(float* x, float* y, float* c, float* s) {
    float r = sqrtf((*x)*(*x) + (*y)*(*y));
    if (r == 0.0f) {
        *c = 1.0f;
        *s = 0.0f;
    } else {
        *c = (*x) / r;
        *s = (*y) / r;
    }
}
