/* sladiv - Safe complex division */
#
void sladiv_(float* a, float* b, float* c, float* d, float* p, float* q) {
    float abscd = fabs(*c) + fabs(*d);
    if (abscd < 1.0e-30f) {
        *p = 0.0f;
        *q = 0.0f;
        return;
    }
    float denom = (*c)*(*c) + (*d)*(*d);
    *p = ((*a)*(*c) + (*b)*(*d)) / denom;
    *q = ((*b)*(*c) - (*a)*(*d)) / denom;
}
