/* dladiv - Double precision safe complex division */
#
void dladiv_(double* a, double* b, double* c, double* d, double* p, double* q) {
    double abscd = fabs(*c) + fabs(*d);
    if (abscd < 1.0e-300) {
        *p = 0.0;
        *q = 0.0;
        return;
    }
    double denom = (*c)*(*c) + (*d)*(*d);
    *p = ((*a)*(*c) + (*b)*(*d)) / denom;
    *q = ((*b)*(*c) - (*a)*(*d)) / denom;
}
