/* dlaesy - Double precision 2x2 symmetric system solve */
#
void dlaesy_(double* a, double* b, double* c, double* rx, double* ry, double* w) {
    double denom = (*a) - (*c);
    if (fabs(denom) < 1.0e-300) denom = 1.0e-300;
    *rx = (*rx - *b) / denom;
    *ry = (*c) * (*rx);
}
