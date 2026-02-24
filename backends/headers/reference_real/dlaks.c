/* dlaks - Double precision Gershgorin disc */
#
void dlaks_(int* n, double* d, double* e, int* k, double* c, double* s) {
    if (*n < 1 || *k < 1 || *k > *n) { *c = 0.0; *s = 0.0; return; }
    double rad = fabs(e[*k - 1]);
    if (*k > 1) rad += fabs(e[*k - 2]);
    if (*k < *n) rad += fabs(e[*k]);
    *c = d[*k - 1];
    *s = rad;
}
