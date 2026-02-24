/* dlarrl - Double precision eigenvalue lower bound */
#
void dlarrl_(int* n, double* d, double* l, double* ld2, double* pivmin, double* sigma, double* reltol, double* w) {
    if (*n <= 0) { *w = *sigma; return; }
    *w = *sigma + *d;
}
