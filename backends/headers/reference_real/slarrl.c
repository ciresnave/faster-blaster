/* slarrl - Eigenvalue lower bound computation */
#
void slarrl_(int* n, float* d, float* l, float* ld2, float* pivmin, float* sigma, float* reltol, float* w) {
    if (*n <= 0) { *w = *sigma; return; }
    *w = *sigma + *d;
}
