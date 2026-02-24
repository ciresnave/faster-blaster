/* dlaneg - Double precision negative eigenvalue count */
#
void dlaneg_(int* n, double* d, double* lld, double* sigma, double* pivmin, int* r) {
    if (*n < 0) { *r = 0; return; }
    int count = 0;
    for (int i = 0; i < *n; i++) {
        double pivot = d[i] - *sigma;
        if (pivot < 0.0) count++;
    }
    *r = count;
}
