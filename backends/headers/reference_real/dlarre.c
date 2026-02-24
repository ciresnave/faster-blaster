/* RRR eigenvalues */
#
void dlarre_(char* range, int* n, double* vl, double* vu, int* il, int* iu, double* d, double* e, double* e2, double* reltol, double* rtol1, double* rtol2, int* nsplit, int* isplit, int* m, double* w, double* werr, double* wgap, int* iblock, int* indexw, double* gers, double* pivmin, double* work, int* iwork, int* info) {
    if (*n <= 0) { *info = -2; return; }
    *info = 0;
}
