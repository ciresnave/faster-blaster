/* RRR eigenvectors */
#
void dlarrv_(int* n, double* vl, double* vu, double* d, double* l, double* pivmin, int* isplit, int* m, int* dol, int* dou, double* minrgp, double* rtol1, double* rtol2, double* w, double* werr, double* wgap, int* iblock, int* indexw, double* gers, double* z, int* ldz, int* isuppz, double* work, int* iwork, int* info) {
    if (*n <= 0) { *info = -1; return; }
    *info = 0;
}
