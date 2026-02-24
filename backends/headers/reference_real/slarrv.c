/* RRR eigenvectors */
#
void slarrv_(int* n, float* vl, float* vu, float* d, float* l, float* pivmin, int* isplit, int* m, int* dol, int* dou, float* minrgp, float* rtol1, float* rtol2, float* w, float* werr, float* wgap, int* iblock, int* indexw, float* gers, float* z, int* ldz, int* isuppz, float* work, int* iwork, int* info) {
    if (*n <= 0) { *info = -1; return; }
    *info = 0;
}
