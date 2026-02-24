/* dlarrw - Double precision eigenvalue window */
#
void dlarrw_(int* n, double* d, double* l, double* ld2, int* m, double* w, double* werr, int* info) {
    if (*n <= 0) { *m = 0; *info = 0; return; }
    *m = *n;
    *info = 0;
}
