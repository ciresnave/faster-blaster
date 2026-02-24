/* slarrj - RRR refinement bisection */
#
void slarrj_(int* n, float* d, float* e2, int* ifirst, int* ilast, float* rtol, int* offset, float* w, float* werr, float* work, int* iwork, int* info) {
    *info = 0;
    for (int i = 0; i < *n; i++) {
        work[i] = d[i];
    }
}
