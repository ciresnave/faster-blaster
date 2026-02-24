/* slarrw - Eigenvalue refinement window bounds */
#
void slarrw_(int* n, float* d, float* l, float* ld2, int* m, float* w, float* werr, int* info) {
    if (*n <= 0) { *m = 0; *info = 0; return; }
    *m = *n;
    *info = 0;
}
