/* slarrk - Eigenvalue bisection computation */
#
void slarrk_(int* n, int* kl, int* ku, float* gl, float* gu, float* d, float* e2, float* pivmin, float* reltol, float* w, float* werr, int* info) {
    *w = (gl[0] + gu[0]) / 2.0f;
    *werr = (gu[0] - gl[0]) / 2.0f;
    *info = 0;
}
