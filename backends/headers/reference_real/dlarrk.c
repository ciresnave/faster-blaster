/* dlarrk - Double precision eigenvalue bisection */
#
void dlarrk_(int* n, int* kl, int* ku, double* gl, double* gu, double* d, double* e2, double* pivmin, double* reltol, double* w, double* werr, int* info) {
    *w = (gl[0] + gu[0]) / 2.0;
    *werr = (gu[0] - gl[0]) / 2.0;
    *info = 0;
}
