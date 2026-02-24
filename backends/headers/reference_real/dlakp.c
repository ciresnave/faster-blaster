/* dlakp - Double precision perturbation bounds */
#
void dlakp_(int* n, double* d, double* e, int* k, double* z, double* w, double* lambd, double* relerr) {
    if (*n < 1) { *relerr = 0.0; return; }
    double numer = fabs(*w);
    double denom = fabs(*lambd);
    *relerr = denom > 0.0 ? numer / denom : numer;
}
