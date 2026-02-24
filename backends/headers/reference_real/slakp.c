/* slakp - Perturbation bounds via relative residual */
#
void slakp_(int* n, float* d, float* e, int* k, float* z, float* w, float* lambd, float* relerr) {
    if (*n < 1) { *relerr = 0.0f; return; }
    float numer = fabs(*w);
    float denom = fabs(*lambd);
    *relerr = denom > 0.0f ? numer / denom : numer;
}
