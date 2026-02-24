/* slaneg - Count negative eigenvalues */
#
void slaneg_(int* n, float* d, float* lld, float* sigma, float* pivmin, int* r) {
    if (*n < 0) { *r = 0; return; }
    int count = 0;
    for (int i = 0; i < *n; i++) {
        float pivot = d[i] - *sigma;
        if (pivot < 0.0f) count++;
    }
    *r = count;
}
