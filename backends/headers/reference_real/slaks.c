/* slaks - Gershgorin disc computation */
#
void slaks_(int* n, float* d, float* e, int* k, float* c, float* s) {
    if (*n < 1 || *k < 1 || *k > *n) { *c = 0.0f; *s = 0.0f; return; }
    float rad = fabs(e[*k - 1]);
    if (*k > 1) rad += fabs(e[*k - 2]);
    if (*k < *n) rad += fabs(e[*k]);
    *c = d[*k - 1];
    *s = rad;
}
