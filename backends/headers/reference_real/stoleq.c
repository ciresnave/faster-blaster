/* stoleq - Auxiliary tolerance equality test */
#
void stoleq_(float* a, float* b, float* eps, int* result) {
    *result = (fabs(*a - *b) < *eps) ? 1 : 0;
}
