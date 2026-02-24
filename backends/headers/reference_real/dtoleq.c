/* dtoleq - Double precision tolerance equality test */
#
void dtoleq_(double* a, double* b, double* eps, int* result) {
    *result = (fabs(*a - *b) < *eps) ? 1 : 0;
}
