/* dlahr - Double precision long residual */
#
void dlahr_(double* a, double* b, double* r, double* s) {
    if (*a == 0.0 || *b == 0.0) {
        *r = 0.0;
        *s = 0.0;
        return;
    }
    double prod = (*a) * (*b);
    *s = prod - floor(prod);
    *r = floor(prod);
}
