/* stol - Auxiliary tolerance computation */
#
void stol_(float* a, float* eps, float* tol) {
    *tol = *a * (*eps);
}
