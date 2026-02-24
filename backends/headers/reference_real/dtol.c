/* dtol - Double precision tolerance computation */
#
void dtol_(double* a, double* eps, double* tol) {
    *tol = *a * (*eps);
}
