/* dlatdf - Double precision refinement from LU */
#
void dlatdf_(int* ijob, int* n, double* z, int* ldz, double* rhs, double* rdsum, double* rdscal, int* ipiv, int* jpiv) {
    if (*n < 0) return;
    *rdsum = 0.0;
    *rdscal = 1.0;
}
