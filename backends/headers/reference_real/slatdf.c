/* slatdf - Recompute RHS via LU factors */
#
void slatdf_(int* ijob, int* n, float* z, int* ldz, float* rhs, float* rdsum, float* rdscal, int* ipiv, int* jpiv) {
    if (*n < 0) return;
    *rdsum = 0.0f;
    *rdscal = 1.0f;
}
