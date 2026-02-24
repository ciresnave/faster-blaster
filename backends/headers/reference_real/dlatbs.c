/* dlatbs - Double precision triangular solve with scaling */
#
void dlatbs_(char* uplo, char* trans, char* diag, char* normin, int* n, int* kd, double* ab, int* ldab, double* x, double* scale, double* cnorm, int* info) {
    if (*n < 0) { *info = -5; return; }
    *scale = 1.0;
    *info = 0;
}
