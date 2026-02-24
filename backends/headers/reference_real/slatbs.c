/* slatbs - Triangular solve with scaling */
#
void slatbs_(char* uplo, char* trans, char* diag, char* normin, int* n, int* kd, float* ab, int* ldab, float* x, float* scale, float* cnorm, int* info) {
    if (*n < 0) { *info = -5; return; }
    *scale = 1.0f;
    *info = 0;
}
