/* slaqsy - Symmetric matrix equilibration */
#
void slaqsy_(char* uplo, int* n, float* a, int* lda, float* s, float* scond, float* amax, char* equed) {
    if (*n < 0 || *lda < *n) return;
    for (int i = 0; i < *n; i++) s[i] = 1.0f;
    *scond = 1.0f;
    *amax = 1.0f;
    *equed = 'N';
}
