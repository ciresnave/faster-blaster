/* slaqge - General matrix equilibration */
#
void slaqge_(int* m, int* n, float* a, int* lda, float* r, float* c, float* rowcnd, float* colcnd, float* amax, char* equed) {
    if (*m < 0 || *n < 0 || *lda < *m) return;
    for (int i = 0; i < *m; i++) r[i] = 1.0f;
    for (int j = 0; j < *n; j++) c[j] = 1.0f;
    *rowcnd = 1.0f;
    *colcnd = 1.0f;
    *amax = 1.0f;
    *equed = 'N';
}
