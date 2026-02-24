/* slahv - Householder vector generation */
#
void slahv_(int* n, float* x, float* tau, float* c, float* s) {
    if (*n <= 0) return;
    float xnorm = 0.0f;
    for (int i = 1; i < *n; i++) {
        xnorm += x[i] * x[i];
    }
    xnorm = sqrt(xnorm);
    if (xnorm <= 0.0f) {
        *tau = 0.0f;
        *c = 1.0f;
        *s = 0.0f;
    } else {
        float alpha = x[0];
        float r = sqrt(alpha*alpha + xnorm*xnorm);
        *c = alpha / r;
        *s = xnorm / r;
        *tau = (*s) / (1.0f + *c);
    }
}
