/* slahr - Long residual computation */
#
void slahr_(float* a, float* b, float* r, float* s) {
    if (*a == 0.0f || *b == 0.0f) {
        *r = 0.0f;
        *s = 0.0f;
        return;
    }
    float prod = (*a) * (*b);
    *s = prod - floorf(prod);
    *r = floorf(prod);
}
