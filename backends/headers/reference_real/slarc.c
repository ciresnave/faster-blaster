/* Reciprocal condition */
#
void slarc_(char* type, int* n, float* a, float* b, float* c, float* s) {
    float r = sqrt((*a)*(*a) + (*b)*(*b));
    if (r > 0.0f) {
        *c = (*a) / r;
        *s = (*b) / r;
    } else {
        *c = 1.0f;
        *s = 0.0f;
    }
}
