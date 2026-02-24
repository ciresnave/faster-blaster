/* slaesy - Solve 2x2 symmetric linear system */
#
void slaesy_(float* a, float* b, float* c, float* rx, float* ry, float* w) {
    float denom = (*a) - (*c);
    if (fabs(denom) < 1.0e-20f) denom = 1.0e-20f;
    *rx = (*rx - *b) / denom;
    *ry = (*c) * (*rx);
}
