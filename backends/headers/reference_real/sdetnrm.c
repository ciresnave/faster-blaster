/* sdetnrm - Determinant normalization */
#
void sdetnrm_(float* det, float* base, int* exp) {
    if (*det == 0.0f) {
        *base = 1.0f;
        *exp = 0;
        return;
    }
    float absdet = fabs(*det);
    *exp = (int)logf(absdet) / (int)logf(2.0f);
    *base = absdet / powf(2.0f, (float)(*exp));
}
