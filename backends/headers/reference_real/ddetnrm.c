/* ddetnrm - Double precision determinant norm */
#
void ddetnrm_(double* det, double* base, int* exp) {
    if (*det == 0.0) {
        *base = 1.0;
        *exp = 0;
        return;
    }
    double absdet = fabs(*det);
    *exp = (int)log(absdet) / (int)log(2.0);
    *base = absdet / pow(2.0, (double)(*exp));
}
