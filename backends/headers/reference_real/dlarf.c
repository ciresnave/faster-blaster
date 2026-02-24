/* dlarf - apply Householder reflection to matrix (double precision) */
void dlarf_(const char *side, const int *m, const int *n, const double *v, const int *incv,
            const double *tau, double *C, const int *ldc, double *work)
{
    int mm = *m;
    int nn = *n;
    int incv_val = *incv;
    int ldc_val = *ldc;
    char side_val = *side;
    double tau_val = *tau;
    
    if (tau_val == 0.0) {
        return;  // No reflection needed
    }
    
    if (side_val == 'L' || side_val == 'l') {
        // H = I - tau*v*v^T, apply from left: H*C
        // w = C^T * v
        for (int j = 0; j < nn; j++) {
            work[j] = 0.0;
        }
        for (int i = 0; i < mm; i++) {
            double vi = v[i * incv_val];
            if (vi != 0.0) {
                for (int j = 0; j < nn; j++) {
                    work[j] += vi * C[i + j * ldc_val];
                }
            }
        }
        
        // C = C - tau*v*w^T
        double tau_scaled = -tau_val;
        for (int i = 0; i < mm; i++) {
            double vi = v[i * incv_val];
            if (vi != 0.0) {
                vi *= tau_scaled;
                for (int j = 0; j < nn; j++) {
                    C[i + j * ldc_val] += vi * work[j];
                }
            }
        }
    } else if (side_val == 'R' || side_val == 'r') {
        // H = I - tau*v*v^T, apply from right: C*H
        // w = C * v
        for (int i = 0; i < mm; i++) {
            work[i] = 0.0;
        }
        for (int j = 0; j < nn; j++) {
            double vj = v[j * incv_val];
            if (vj != 0.0) {
                for (int i = 0; i < mm; i++) {
                    work[i] += vj * C[i + j * ldc_val];
                }
            }
        }
        
        // C = C - tau*w*v^T
        double tau_scaled = -tau_val;
        for (int j = 0; j < nn; j++) {
            double vj = v[j * incv_val];
            if (vj != 0.0) {
                vj *= tau_scaled;
                for (int i = 0; i < mm; i++) {
                    C[i + j * ldc_val] += work[i] * vj;
                }
            }
        }
    }
}
