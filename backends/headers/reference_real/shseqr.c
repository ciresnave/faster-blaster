#
#include <float.h>
#include <string.h>

/**
 * SHSEQR - Compute eigenvalues and Schur decomposition of upper Hessenberg matrix
 *
 * Computes the eigenvalues and optionally the Schur form of an upper Hessenberg
 * matrix using the implicit shifted QR algorithm.
 *
 * Parameters:
 * job - 'E' (eigenvalues only), 'S' (eigenvalues + Schur form)
 * compz - 'N' (no Z), 'I' (initialize Z), 'V' (update Z)
 * n - order of matrix
 * ilo, ihi - rows of interest (typically 1 and n)
 * H - upper Hessenberg matrix (n x n)
 * ldh - leading dimension of H
 * wr, wi - real and imaginary parts of eigenvalues
 * z - orthogonal matrix (if compz != 'N')
 * ldz - leading dimension of Z
 * work - workspace
 * lwork - size of workspace
 * info - output status
 */
void shseqr_(const char *job, const char *compz, const int *n, const int *ilo,
             const int *ihi, float *H, const int *ldh, float *wr, float *wi,
             float *z, const int *ldz, float *work, const int *lwork, int *info)
{
    int i, j, k, l, m, iter;
    int istart, iend;
    float h11, h12, h21, h22;
    float p, q, r, s, t;
    float cs, sn, w;
    float abs_h;
    const float eps = FLT_EPSILON;
    const float safmin = 2.0f * FLT_MIN;
    const float safmax = 1.0f / safmin;
    const float small_thresh = safmin * safmax;
    const int max_iters = 10000;
    
    *info = 0;
    
    /* Validate inputs */
    if (*n < 0) { *info = -3; return; }
    if (*ldh < (*n > 1 ? *n : 1)) { *info = -6; return; }
    if (*ldz < (*n > 1 ? *n : 1)) { *info = -10; return; }
    
    if (*n == 0) return;
    if (*n == 1) {
        wr[0] = H[0];
        wi[0] = 0.0f;
        return;
    }
    
    /* Implicit shifted QR algorithm */
    istart = *ilo;
    iend = *ihi;
    
    /* Main iteration loop */
    for (l = istart; l <= iend; l++) {
        iter = 0;
        
        /* Find active block */
        while (l < iend && iter < max_iters) {
            iter++;
            
            /* Check for deflation */
            if (l < iend - 1) {
                abs_h = fabsf(H[l * *ldh + (l-1)]);
                if (abs_h < safmin) break;
                if (abs_h < eps * (fabsf(H[l * *ldh + l]) + fabsf(H[(l+1) * *ldh + (l+1)]))) {
                    H[l * *ldh + (l-1)] = 0.0f;
                    break;
                }
            }
            
            /* Select shift from 2x2 bottom block */
            h11 = H[l * *ldh + l];
            h12 = H[l * *ldh + (l+1)];
            h21 = H[(l+1) * *ldh + l];
            h22 = H[(l+1) * *ldh + (l+1)];
            
            /* Wilkinson shift */
            w = (h11 - h22) / 2.0f;
            p = w * w + h12 * h21;
            q = h11 - (p / w) * (h12 / (h11 + h22));
            if (q < 0.0f) q = -q;
            
            /* Apply shifts via Francis double-shift QR */
            for (m = l; m < iend - 1; m++) {
                /* Compute Givens rotation */
                if (m == l) {
                    p = H[m * *ldh + m] - (h11 + h22) + sqrtf(w * w + h12 * h21);
                    q = H[(m+1) * *ldh + m];
                    r = (m + 2 <= iend) ? H[(m+2) * *ldh + m] : 0.0f;
                } else {
                    p = H[m * *ldh + (m-1)];
                    q = H[(m+1) * *ldh + (m-1)];
                    r = (m + 2 <= iend) ? H[(m+2) * *ldh + (m-1)] : 0.0f;
                    H[m * *ldh + (m-1)] = 0.0f;
                }
                
                /* Givens rotation parameters */
                t = sqrtf(p * p + q * q + r * r);
                if (t < safmin) {
                    cs = 1.0f;
                    sn = 0.0f;
                } else {
                    cs = p / t;
                    sn = q / t;
                }
                
                /* Apply rotation to H */
                float h_tmp1, h_tmp2, h_tmp3;
                for (j = m; j < iend; j++) {
                    h_tmp1 = H[m * *ldh + j];
                    h_tmp2 = H[(m+1) * *ldh + j];
                    H[m * *ldh + j] = cs * h_tmp1 + sn * h_tmp2;
                    H[(m+1) * *ldh + j] = -sn * h_tmp1 + cs * h_tmp2;
                }
                
                /* Apply rotation to left */
                for (i = l; i <= m + 2 && i < iend; i++) {
                    h_tmp1 = H[i * *ldh + m];
                    h_tmp2 = H[i * *ldh + (m+1)];
                    H[i * *ldh + m] = cs * h_tmp1 + sn * h_tmp2;
                    H[i * *ldh + (m+1)] = -sn * h_tmp1 + cs * h_tmp2;
                }
                
                /* Update Z if needed */
                if (compz[0] != 'N') {
                    for (i = 0; i < *n; i++) {
                        h_tmp1 = z[i * *ldz + m];
                        h_tmp2 = z[i * *ldz + (m+1)];
                        z[i * *ldz + m] = cs * h_tmp1 + sn * h_tmp2;
                        z[i * *ldz + (m+1)] = -sn * h_tmp1 + cs * h_tmp2;
                    }
                }
            }
        }
    }
    
    /* Extract eigenvalues */
    for (i = 0; i < *n; i++) {
        wr[i] = H[i * *ldh + i];
        wi[i] = 0.0f;
        
        /* Check for 2x2 blocks (complex eigenvalues) */
        if (i < *n - 1) {
            float sub_diag = H[(i+1) * *ldh + i];
            if (fabsf(sub_diag) > safmin) {
                /* Complex pair */
                float trace = H[i * *ldh + i] + H[(i+1) * *ldh + (i+1)];
                float prod = H[i * *ldh + i] * H[(i+1) * *ldh + (i+1)] - H[i * *ldh + (i+1)] * sub_diag;
                float disc = trace * trace - 4.0f * prod;
                
                if (disc < 0.0f) {
                    /* Complex eigenvalues */
                    wr[i] = trace / 2.0f;
                    wi[i] = sqrtf(-disc) / 2.0f;
                    wr[i+1] = trace / 2.0f;
                    wi[i+1] = -wi[i];
                    i++; /* Skip next eigenvalue */
                }
            }
        }
    }
}
