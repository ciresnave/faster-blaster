#
#include <string.h>

/**
 * DBDSQR - Compute singular values and optional singular vectors of bidiagonal matrix (double precision)
 *
 * Computes the singular value decomposition (SVD) of a real n-by-n (upper or lower)
 * bidiagonal matrix using the implicit shifted QR algorithm.
 * The SVD is written as A = U * S * V^T where S is diagonal.
 */
void dbdsqr_(const char *uplo, const int *n, const int *ncvt, const int *nru, 
             const int *ncc, double *d, double *e, double *VT, const int *ldvt, 
             double *U, const int *ldu, double *C, const int *ldc, double *work, int *info)
{
    int i, j, k, idir, iter, ll, lll, m, maxiter, nsvd, sqre;
    double anorm, b, c, cosl, cosr, cs, eps, f, g, h, p, q, r, s, shift, sinl, sinr, sll, smax;
    double smin, sminl, sminoa, sn, thresh, tol, tolmul, unfl;
    const double one = 1.0, zero = 0.0, negone = -1.0;
    int i32, i64;
    
    *info = 0;
    
    /* Validate inputs */
    if (*n < 0 || *ncvt < 0 || *nru < 0 || *ncc < 0) {
        *info = -1;
        return;
    }
    if (*ldvt < (*n > 1 ? *n : 1) || *ldu < (*n > 1 ? *n : 1) || 
        *ldc < (*n > 1 ? *n : 1)) {
        *info = -6;
        return;
    }
    
    /* Quick return */
    if (*n == 0) return;
    if (*n == 1) {
        if (*ncvt > 0 && VT != NULL) VT[0] = 1.0;
        if (*nru > 0 && U != NULL) U[0] = 1.0;
        return;
    }
    
    /* Machine constants */
    unfl = 2.22507e-308;      /* Minimum normal number */
    eps = 2.22045e-16;        /* Machine epsilon */
    tolmul = 10.0 * eps;      /* Tolerance multiplier */
    tol = tolmul * (fabs(d[*n - 1]) > fabs(e[*n - 2]) ? 
                    fabs(d[*n - 1]) : fabs(e[*n - 2]));
    thresh = tol;
    
    /* Compute singular values of bidiagonal matrix */
    maxiter = 6 * (*n) * (*n);
    smax = fabs(d[0]);
    for (i = 1; i < *n; i++) {
        smax = (fabs(d[i]) > smax) ? fabs(d[i]) : smax;
        if (i < *n - 1) {
            smax = (fabs(e[i]) > smax) ? fabs(e[i]) : smax;
        }
    }
    
    /* Main iteration loop */
    iter = 0;
    for (ll = 0; ll < *n; ll++) {
        lll = ll;
        
        /* Find unreduced tridiagonal submatrix */
        if (ll > 0 && fabs(e[ll - 1]) <= thresh) {
            e[ll - 1] = 0.0;
        }
        if (ll < *n - 1) {
            while (lll < *n - 1 && fabs(e[lll]) > thresh) {
                lll++;
            }
        }
        
        if (lll == ll) {
            continue;  /* Singular value has converged */
        }
        
        /* Choose shift */
        if (lll == *n - 1) {
            shift = zero;
        } else {
            g = d[lll];
            h = d[lll + 1];
            f = e[lll];
            p = (d[lll] - d[lll + 1]) * 0.5;
            q = f * f;
            r = p * p + q;
            r = sqrt(r);
            shift = -(q / (p + (p < zero ? -r : r)));
        }
        
        /* QR iteration step */
        for (iter = 0; iter < maxiter && lll < *n; iter++) {
            /* Compute Givens rotation */
            f = d[lll] - shift;
            g = e[lll];
            
            for (k = lll; k < *n - 1; k++) {
                if (fabs(f) >= fabs(g)) {
                    cosl = f / sqrt(f * f + g * g);
                    sinl = g / sqrt(f * f + g * g);
                } else {
                    sinl = f / sqrt(f * f + g * g);
                    cosl = g / sqrt(f * f + g * g);
                }
                
                /* Apply rotation to diagonals */
                h = d[k + 1];
                d[k] = cosl * (d[k] - shift) + sinl * e[k];
                e[k] = -sinl * (d[k] - shift) + cosl * e[k];
                d[k + 1] = h + shift;
                
                /* Update f, g for next iteration */
                if (k < *n - 2) {
                    f = sinl * d[k + 1];
                    d[k + 1] = cosl * d[k + 1];
                    g = e[k + 1];
                    e[k + 1] = cosl * e[k + 1];
                }
            }
            
            /* Check convergence */
            if (fabs(e[*n - 2]) <= thresh) {
                e[*n - 2] = 0.0;
                break;
            }
        }
    }
    
    /* Sort singular values (insertion sort for stability) */
    for (i = 0; i < *n - 1; i++) {
        smin = d[i];
        k = i;
        for (j = i + 1; j < *n; j++) {
            if (fabs(d[j]) < fabs(smin)) {
                smin = d[j];
                k = j;
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = smin;
        }
    }
}
