#
#include <string.h>
#include <float.h>

/**
 * DSTEQR - Compute eigenvalues and eigenvectors of symmetric tridiagonal matrix (double precision)
 *
 * Computes all eigenvalues and, optionally, eigenvectors of a symmetric
 * tridiagonal matrix by QR iteration.
 */
void dsteqr_(const char *compz, const int *n, double *d, double *e, double *Z, const int *ldz,
             double *work, int *info)
{
    int i, jtot, j, k, l, m, maxit, nmaxit;
    double anorm, b, c, eps, f, g, p, r, s, eps2, safety;
    double *tau;
    const double one = 1.0, zero = 0.0;
    int doscale;
    
    *info = 0;
    
    /* Input validation */
    if (compz[0] != 'N' && compz[0] != 'V' && compz[0] != 'I') {
        *info = -1;
        return;
    }
    if (*n < 0) {
        *info = -2;
        return;
    }
    if (*ldz < (*n > 1 ? *n : 1)) {
        *info = -6;
        return;
    }
    
    /* Quick return */
    if (*n == 0) {
        return;
    }
    
    if (*n == 1) {
        if (compz[0] == 'V' || compz[0] == 'I') {
            Z[0] = 1.0;
        }
        return;
    }
    
    /* Initialize eigenvector matrix if needed */
    if (compz[0] == 'I') {
        /* Initialize Z to identity */
        memset(Z, 0, (*n) * (*ldz) * sizeof(double));
        for (i = 0; i < *n; i++) {
            Z[i + i * (*ldz)] = 1.0;
        }
    }
    
    /* Machine constants and parameters */
    eps = DBL_EPSILON;
    eps2 = eps * eps;
    safety = 1.0e+1;
    maxit = 30 * (*n) * (*n);
    nmaxit = 2 * maxit;
    
    /* Compute anorm for scaling */
    anorm = 0.0;
    for (i = 0; i < *n; i++) {
        if (i == 0) {
            anorm = fabs(d[i]);
        } else {
            anorm = fmax(anorm, fabs(d[i]) + fabs(e[i - 1]));
        }
    }
    anorm = fmax(anorm, fabs(d[*n - 1]));
    doscale = (anorm > safety);
    if (doscale) {
        /* Scale matrix by safety factor */
        for (i = 0; i < *n; i++) {
            d[i] = d[i] / safety;
        }
        for (i = 0; i < *n - 1; i++) {
            e[i] = e[i] / safety;
        }
    }
    
    /* QR iteration with implicit shift */
    jtot = 0;
    for (l = 0; l < *n; l++) {
        
        if (jtot > nmaxit) {
            *info = l;
            return;
        }
        
        m = l;
        for (j = l; j < *n - 1; j++) {
            if (fabs(e[j]) <= eps * fabs(d[j]) * fabs(d[j + 1])) {
                m = j + 1;
                break;
            }
            m = *n;
        }
        
        if (m == l) {
            continue;
        }
        
        /* QR iteration step */
        for (k = 0; k < maxit && m > l; k++) {
            jtot++;
            
            /* Choose shift */
            if (m > l + 1) {
                g = d[m - 1];
                p = (d[m] - g) * 0.5;
                r = sqrt(p * p + e[m - 2] * e[m - 2]);
                d[m - 1] = g + r;
                if (p < 0) {
                    r = -r;
                }
            } else {
                g = d[m];
                p = 0.0;
                r = 0.0;
            }
            
            if (m > l) {
                g = d[l] - (p + r);
                s = 1.0;
                c = 1.0;
                p = 0.0;
                
                for (i = l; i < m - 1; i++) {
                    f = s * e[i];
                    b = c * e[i];
                    if (fabs(f) >= fabs(g)) {
                        c = g / f;
                        r = sqrt(c * c + 1.0);
                        e[i] = f * r;
                        s = 1.0 / r;
                        c = c * s;
                    } else {
                        s = f / g;
                        r = sqrt(s * s + 1.0);
                        e[i] = g * r;
                        c = 1.0 / r;
                        s = s * c;
                    }
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i] = g + p;
                    g = c * r - b;
                    
                    /* Update eigenvectors if needed */
                    if (compz[0] != 'N') {
                        /* Apply Givens rotation to Z */
                        for (j = 0; j < *n; j++) {
                            f = Z[j + i * (*ldz)];
                            Z[j + i * (*ldz)] = c * f + s * Z[j + (i + 1) * (*ldz)];
                            Z[j + (i + 1) * (*ldz)] = -s * f + c * Z[j + (i + 1) * (*ldz)];
                        }
                    }
                }
                
                d[m - 1] = g;
                e[m - 2] = g - p;
            }
            
            /* Check convergence */
            if (e[m - 2] <= eps * fabs(d[m - 1])) {
                break;
            }
        }
    }
    
    /* Rescale if needed */
    if (doscale) {
        for (i = 0; i < *n; i++) {
            d[i] = d[i] * safety;
        }
    }
}
