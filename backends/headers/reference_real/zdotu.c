/**
 * @file zdotu.c
 * @brief Reference implementation of ZDOTU (Complex double precision dot product, unconjugated)
 *
 * ZDOTU: sum = x^T * y (complex double, no conjugation)
 */

#include <blas_reference.h>

typedef struct {
    double real;
    double imag;
} complex_d;

/**
 * Complex double precision multiplication
 */
static inline complex_d cmul_d(complex_d a, complex_d b) {
    return (complex_d){
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

/**
 * Complex double precision dot product (unconjugated)
 */
complex_d zdotu_ref(int n, const complex_d *x, int incx, const complex_d *y, int incy) {
    complex_d result = {0.0, 0.0};
    if (n <= 0) return result;
    
    complex_d sum = {0.0, 0.0};
    complex_d corr_r = {0.0, 0.0};
    complex_d corr_i = {0.0, 0.0};
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            complex_d prod = cmul_d(x[i], y[i]);
            
            double real_corr = prod.real - corr_r.real;
            double real_t = sum.real + real_corr;
            corr_r.real = (real_t - sum.real) - real_corr;
            sum.real = real_t;
            
            double imag_corr = prod.imag - corr_i.real;
            double imag_t = sum.imag + imag_corr;
            corr_i.real = (imag_t - sum.imag) - imag_corr;
            sum.imag = imag_t;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            complex_d prod = cmul_d(x[ix], y[iy]);
            
            double real_corr = prod.real - corr_r.real;
            double real_t = sum.real + real_corr;
            corr_r.real = (real_t - sum.real) - real_corr;
            sum.real = real_t;
            
            double imag_corr = prod.imag - corr_i.real;
            double imag_t = sum.imag + imag_corr;
            corr_i.real = (imag_t - sum.imag) - imag_corr;
            sum.imag = imag_t;
            
            ix += incx;
            iy += incy;
        }
    }
    
    return sum;
}
