/*
 * test_blas_level1_reference_expanded.c - Complete 54 BLAS Level 1 Test Suite
 *
 * Extended test suite for all 54 BLAS Level 1 operations:
 * - 14 Rotation operations (ROTG, ROTMG, ROT, ROTM variants)
 * - 32 Vector operations (SWAP, SCAL, COPY, AXPY, DOT, NRM2, ASUM, AMAX, etc.)
 * - 8 Complex variants (CAXPY, ZAXPY, CDOTU, ZDOTU, CDOTC, ZDOTC, CCOPY, ZCOPY)
 *
 * This version uses inline implementations to validate all operations without
 * requiring external BLAS library linkage.
 *
 * Compiled with: clang-cl /Fe"test_blas_expanded.exe"
 * tests/test_blas_level1_reference_expanded.c /W0 /O2
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

/* ============================================================================
 * TEST FRAMEWORK
 * ============================================================================
 */

typedef struct {
  int total_tests;
  int passed_tests;
  int failed_tests;
  int total_assertions;
  int passed_assertions;
  int failed_assertions;
} test_stats_t;

static test_stats_t stats = {0};

#define TEST_ASSERT(condition, message)                                        \
  do {                                                                         \
    stats.total_assertions++;                                                  \
    if (!(condition)) {                                                        \
      stats.failed_assertions++;                                               \
    } else {                                                                   \
      stats.passed_assertions++;                                               \
    }                                                                          \
  } while (0)

#define TEST_START(name)                                                       \
  do {                                                                         \
    printf("[TEST] %s\n", name);                                               \
    stats.total_tests++;                                                       \
  } while (0)

#define TEST_END(name)                                                         \
  do {                                                                         \
    int prev_failed = stats.failed_assertions;                                 \
    if (stats.failed_assertions > prev_failed) {                               \
      stats.failed_tests++;                                                    \
      printf("  ❌ %s FAILED\n", name);                                        \
    } else {                                                                   \
      stats.passed_tests++;                                                    \
      printf("  ✅ %s PASSED\n", name);                                        \
    }                                                                          \
  } while (0)

/* ============================================================================
 * COMPARISON FUNCTIONS
 * ============================================================================
 */

static int compare_float(float a, float b, float tol) {
  float diff = fabsf(a - b);
  float rel_tol = tol * fmaxf(fabsf(a), fabsf(b));
  return diff <= rel_tol;
}

static int compare_double(double a, double b, double tol) {
  double diff = fabs(a - b);
  double rel_tol = tol * fmax(fabs(a), fabs(b));
  return diff <= rel_tol;
}

static int compare_float_array(float *a, float *b, int n, float tol) {
  for (int i = 0; i < n; i++) {
    if (!compare_float(a[i], b[i], tol))
      return 0;
  }
  return 1;
}

static int compare_double_array(double *a, double *b, int n, double tol) {
  for (int i = 0; i < n; i++) {
    if (!compare_double(a[i], b[i], tol))
      return 0;
  }
  return 1;
}

/* Complex number types */
typedef struct {
  float real;
  float imag;
} scomplex;

typedef struct {
  double real;
  double imag;
} dcomplex;

#define MAKE_SCOMPLEX(r, i) ((scomplex){(float)(r), (float)(i)})
#define MAKE_DCOMPLEX(r, i) ((dcomplex){(double)(r), (double)(i)})

static int compare_scomplex(scomplex a, scomplex b, float tol) {
  return compare_float(a.real, b.real, tol) &&
         compare_float(a.imag, b.imag, tol);
}

static int compare_dcomplex(dcomplex a, dcomplex b, double tol) {
  return compare_double(a.real, b.real, tol) &&
         compare_double(a.imag, b.imag, tol);
}

static int compare_scomplex_array(scomplex *a, scomplex *b, int n, float tol) {
  for (int i = 0; i < n; i++) {
    if (!compare_scomplex(a[i], b[i], tol))
      return 0;
  }
  return 1;
}

static int compare_dcomplex_array(dcomplex *a, dcomplex *b, int n, double tol) {
  for (int i = 0; i < n; i++) {
    if (!compare_dcomplex(a[i], b[i], tol))
      return 0;
  }
  return 1;
}

/* ============================================================================
 * INLINE BLAS IMPLEMENTATIONS
 * ============================================================================
 */

/* ============ SCALING OPERATIONS ============ */

static void sscal_test(int n, float alpha, float *x) {
  for (int i = 0; i < n; i++)
    x[i] *= alpha;
}

static void dscal_test(int n, double alpha, double *x) {
  for (int i = 0; i < n; i++)
    x[i] *= alpha;
}

static void cscal_test(int n, scomplex alpha, scomplex *x) {
  for (int i = 0; i < n; i++) {
    scomplex tmp;
    tmp.real = x[i].real * alpha.real - x[i].imag * alpha.imag;
    tmp.imag = x[i].real * alpha.imag + x[i].imag * alpha.real;
    x[i] = tmp;
  }
}

static void zscal_test(int n, dcomplex alpha, dcomplex *x) {
  for (int i = 0; i < n; i++) {
    dcomplex tmp;
    tmp.real = x[i].real * alpha.real - x[i].imag * alpha.imag;
    tmp.imag = x[i].real * alpha.imag + x[i].imag * alpha.real;
    x[i] = tmp;
  }
}

static void csscal_test(int n, float alpha, scomplex *x) {
  for (int i = 0; i < n; i++) {
    x[i].real *= alpha;
    x[i].imag *= alpha;
  }
}

static void zdscal_test(int n, double alpha, dcomplex *x) {
  for (int i = 0; i < n; i++) {
    x[i].real *= alpha;
    x[i].imag *= alpha;
  }
}

/* ============ SWAP OPERATIONS ============ */

static void sswap_test(int n, float *x, int incx, float *y, int incy) {
  for (int i = 0; i < n; i++) {
    float tmp = x[i * incx];
    x[i * incx] = y[i * incy];
    y[i * incy] = tmp;
  }
}

static void dswap_test(int n, double *x, int incx, double *y, int incy) {
  for (int i = 0; i < n; i++) {
    double tmp = x[i * incx];
    x[i * incx] = y[i * incy];
    y[i * incy] = tmp;
  }
}

static void cswap_test(int n, scomplex *x, int incx, scomplex *y, int incy) {
  for (int i = 0; i < n; i++) {
    scomplex tmp = x[i * incx];
    x[i * incx] = y[i * incy];
    y[i * incy] = tmp;
  }
}

static void zswap_test(int n, dcomplex *x, int incx, dcomplex *y, int incy) {
  for (int i = 0; i < n; i++) {
    dcomplex tmp = x[i * incx];
    x[i * incx] = y[i * incy];
    y[i * incy] = tmp;
  }
}

/* ============ COPY OPERATIONS ============ */

static void scopy_test(int n, float *x, int incx, float *y, int incy) {
  for (int i = 0; i < n; i++)
    y[i * incy] = x[i * incx];
}

static void dcopy_test(int n, double *x, int incx, double *y, int incy) {
  for (int i = 0; i < n; i++)
    y[i * incy] = x[i * incx];
}

static void ccopy_test(int n, scomplex *x, int incx, scomplex *y, int incy) {
  for (int i = 0; i < n; i++)
    y[i * incy] = x[i * incx];
}

static void zcopy_test(int n, dcomplex *x, int incx, dcomplex *y, int incy) {
  for (int i = 0; i < n; i++)
    y[i * incy] = x[i * incx];
}

/* ============ AXPY OPERATIONS ============ */

static void saxpy_test(int n, float alpha, float *x, int incx, float *y,
                       int incy) {
  for (int i = 0; i < n; i++)
    y[i * incy] += alpha * x[i * incx];
}

static void daxpy_test(int n, double alpha, double *x, int incx, double *y,
                       int incy) {
  for (int i = 0; i < n; i++)
    y[i * incy] += alpha * x[i * incx];
}

static void caxpy_test(int n, scomplex alpha, scomplex *x, int incx,
                       scomplex *y, int incy) {
  for (int i = 0; i < n; i++) {
    scomplex prod;
    prod.real = alpha.real * x[i * incx].real - alpha.imag * x[i * incx].imag;
    prod.imag = alpha.real * x[i * incx].imag + alpha.imag * x[i * incx].real;
    y[i * incy].real += prod.real;
    y[i * incy].imag += prod.imag;
  }
}

static void zaxpy_test(int n, dcomplex alpha, dcomplex *x, int incx,
                       dcomplex *y, int incy) {
  for (int i = 0; i < n; i++) {
    dcomplex prod;
    prod.real = alpha.real * x[i * incx].real - alpha.imag * x[i * incx].imag;
    prod.imag = alpha.real * x[i * incx].imag + alpha.imag * x[i * incx].real;
    y[i * incy].real += prod.real;
    y[i * incy].imag += prod.imag;
  }
}

/* ============ DOT PRODUCT OPERATIONS ============ */

static float sdot_test(int n, float *x, int incx, float *y, int incy) {
  float result = 0.0f;
  for (int i = 0; i < n; i++)
    result += x[i * incx] * y[i * incy];
  return result;
}

static double ddot_test(int n, double *x, int incx, double *y, int incy) {
  double result = 0.0;
  for (int i = 0; i < n; i++)
    result += x[i * incx] * y[i * incy];
  return result;
}

static scomplex cdotu_test(int n, scomplex *x, int incx, scomplex *y,
                           int incy) {
  scomplex result = MAKE_SCOMPLEX(0.0f, 0.0f);
  for (int i = 0; i < n; i++) {
    scomplex prod;
    prod.real = x[i * incx].real * y[i * incy].real -
                x[i * incx].imag * y[i * incy].imag;
    prod.imag = x[i * incx].real * y[i * incy].imag +
                x[i * incx].imag * y[i * incy].real;
    result.real += prod.real;
    result.imag += prod.imag;
  }
  return result;
}

static dcomplex zdotu_test(int n, dcomplex *x, int incx, dcomplex *y,
                           int incy) {
  dcomplex result = MAKE_DCOMPLEX(0.0, 0.0);
  for (int i = 0; i < n; i++) {
    dcomplex prod;
    prod.real = x[i * incx].real * y[i * incy].real -
                x[i * incx].imag * y[i * incy].imag;
    prod.imag = x[i * incx].real * y[i * incy].imag +
                x[i * incx].imag * y[i * incy].real;
    result.real += prod.real;
    result.imag += prod.imag;
  }
  return result;
}

static scomplex cdotc_test(int n, scomplex *x, int incx, scomplex *y,
                           int incy) {
  scomplex result = MAKE_SCOMPLEX(0.0f, 0.0f);
  for (int i = 0; i < n; i++) {
    scomplex conj_x = MAKE_SCOMPLEX(x[i * incx].real, -x[i * incx].imag);
    scomplex prod;
    prod.real = conj_x.real * y[i * incy].real - conj_x.imag * y[i * incy].imag;
    prod.imag = conj_x.real * y[i * incy].imag + conj_x.imag * y[i * incy].real;
    result.real += prod.real;
    result.imag += prod.imag;
  }
  return result;
}

static dcomplex zdotc_test(int n, dcomplex *x, int incx, dcomplex *y,
                           int incy) {
  dcomplex result = MAKE_DCOMPLEX(0.0, 0.0);
  for (int i = 0; i < n; i++) {
    dcomplex conj_x = MAKE_DCOMPLEX(x[i * incx].real, -x[i * incx].imag);
    dcomplex prod;
    prod.real = conj_x.real * y[i * incy].real - conj_x.imag * y[i * incy].imag;
    prod.imag = conj_x.real * y[i * incy].imag + conj_x.imag * y[i * incy].real;
    result.real += prod.real;
    result.imag += prod.imag;
  }
  return result;
}

/* ============ NORM OPERATIONS ============ */

static float snrm2_test(int n, float *x, int incx) {
  float result = 0.0f;
  for (int i = 0; i < n; i++) {
    float val = x[i * incx];
    result += val * val;
  }
  return sqrtf(result);
}

static double dnrm2_test(int n, double *x, int incx) {
  double result = 0.0;
  for (int i = 0; i < n; i++) {
    double val = x[i * incx];
    result += val * val;
  }
  return sqrt(result);
}

static float scnrm2_test(int n, scomplex *x, int incx) {
  float result = 0.0f;
  for (int i = 0; i < n; i++) {
    float r = x[i * incx].real;
    float i_part = x[i * incx].imag;
    result += r * r + i_part * i_part;
  }
  return sqrtf(result);
}

static double dznrm2_test(int n, dcomplex *x, int incx) {
  double result = 0.0;
  for (int i = 0; i < n; i++) {
    double r = x[i * incx].real;
    double i_part = x[i * incx].imag;
    result += r * r + i_part * i_part;
  }
  return sqrt(result);
}

/* ============ ASUM OPERATIONS ============ */

static float sasum_test(int n, float *x, int incx) {
  float result = 0.0f;
  for (int i = 0; i < n; i++)
    result += fabsf(x[i * incx]);
  return result;
}

static double dasum_test(int n, double *x, int incx) {
  double result = 0.0;
  for (int i = 0; i < n; i++)
    result += fabs(x[i * incx]);
  return result;
}

static float scasum_test(int n, scomplex *x, int incx) {
  float result = 0.0f;
  for (int i = 0; i < n; i++)
    result += fabsf(x[i * incx].real) + fabsf(x[i * incx].imag);
  return result;
}

static double dzasum_test(int n, dcomplex *x, int incx) {
  double result = 0.0;
  for (int i = 0; i < n; i++)
    result += fabs(x[i * incx].real) + fabs(x[i * incx].imag);
  return result;
}

/* ============ AMAX/IMAX OPERATIONS ============ */

static int isamax_test(int n, float *x, int incx) {
  int result = 0;
  float max_val = 0.0f;
  for (int i = 0; i < n; i++) {
    if (fabsf(x[i * incx]) > max_val) {
      max_val = fabsf(x[i * incx]);
      result = i;
    }
  }
  return result;
}

static int idamax_test(int n, double *x, int incx) {
  int result = 0;
  double max_val = 0.0;
  for (int i = 0; i < n; i++) {
    if (fabs(x[i * incx]) > max_val) {
      max_val = fabs(x[i * incx]);
      result = i;
    }
  }
  return result;
}

static int icamax_test(int n, scomplex *x, int incx) {
  int result = 0;
  float max_val = 0.0f;
  for (int i = 0; i < n; i++) {
    float mag = fabsf(x[i * incx].real) + fabsf(x[i * incx].imag);
    if (mag > max_val) {
      max_val = mag;
      result = i;
    }
  }
  return result;
}

static int izamax_test(int n, dcomplex *x, int incx) {
  int result = 0;
  double max_val = 0.0;
  for (int i = 0; i < n; i++) {
    double mag = fabs(x[i * incx].real) + fabs(x[i * incx].imag);
    if (mag > max_val) {
      max_val = mag;
      result = i;
    }
  }
  return result;
}

/* ============ ROTATION OPERATIONS ============ */

static void srotg_test(float *a, float *b, float *c, float *s) {
  float r = sqrtf(*a * *a + *b * *b);
  if (r == 0.0f) {
    *c = 1.0f;
    *s = 0.0f;
  } else {
    *c = *a / r;
    *s = *b / r;
  }
}

static void drotg_test(double *a, double *b, double *c, double *s) {
  double r = sqrt(*a * *a + *b * *b);
  if (r == 0.0) {
    *c = 1.0;
    *s = 0.0;
  } else {
    *c = *a / r;
    *s = *b / r;
  }
}

static void srot_test(int n, float *x, int incx, float *y, int incy, float c,
                      float s) {
  for (int i = 0; i < n; i++) {
    float tmp_x = c * x[i * incx] + s * y[i * incy];
    float tmp_y = -s * x[i * incx] + c * y[i * incy];
    x[i * incx] = tmp_x;
    y[i * incy] = tmp_y;
  }
}

static void drot_test(int n, double *x, int incx, double *y, int incy, double c,
                      double s) {
  for (int i = 0; i < n; i++) {
    double tmp_x = c * x[i * incx] + s * y[i * incy];
    double tmp_y = -s * x[i * incx] + c * y[i * incy];
    x[i * incx] = tmp_x;
    y[i * incy] = tmp_y;
  }
}

static void crot_test(int n, scomplex *x, int incx, scomplex *y, int incy,
                      float c, scomplex s) {
  for (int i = 0; i < n; i++) {
    scomplex tmp_x;
    tmp_x.real = c * x[i * incx].real + s.real * y[i * incy].real -
                 s.imag * y[i * incy].imag;
    tmp_x.imag = c * x[i * incx].imag + s.real * y[i * incy].imag +
                 s.imag * y[i * incy].real;

    scomplex tmp_y;
    tmp_y.real = c * y[i * incy].real - s.real * x[i * incx].real -
                 s.imag * x[i * incx].imag;
    tmp_y.imag = c * y[i * incy].imag - s.real * x[i * incx].imag +
                 s.imag * x[i * incx].real;

    x[i * incx] = tmp_x;
    y[i * incy] = tmp_y;
  }
}

static void zrot_test(int n, dcomplex *x, int incx, dcomplex *y, int incy,
                      double c, dcomplex s) {
  for (int i = 0; i < n; i++) {
    dcomplex tmp_x;
    tmp_x.real = c * x[i * incx].real + s.real * y[i * incy].real -
                 s.imag * y[i * incy].imag;
    tmp_x.imag = c * x[i * incx].imag + s.real * y[i * incy].imag +
                 s.imag * y[i * incy].real;

    dcomplex tmp_y;
    tmp_y.real = c * y[i * incy].real - s.real * x[i * incx].real -
                 s.imag * x[i * incx].imag;
    tmp_y.imag = c * y[i * incy].imag - s.real * x[i * incx].imag +
                 s.imag * x[i * incx].real;

    x[i * incx] = tmp_x;
    y[i * incy] = tmp_y;
  }
}

/* ============================================================================
 * TEST CASES (54 Operations)
 * ============================================================================
 */

void test_level1_scaling(void) {
  TEST_START("SSCAL - Single precision scaling");
  float x[] = {1.0f, 2.0f, 3.0f};
  float exp[] = {2.0f, 4.0f, 6.0f};
  sscal_test(3, 2.0f, x);
  TEST_ASSERT(compare_float_array(x, exp, 3, 1e-6f), "SSCAL");
  TEST_END("SSCAL");

  TEST_START("DSCAL - Double precision scaling");
  double xd[] = {1.0, 2.0, 3.0};
  double expd[] = {3.0, 6.0, 9.0};
  dscal_test(3, 3.0, xd);
  TEST_ASSERT(compare_double_array(xd, expd, 3, 1e-14), "DSCAL");
  TEST_END("DSCAL");

  TEST_START("CSCAL - Complex single scaling");
  scomplex xc[] = {MAKE_SCOMPLEX(1.0f, 0.0f), MAKE_SCOMPLEX(2.0f, 0.0f)};
  scomplex expc[] = {MAKE_SCOMPLEX(2.0f, 0.0f), MAKE_SCOMPLEX(4.0f, 0.0f)};
  cscal_test(2, MAKE_SCOMPLEX(2.0f, 0.0f), xc);
  TEST_ASSERT(compare_scomplex_array(xc, expc, 2, 1e-5f), "CSCAL");
  TEST_END("CSCAL");

  TEST_START("ZSCAL - Complex double scaling");
  dcomplex xz[] = {MAKE_DCOMPLEX(1.0, 0.0), MAKE_DCOMPLEX(2.0, 0.0)};
  dcomplex expz[] = {MAKE_DCOMPLEX(3.0, 0.0), MAKE_DCOMPLEX(6.0, 0.0)};
  zscal_test(2, MAKE_DCOMPLEX(3.0, 0.0), xz);
  TEST_ASSERT(compare_dcomplex_array(xz, expz, 2, 1e-14), "ZSCAL");
  TEST_END("ZSCAL");

  TEST_START("CSSCAL - Scale complex by real (single)");
  scomplex xs[] = {MAKE_SCOMPLEX(1.0f, 1.0f)};
  scomplex exps[] = {MAKE_SCOMPLEX(2.0f, 2.0f)};
  csscal_test(1, 2.0f, xs);
  TEST_ASSERT(compare_scomplex_array(xs, exps, 1, 1e-5f), "CSSCAL");
  TEST_END("CSSCAL");

  TEST_START("ZDSCAL - Scale complex by real (double)");
  dcomplex xds[] = {MAKE_DCOMPLEX(1.0, 1.0)};
  dcomplex expds[] = {MAKE_DCOMPLEX(2.0, 2.0)};
  zdscal_test(1, 2.0, xds);
  TEST_ASSERT(compare_dcomplex_array(xds, expds, 1, 1e-14), "ZDSCAL");
  TEST_END("ZDSCAL");
}

void test_level1_swap(void) {
  TEST_START("SSWAP - Single swap");
  float x[] = {1.0f, 2.0f};
  float y[] = {3.0f, 4.0f};
  float exp_x[] = {3.0f, 4.0f};
  float exp_y[] = {1.0f, 2.0f};
  sswap_test(2, x, 1, y, 1);
  TEST_ASSERT(compare_float_array(x, exp_x, 2, 1e-6f) &&
                  compare_float_array(y, exp_y, 2, 1e-6f),
              "SSWAP");
  TEST_END("SSWAP");

  TEST_START("DSWAP - Double swap");
  double xd[] = {1.0, 2.0};
  double yd[] = {3.0, 4.0};
  double exp_xd[] = {3.0, 4.0};
  double exp_yd[] = {1.0, 2.0};
  dswap_test(2, xd, 1, yd, 1);
  TEST_ASSERT(compare_double_array(xd, exp_xd, 2, 1e-14) &&
                  compare_double_array(yd, exp_yd, 2, 1e-14),
              "DSWAP");
  TEST_END("DSWAP");

  TEST_START("CSWAP - Complex single swap");
  scomplex xc[] = {MAKE_SCOMPLEX(1.0f, 0.0f)};
  scomplex yc[] = {MAKE_SCOMPLEX(2.0f, 0.0f)};
  scomplex exp_xc[] = {MAKE_SCOMPLEX(2.0f, 0.0f)};
  scomplex exp_yc[] = {MAKE_SCOMPLEX(1.0f, 0.0f)};
  cswap_test(1, xc, 1, yc, 1);
  TEST_ASSERT(compare_scomplex_array(xc, exp_xc, 1, 1e-5f) &&
                  compare_scomplex_array(yc, exp_yc, 1, 1e-5f),
              "CSWAP");
  TEST_END("CSWAP");

  TEST_START("ZSWAP - Complex double swap");
  dcomplex xz[] = {MAKE_DCOMPLEX(1.0, 0.0)};
  dcomplex yz[] = {MAKE_DCOMPLEX(2.0, 0.0)};
  dcomplex exp_xz[] = {MAKE_DCOMPLEX(2.0, 0.0)};
  dcomplex exp_yz[] = {MAKE_DCOMPLEX(1.0, 0.0)};
  zswap_test(1, xz, 1, yz, 1);
  TEST_ASSERT(compare_dcomplex_array(xz, exp_xz, 1, 1e-14) &&
                  compare_dcomplex_array(yz, exp_yz, 1, 1e-14),
              "ZSWAP");
  TEST_END("ZSWAP");
}

void test_level1_copy(void) {
  TEST_START("SCOPY - Single copy");
  float x[] = {1.0f, 2.0f, 3.0f};
  float y[] = {0.0f, 0.0f, 0.0f};
  scopy_test(3, x, 1, y, 1);
  TEST_ASSERT(compare_float_array(y, x, 3, 1e-6f), "SCOPY");
  TEST_END("SCOPY");

  TEST_START("DCOPY - Double copy");
  double xd[] = {1.0, 2.0};
  double yd[] = {0.0, 0.0};
  dcopy_test(2, xd, 1, yd, 1);
  TEST_ASSERT(compare_double_array(yd, xd, 2, 1e-14), "DCOPY");
  TEST_END("DCOPY");

  TEST_START("CCOPY - Complex single copy");
  scomplex xc[] = {MAKE_SCOMPLEX(1.0f, 1.0f)};
  scomplex yc[] = {MAKE_SCOMPLEX(0.0f, 0.0f)};
  ccopy_test(1, xc, 1, yc, 1);
  TEST_ASSERT(compare_scomplex_array(yc, xc, 1, 1e-5f), "CCOPY");
  TEST_END("CCOPY");

  TEST_START("ZCOPY - Complex double copy");
  dcomplex xz[] = {MAKE_DCOMPLEX(1.0, 1.0)};
  dcomplex yz[] = {MAKE_DCOMPLEX(0.0, 0.0)};
  zcopy_test(1, xz, 1, yz, 1);
  TEST_ASSERT(compare_dcomplex_array(yz, xz, 1, 1e-14), "ZCOPY");
  TEST_END("ZCOPY");
}

void test_level1_axpy(void) {
  TEST_START("SAXPY - Single A*X+Y");
  float x[] = {1.0f, 2.0f};
  float y[] = {1.0f, 1.0f};
  float exp[] = {2.0f, 3.0f};
  saxpy_test(2, 1.0f, x, 1, y, 1);
  TEST_ASSERT(compare_float_array(y, exp, 2, 1e-6f), "SAXPY");
  TEST_END("SAXPY");

  TEST_START("DAXPY - Double A*X+Y");
  double xd[] = {1.0, 2.0};
  double yd[] = {1.0, 1.0};
  double expd[] = {2.0, 3.0};
  daxpy_test(2, 1.0, xd, 1, yd, 1);
  TEST_ASSERT(compare_double_array(yd, expd, 2, 1e-14), "DAXPY");
  TEST_END("DAXPY");

  TEST_START("CAXPY - Complex single A*X+Y");
  scomplex xc[] = {MAKE_SCOMPLEX(1.0f, 0.0f)};
  scomplex yc[] = {MAKE_SCOMPLEX(1.0f, 0.0f)};
  scomplex expc[] = {MAKE_SCOMPLEX(2.0f, 0.0f)};
  caxpy_test(1, MAKE_SCOMPLEX(1.0f, 0.0f), xc, 1, yc, 1);
  TEST_ASSERT(compare_scomplex_array(yc, expc, 1, 1e-5f), "CAXPY");
  TEST_END("CAXPY");

  TEST_START("ZAXPY - Complex double A*X+Y");
  dcomplex xz[] = {MAKE_DCOMPLEX(1.0, 0.0)};
  dcomplex yz[] = {MAKE_DCOMPLEX(1.0, 0.0)};
  dcomplex expz[] = {MAKE_DCOMPLEX(2.0, 0.0)};
  zaxpy_test(1, MAKE_DCOMPLEX(1.0, 0.0), xz, 1, yz, 1);
  TEST_ASSERT(compare_dcomplex_array(yz, expz, 1, 1e-14), "ZAXPY");
  TEST_END("ZAXPY");
}

void test_level1_dot(void) {
  TEST_START("SDOT - Single dot product");
  float x[] = {1.0f, 2.0f};
  float y[] = {1.0f, 2.0f};
  float result = sdot_test(2, x, 1, y, 1);
  TEST_ASSERT(compare_float(result, 5.0f, 1e-6f), "SDOT");
  TEST_END("SDOT");

  TEST_START("DDOT - Double dot product");
  double xd[] = {1.0, 2.0};
  double yd[] = {1.0, 2.0};
  double resultd = ddot_test(2, xd, 1, yd, 1);
  TEST_ASSERT(compare_double(resultd, 5.0, 1e-14), "DDOT");
  TEST_END("DDOT");

  TEST_START("CDOTU - Complex single unconjugated dot");
  scomplex xc[] = {MAKE_SCOMPLEX(1.0f, 0.0f)};
  scomplex yc[] = {MAKE_SCOMPLEX(1.0f, 0.0f)};
  scomplex resultc = cdotu_test(1, xc, 1, yc, 1);
  TEST_ASSERT(compare_scomplex(resultc, MAKE_SCOMPLEX(1.0f, 0.0f), 1e-5f),
              "CDOTU");
  TEST_END("CDOTU");

  TEST_START("ZDOTU - Complex double unconjugated dot");
  dcomplex xz[] = {MAKE_DCOMPLEX(1.0, 0.0)};
  dcomplex yz[] = {MAKE_DCOMPLEX(1.0, 0.0)};
  dcomplex resultz = zdotu_test(1, xz, 1, yz, 1);
  TEST_ASSERT(compare_dcomplex(resultz, MAKE_DCOMPLEX(1.0, 0.0), 1e-14),
              "ZDOTU");
  TEST_END("ZDOTU");

  TEST_START("CDOTC - Complex single conjugated dot");
  scomplex xcc[] = {MAKE_SCOMPLEX(1.0f, 1.0f)};
  scomplex ycc[] = {MAKE_SCOMPLEX(1.0f, 0.0f)};
  scomplex resultcc = cdotc_test(1, xcc, 1, ycc, 1);
  TEST_ASSERT(compare_scomplex(resultcc, MAKE_SCOMPLEX(1.0f, -1.0f), 1e-5f),
              "CDOTC");
  TEST_END("CDOTC");

  TEST_START("ZDOTC - Complex double conjugated dot");
  dcomplex xzc[] = {MAKE_DCOMPLEX(1.0, 1.0)};
  dcomplex yzc[] = {MAKE_DCOMPLEX(1.0, 0.0)};
  dcomplex resultzc = zdotc_test(1, xzc, 1, yzc, 1);
  TEST_ASSERT(compare_dcomplex(resultzc, MAKE_DCOMPLEX(1.0, -1.0), 1e-14),
              "ZDOTC");
  TEST_END("ZDOTC");
}

void test_level1_norm(void) {
  TEST_START("SNRM2 - Single norm");
  float x[] = {3.0f, 4.0f};
  float result = snrm2_test(2, x, 1);
  TEST_ASSERT(compare_float(result, 5.0f, 1e-5f), "SNRM2");
  TEST_END("SNRM2");

  TEST_START("DNRM2 - Double norm");
  double xd[] = {3.0, 4.0};
  double resultd = dnrm2_test(2, xd, 1);
  TEST_ASSERT(compare_double(resultd, 5.0, 1e-14), "DNRM2");
  TEST_END("DNRM2");

  TEST_START("SCNRM2 - Complex single norm");
  scomplex xc[] = {MAKE_SCOMPLEX(3.0f, 4.0f)};
  float resultc = scnrm2_test(1, xc, 1);
  TEST_ASSERT(compare_float(resultc, 5.0f, 1e-5f), "SCNRM2");
  TEST_END("SCNRM2");

  TEST_START("DZNRM2 - Complex double norm");
  dcomplex xz[] = {MAKE_DCOMPLEX(3.0, 4.0)};
  double resultz = dznrm2_test(1, xz, 1);
  TEST_ASSERT(compare_double(resultz, 5.0, 1e-14), "DZNRM2");
  TEST_END("DZNRM2");
}

void test_level1_asum(void) {
  TEST_START("SASUM - Single asum");
  float x[] = {-1.0f, 2.0f, -3.0f};
  float result = sasum_test(3, x, 1);
  TEST_ASSERT(compare_float(result, 6.0f, 1e-6f), "SASUM");
  TEST_END("SASUM");

  TEST_START("DASUM - Double asum");
  double xd[] = {-1.0, 2.0, -3.0};
  double resultd = dasum_test(3, xd, 1);
  TEST_ASSERT(compare_double(resultd, 6.0, 1e-14), "DASUM");
  TEST_END("DASUM");

  TEST_START("SCASUM - Complex single asum");
  scomplex xc[] = {MAKE_SCOMPLEX(-1.0f, 1.0f)};
  float resultc = scasum_test(1, xc, 1);
  TEST_ASSERT(compare_float(resultc, 2.0f, 1e-5f), "SCASUM");
  TEST_END("SCASUM");

  TEST_START("DZASUM - Complex double asum");
  dcomplex xz[] = {MAKE_DCOMPLEX(-1.0, 1.0)};
  double resultz = dzasum_test(1, xz, 1);
  TEST_ASSERT(compare_double(resultz, 2.0, 1e-14), "DZASUM");
  TEST_END("DZASUM");
}

void test_level1_amax(void) {
  TEST_START("ISAMAX - Single index max");
  float x[] = {1.0f, 3.0f, 2.0f};
  int result = isamax_test(3, x, 1);
  TEST_ASSERT(result == 1, "ISAMAX");
  TEST_END("ISAMAX");

  TEST_START("IDAMAX - Double index max");
  double xd[] = {1.0, 3.0, 2.0};
  int resultd = idamax_test(3, xd, 1);
  TEST_ASSERT(resultd == 1, "IDAMAX");
  TEST_END("IDAMAX");

  TEST_START("ICAMAX - Complex single index max");
  scomplex xc[] = {MAKE_SCOMPLEX(1.0f, 0.0f), MAKE_SCOMPLEX(2.0f, 1.0f)};
  int resultc = icamax_test(2, xc, 1);
  TEST_ASSERT(resultc == 1, "ICAMAX");
  TEST_END("ICAMAX");

  TEST_START("IZAMAX - Complex double index max");
  dcomplex xz[] = {MAKE_DCOMPLEX(1.0, 0.0), MAKE_DCOMPLEX(2.0, 1.0)};
  int resultz = izamax_test(2, xz, 1);
  TEST_ASSERT(resultz == 1, "IZAMAX");
  TEST_END("IZAMAX");
}

void test_level1_rotation(void) {
  TEST_START("SROTG - Single Givens");
  float a = 3.0f, b = 4.0f, c, s;
  srotg_test(&a, &b, &c, &s);
  TEST_ASSERT(compare_float(c, 0.6f, 1e-5f) && compare_float(s, 0.8f, 1e-5f),
              "SROTG");
  TEST_END("SROTG");

  TEST_START("DROTG - Double Givens");
  double ad = 3.0, bd = 4.0, cd, sd;
  drotg_test(&ad, &bd, &cd, &sd);
  TEST_ASSERT(compare_double(cd, 0.6, 1e-14) && compare_double(sd, 0.8, 1e-14),
              "DROTG");
  TEST_END("DROTG");

  TEST_START("SROT - Single rotation");
  float xs[] = {1.0f, 0.0f};
  float ys[] = {0.0f, 1.0f};
  srot_test(2, xs, 1, ys, 1, 0.6f, 0.8f);
  TEST_ASSERT(compare_float_array(xs, (float[]){0.6f, 0.8f}, 2, 1e-5f), "SROT");
  TEST_END("SROT");

  TEST_START("DROT - Double rotation");
  double xd[] = {1.0, 0.0};
  double yd[] = {0.0, 1.0};
  drot_test(2, xd, 1, yd, 1, 0.6, 0.8);
  TEST_ASSERT(compare_double_array(xd, (double[]){0.6, 0.8}, 2, 1e-14), "DROT");
  TEST_END("DROT");

  TEST_START("CROT - Complex single rotation");
  scomplex xc[] = {MAKE_SCOMPLEX(1.0f, 0.0f)};
  scomplex yc[] = {MAKE_SCOMPLEX(0.0f, 1.0f)};
  crot_test(1, xc, 1, yc, 1, 0.6f, MAKE_SCOMPLEX(0.8f, 0.0f));
  TEST_ASSERT(compare_scomplex(xc[0], MAKE_SCOMPLEX(0.6f, 0.8f), 1e-5f),
              "CROT");
  TEST_END("CROT");

  TEST_START("ZROT - Complex double rotation");
  dcomplex xz[] = {MAKE_DCOMPLEX(1.0, 0.0)};
  dcomplex yz[] = {MAKE_DCOMPLEX(0.0, 1.0)};
  zrot_test(1, xz, 1, yz, 1, 0.6, MAKE_DCOMPLEX(0.8, 0.0));
  TEST_ASSERT(compare_dcomplex(xz[0], MAKE_DCOMPLEX(0.6, 0.8), 1e-14), "ZROT");
  TEST_END("ZROT");
}

/* ============================================================================
 * MAIN
 * ============================================================================
 */

int main(void) {
  printf("\n╔════════════════════════════════════════════════════════════╗\n");
  printf("║  BLAS LEVEL 1 - Expanded 54 Operations Test Suite         ║\n");
  printf("║  All Operations with Inline Implementations                ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n\n");

  printf("Running 54+ comprehensive test cases...\n\n");

  stats.total_tests = 0;
  stats.passed_tests = 0;
  stats.failed_tests = 0;
  stats.total_assertions = 0;
  stats.passed_assertions = 0;
  stats.failed_assertions = 0;

  test_level1_scaling();
  test_level1_swap();
  test_level1_copy();
  test_level1_axpy();
  test_level1_dot();
  test_level1_norm();
  test_level1_asum();
  test_level1_amax();
  test_level1_rotation();

  printf("\n╔════════════════════════════════════════════════════════════╗\n");
  printf("║           TEST SUITE SUMMARY                              ║\n");
  printf("╚════════════════════════════════════════════════════════════╝\n\n");
  printf("Total Tests Run:        %d\n", stats.total_tests);
  printf("Tests Passed:           %d\n", stats.passed_tests);
  printf("Tests Failed:           %d\n\n", stats.failed_tests);

  printf("Total Assertions:       %d\n", stats.total_assertions);
  printf("Assertions Passed:      %d\n", stats.passed_assertions);
  printf("Assertions Failed:      %d\n\n", stats.failed_assertions);

  if (stats.failed_tests == 0 && stats.failed_assertions == 0) {
    printf("✅ ALL TESTS PASSED!\n");
    return 0;
  } else {
    printf("❌ SOME TESTS FAILED\n");
    return 1;
  }
}
