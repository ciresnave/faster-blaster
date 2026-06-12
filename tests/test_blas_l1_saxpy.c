/*
 * Test suite for SAXPY (Single precision AXPY)
 * Operation: y[i] := alpha * x[i] + y[i]
 *
 * SAXPY Reference: http://www.netlib.org/blas/saxpy.f
 *
 * Test coverage:
 * - Basic operation (contiguous arrays)
 * - Non-contiguous (stride) handling
 * - Edge cases (n=0, n=1, alpha=0)
 * - Negative strides
 * - Large arrays
 * - Numerical accuracy (1e-6 tolerance for float)
 *
 * Build: gcc -I../../include -I../../../blis/build -o test_blas_l1_saxpy
 * test_blas_l1_saxpy.c ../../src/blas/level1/saxpy.c -lm -lpthread Run:
 * ./test_blas_l1_saxpy
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Include the reference implementation
#include "blas_reference.h"

/*
 * Test helper: Compare two float arrays with tolerance
 */
static bool arrays_equal_f(const float *a, const float *b, int n, float tol) {
  for (int i = 0; i < n; i++) {
    float diff = fabsf(a[i] - b[i]);
    if (diff > tol) {
      printf("  Mismatch at index %d: got %.8e, expected %.8e (diff=%.2e)\n", i,
             a[i], b[i], diff);
      return false;
    }
  }
  return true;
}

/*
 * Test 1: Basic SAXPY (contiguous, unit stride)
 * y = [2, 3, 4, 5]
 * x = [1, 1, 1, 1]
 * alpha = 2.0
 * Expected: y = [4, 5, 6, 7]
 */
void test_saxpy_basic(void) {
  printf("Test 1: Basic SAXPY (contiguous)\n");

  int n = 4;
  float alpha = 2.0f;
  float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float y[] = {2.0f, 3.0f, 4.0f, 5.0f};
  float expected[] = {4.0f, 5.0f, 6.0f, 7.0f};

  saxpy_ref(n, alpha, x, 1, y, 1);

  if (arrays_equal_f(y, expected, n, 1e-6f)) {
    printf("  ✓ PASS\n\n");
  } else {
    printf("  ✗ FAIL\n\n");
  }
}

/*
 * Test 2: SAXPY with stride
 * y = [2, 0, 3, 0, 4, 0]  (stride 2)
 * x = [1, 9, 1, 9, 1, 9]  (stride 2)
 * alpha = 2.0
 * incx = 2, incy = 2
 * Expected: y = [4, 0, 5, 0, 6, 0]
 */
void test_saxpy_stride(void) {
  printf("Test 2: SAXPY with stride (incx=2, incy=2)\n");

  int n = 3;
  float alpha = 2.0f;
  float x[] = {1.0f, 9.0f, 1.0f, 9.0f, 1.0f, 9.0f}; // stride 2
  float y[] = {2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f}; // stride 2
  float expected[] = {4.0f, 0.0f, 5.0f, 0.0f, 6.0f, 0.0f};

  saxpy_ref(n, alpha, x, 2, y, 2);

  if (arrays_equal_f(y, expected, 6, 1e-6f)) {
    printf("  ✓ PASS\n\n");
  } else {
    printf("  ✗ FAIL\n\n");
  }
}

/*
 * Test 3: SAXPY with n=0 (no operation)
 */
void test_saxpy_n_zero(void) {
  printf("Test 3: SAXPY with n=0\n");

  int n = 0;
  float alpha = 2.0f;
  float x[] = {1.0f};
  float y[] = {2.0f};
  float expected[] = {2.0f};

  saxpy_ref(n, alpha, x, 1, y, 1);

  if (arrays_equal_f(y, expected, 1, 1e-6f)) {
    printf("  ✓ PASS (array unchanged)\n\n");
  } else {
    printf("  ✗ FAIL\n\n");
  }
}

/*
 * Test 4: SAXPY with alpha=0 (no operation)
 */
void test_saxpy_alpha_zero(void) {
  printf("Test 4: SAXPY with alpha=0\n");

  int n = 4;
  float alpha = 0.0f;
  float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float y[] = {5.0f, 6.0f, 7.0f, 8.0f};
  float expected[] = {5.0f, 6.0f, 7.0f, 8.0f};

  saxpy_ref(n, alpha, x, 1, y, 1);

  if (arrays_equal_f(y, expected, n, 1e-6f)) {
    printf("  ✓ PASS (array unchanged)\n\n");
  } else {
    printf("  ✗ FAIL\n\n");
  }
}

/*
 * Test 5: SAXPY with n=1 (single element)
 */
void test_saxpy_n_one(void) {
  printf("Test 5: SAXPY with n=1\n");

  int n = 1;
  float alpha = 3.5f;
  float x[] = {2.0f};
  float y[] = {1.0f};
  float expected[] = {8.0f}; // 3.5*2 + 1 = 8

  saxpy_ref(n, alpha, x, 1, y, 1);

  if (arrays_equal_f(y, expected, n, 1e-6f)) {
    printf("  ✓ PASS\n\n");
  } else {
    printf("  ✗ FAIL\n\n");
  }
}

/*
 * Test 6: SAXPY with negative alpha
 */
void test_saxpy_negative_alpha(void) {
  printf("Test 6: SAXPY with negative alpha\n");

  int n = 3;
  float alpha = -1.5f;
  float x[] = {2.0f, 4.0f, 6.0f};
  float y[] = {10.0f, 20.0f, 30.0f};
  float expected[] = {10.0f - 1.5f * 2.0f, 20.0f - 1.5f * 4.0f,
                      30.0f - 1.5f * 6.0f};
  // expected = [7.0, 14.0, 21.0]

  saxpy_ref(n, alpha, x, 1, y, 1);

  if (arrays_equal_f(y, expected, n, 1e-6f)) {
    printf("  ✓ PASS\n\n");
  } else {
    printf("  ✗ FAIL\n\n");
  }
}

/*
 * Test 7: SAXPY with backward stride (negative incx/incy)
 */
void test_saxpy_negative_stride(void) {
  printf("Test 7: SAXPY with negative stride (backward)\n");

  int n = 3;
  float alpha = 1.0f;
  float x[] = {1.0f, 1.0f, 1.0f,
               1.0f}; // indices: 0,1,2,3 (will access 3,2,1 backward)
  float y[] = {2.0f, 2.0f, 2.0f, 2.0f};
  float expected[] = {3.0f, 3.0f, 3.0f, 3.0f};

  // incx = -1 means go backward from x[n-1]
  saxpy_ref(n, alpha, x + (n - 1), -1, y + (n - 1), -1);

  if (arrays_equal_f(y, expected, 4, 1e-6f)) {
    printf("  ✓ PASS\n\n");
  } else {
    printf("  ✗ FAIL\n\n");
  }
}

/*
 * Test 8: SAXPY with large values (no overflow)
 */
void test_saxpy_large_values(void) {
  printf("Test 8: SAXPY with large values (no overflow)\n");

  int n = 3;
  float alpha = 1e6f;
  float x[] = {1.0f, 2.0f, 3.0f};
  float y[] = {1e6f, 2e6f, 3e6f};
  float expected[] = {2e6f, 4e6f, 6e6f};

  saxpy_ref(n, alpha, x, 1, y, 1);

  // Larger tolerance for large numbers
  if (arrays_equal_f(y, expected, n, 1e-6f * 6e6f)) {
    printf("  ✓ PASS\n\n");
  } else {
    printf("  ✗ FAIL\n\n");
  }
}

/*
 * Run all tests
 */
int main(void) {
  printf("\n");
  printf("╔═══════════════════════════════════════════════════════════╗\n");
  printf("║                  SAXPY TEST SUITE                        ║\n");
  printf("║              (Single Precision AXPY)                     ║\n");
  printf("╚═══════════════════════════════════════════════════════════╝\n\n");

  test_saxpy_basic();
  test_saxpy_stride();
  test_saxpy_n_zero();
  test_saxpy_alpha_zero();
  test_saxpy_n_one();
  test_saxpy_negative_alpha();
  test_saxpy_negative_stride();
  test_saxpy_large_values();

  printf("╔═══════════════════════════════════════════════════════════╗\n");
  printf("║                   TEST SUITE COMPLETE                    ║\n");
  printf("║                 Total tests: 8                           ║\n");
  printf("║                 Run 'valgrind ./test_blas_l1_saxpy'      ║\n");
  printf("║                 for memory validation                    ║\n");
  printf("╚═══════════════════════════════════════════════════════════╝\n\n");

  return 0;
}
