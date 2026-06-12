/**
 * @file test_oxiblas_backend.c
 * @brief Test suite for OxiBLAS CPU backend
 *
 * Tests the OxiBLAS backend implementation.
 * Pure-Rust BLAS/LAPACK library with SIMD kernels across all CPU architectures.
 *
 * OxiBLAS exports Fortran-convention (saxpy_, dgemm_, …) symbols via its
 * C-ABI oxiblas-ffi crate.  The backend adapter wraps these with CBLAS call
 * signatures for use by faster-blaster's vtable dispatch.
 *
 * @copyright Copyright (c) 2025
 * @license   MIT OR Apache-2.0
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "../src/backends/oxiblas_backend.h"

#define TOLERANCE_SINGLE 1e-5f
#define TOLERANCE_DOUBLE 1e-12
#define TEST_PASSED "✅ PASSED"
#define TEST_FAILED "❌ FAILED"

static int tests_passed = 0;
static int tests_failed = 0;

static bool float_equal(float a, float b, float tol) {
  return fabsf(a - b) <= tol;
}

static void report_test(const char *test_name, bool passed) {
  if (passed) {
    printf("  %s %s\n", TEST_PASSED, test_name);
    tests_passed++;
  } else {
    printf("  %s %s\n", TEST_FAILED, test_name);
    tests_failed++;
  }
}

/* =========================================================================
 * Test 1: Availability
 * ======================================================================== */
static void test_oxiblas_availability(void) {
  printf("\n📋 Test 1: OxiBLAS Availability\n");
  printf("═══════════════════════════════════════\n");

  bool available = fb_oxiblas_is_available();
  printf("  OxiBLAS library available: %s\n", available ? "Yes" : "No");

  if (!available) {
    printf("  ⚠️  OxiBLAS not found. Build from source with:\n");
    printf("      cargo build --release -p oxiblas-ffi\n");
    printf("      (requires Rust 1.85+ — https://git.io/oxiblas)\n");
    report_test("OxiBLAS availability check", false);
    return;
  }

  const char *version = fb_oxiblas_get_version();
  if (version) {
    printf("  Version: %s\n", version);
  }

  report_test("OxiBLAS availability check", available);
}

/* =========================================================================
 * Test 2: SAXPY  (y = alpha*x + y)
 * ======================================================================== */
static void test_oxiblas_saxpy(void) {
  printf("\n📋 Test 2: SAXPY (y = alpha*x + y)\n");
  printf("═══════════════════════════════════════\n");

  if (!fb_oxiblas_is_available()) {
    printf("  ⚠️  OxiBLAS not available, skipping\n");
    report_test("SAXPY operation", false);
    return;
  }

  if (fb_oxiblas_init() != 0) {
    printf("  ❌ fb_oxiblas_init() failed\n");
    report_test("SAXPY operation", false);
    return;
  }

  const int n = 5;
  float alpha = 2.0f;
  float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float y[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  float expected[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

  const fb_backend_vtable_t *vtable = fb_oxiblas_get_vtable();
  if (!vtable || !vtable->saxpy) {
    printf("  ⚠️  SAXPY not available in vtable\n");
    report_test("SAXPY operation", false);
    return;
  }

  vtable->saxpy(n, alpha, x, 1, y, 1);

  bool passed = true;
  for (int i = 0; i < n; i++) {
    if (!float_equal(y[i], expected[i], TOLERANCE_SINGLE)) {
      printf("  ❌ y[%d] = %.6f, expected %.6f\n", i, y[i], expected[i]);
      passed = false;
    }
  }

  if (passed) {
    printf("  All values correct: [7, 8, 9, 10, 11]\n");
  }

  report_test("SAXPY operation", passed);
}

/* =========================================================================
 * Test 3: DAXPY (double precision)
 * ======================================================================== */
static void test_oxiblas_daxpy(void) {
  printf("\n📋 Test 3: DAXPY (y = alpha*x + y, double)\n");
  printf("═══════════════════════════════════════\n");

  if (!fb_oxiblas_is_available()) {
    printf("  ⚠️  OxiBLAS not available, skipping\n");
    report_test("DAXPY operation", false);
    return;
  }

  const fb_backend_vtable_t *vtable = fb_oxiblas_get_vtable();
  if (!vtable || !vtable->daxpy) {
    printf("  ⚠️  DAXPY not available in vtable\n");
    report_test("DAXPY operation", false);
    return;
  }

  const int n = 4;
  double alpha = 3.0;
  double x[] = {1.0, 2.0, 3.0, 4.0};
  double y[] = {4.0, 3.0, 2.0, 1.0};
  double expected[] = {7.0, 9.0, 11.0, 13.0};

  vtable->daxpy(n, alpha, x, 1, y, 1);

  bool passed = true;
  for (int i = 0; i < n; i++) {
    if (fabs(y[i] - expected[i]) > TOLERANCE_DOUBLE) {
      printf("  ❌ y[%d] = %.15g, expected %.15g\n", i, y[i], expected[i]);
      passed = false;
    }
  }

  if (passed) {
    printf("  All values correct: [7, 9, 11, 13]\n");
  }

  report_test("DAXPY operation", passed);
}

/* =========================================================================
 * main
 * ======================================================================== */
int main(void) {
  printf("╔═══════════════════════════════════════════════════════════╗\n");
  printf("║       OxiBLAS CPU Backend Test Suite                     ║\n");
  printf("║       Pure-Rust BLAS/LAPACK with SIMD kernels             ║\n");
  printf("╚═══════════════════════════════════════════════════════════╝\n");

  test_oxiblas_availability();
  test_oxiblas_saxpy();
  test_oxiblas_daxpy();

  /* Clean up */
  fb_oxiblas_shutdown();

  printf("\n╔═══════════════════════════════════════════════════════════╗\n");
  printf("║                   Test Summary                            ║\n");
  printf("╠═══════════════════════════════════════════════════════════╣\n");
  printf("║  Total Tests: %-3d                                         ║\n",
         tests_passed + tests_failed);
  printf("║  Passed:      %-3d                                         ║\n",
         tests_passed);
  printf("║  Failed:      %-3d                                         ║\n",
         tests_failed);
  printf("╚═══════════════════════════════════════════════════════════╝\n");

  return (tests_failed > 0) ? 1 : 0;
}
