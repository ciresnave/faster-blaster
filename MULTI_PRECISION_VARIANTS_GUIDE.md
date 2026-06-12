# Multi-Precision Variants Guide

**Purpose**: Explain how BLAS operations scale across S/D/C/Z (single/double/complex) variants without duplicating code

**Key Insight**: While each variant (SAXPY, DAXPY, CAXPY, ZAXPY) has its own function pointer slot in the vtable, the implementation pattern is identical — only the data type changes.

---

## Pattern: From Single (S) to All Variants

### Example: SAXPY → DAXPY → CAXPY → ZAXPY

#### Step 1: Implement SAXPY (Single Precision)

**File**: `src/blas/l1/saxpy.c`

```c
#include <stdlib.h>

void saxpy_(const int *n, const float *alpha, const float *x, const int *incx,
            float *y, const int *incy) {
    if (!n || !alpha || !x || !y || !incx || !incy) return;
    if (*n <= 0) return;
    if (*alpha == 0.0f) return;
    
    const int ix = (*incx > 0) ? 0 : (-(*n - 1) * *incx);
    const int iy = (*incy > 0) ? 0 : (-(*n - 1) * *incy);
    
    for (int i = 0; i < *n; ++i) {
        y[iy + i * *incy] += *alpha * x[ix + i * *incx];
    }
}
```

**Tests**: 7 tests written (basic, zero_alpha, zero_n, strides, negative_stride, large_values, small_values)

---

#### Step 2: Implement DAXPY (Double Precision)

**File**: `src/blas/l1/daxpy.c`

```c
#include <stdlib.h>

void daxpy_(const int *n, const double *alpha, const double *x, const int *incx,
            double *y, const int *incy) {
    if (!n || !alpha || !x || !y || !incx || !incy) return;
    if (*n <= 0) return;
    if (*alpha == 0.0) return;  // Note: 0.0 (double) instead of 0.0f
    
    const int ix = (*incx > 0) ? 0 : (-(*n - 1) * *incx);
    const int iy = (*incy > 0) ? 0 : (-(*n - 1) * *incy);
    
    for (int i = 0; i < *n; ++i) {
        y[iy + i * *incy] += *alpha * x[ix + i * *incx];
    }
}
```

**Key Changes from SAXPY**:
1. `float` → `double` in parameter types
2. `0.0f` → `0.0` (double literal)
3. Function name: `saxpy_` → `daxpy_`
4. DAXPY tests identical to SAXPY (just different precision)

**Test Tolerance**: 1e-15 (double) instead of 1e-6 (single)

---

#### Step 3: Implement CAXPY (Complex Single Precision)

**File**: `src/blas/l1/caxpy.c`

```c
#include <stdlib.h>
#include <complex.h>

void caxpy_(const int *n, const float _Complex *alpha, const float _Complex *x, const int *incx,
            float _Complex *y, const int *incy) {
    if (!n || !alpha || !x || !y || !incx || !incy) return;
    if (*n <= 0) return;
    if (*alpha == 0.0f + 0.0f * I) return;  // Complex zero
    
    const int ix = (*incx > 0) ? 0 : (-(*n - 1) * *incx);
    const int iy = (*incy > 0) ? 0 : (-(*n - 1) * *incy);
    
    for (int i = 0; i < *n; ++i) {
        y[iy + i * *incy] += *alpha * x[ix + i * *incx];
    }
}
```

**Key Changes from SAXPY**:
1. `float` → `float _Complex` (C23 complex type)
2. Complex arithmetic handled by C compiler (a + b*I notation)
3. Zero check: `0.0f + 0.0f * I` (both real and imaginary parts zero)
4. All arithmetic (addition, multiplication) works naturally with complex types

**Test Tolerance**: 1e-6 for both real and imaginary parts

---

#### Step 4: Implement ZAXPY (Complex Double Precision)

**File**: `src/blas/l1/zaxpy.c`

```c
#include <stdlib.h>
#include <complex.h>

void zaxpy_(const int *n, const double _Complex *alpha, const double _Complex *x, const int *incx,
            double _Complex *y, const int *incy) {
    if (!n || !alpha || !x || !y || !incx || !incy) return;
    if (*n <= 0) return;
    if (*alpha == 0.0 + 0.0 * I) return;  // Complex zero (double)
    
    const int ix = (*incx > 0) ? 0 : (-(*n - 1) * *incx);
    const int iy = (*incy > 0) ? 0 : (-(*n - 1) * *incy);
    
    for (int i = 0; i < *n; ++i) {
        y[iy + i * *incy] += *alpha * x[ix + i * *incx];
    }
}
```

**Key Changes from CAXPY**:
1. `float _Complex` → `double _Complex`
2. `0.0f` → `0.0` (double literals)
3. Test tolerance: 1e-15 for real and imaginary parts

---

## Testing Pattern for Multi-Precision

### For SAXPY (template)
```c
void test_saxpy_basic_vectors() {
    int n = 5;
    int incx = 1, incy = 1;
    float alpha = 2.0f;
    
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float expected[] = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};
    
    saxpy_(&n, &alpha, x, &incx, y, &incy);
    
    for (int i = 0; i < n; ++i) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected[i], y[i]);
    }
}
```

### For DAXPY (copy test with tolerance change)
```c
void test_daxpy_basic_vectors() {
    int n = 5;
    int incx = 1, incy = 1;
    double alpha = 2.0;
    
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double expected[] = {3.0, 5.0, 7.0, 9.0, 11.0};
    
    daxpy_(&n, &alpha, x, &incx, y, &incy);
    
    for (int i = 0; i < n; ++i) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, expected[i], y[i]);  // Higher precision
    }
}
```

### For CAXPY (complex test)
```c
void test_caxpy_basic_vectors() {
    int n = 3;
    int incx = 1, incy = 1;
    float _Complex alpha = 1.0f + 1.0f * I;  // (1 + i)
    
    float _Complex x[] = {1.0f + 0.0f*I, 2.0f + 0.0f*I, 3.0f + 0.0f*I};
    float _Complex y[] = {1.0f + 0.0f*I, 1.0f + 0.0f*I, 1.0f + 0.0f*I};
    
    caxpy_(&n, &alpha, x, &incx, y, &incy);
    
    // y[0] = (1+i) + (1+i)*1 = (1+i) + (1+i) = (2+2i)
    // y[1] = (1+i) + (1+i)*2 = (1+i) + (2+2i) = (3+3i)
    // y[2] = (1+i) + (1+i)*3 = (1+i) + (3+3i) = (4+4i)
    
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, crealf(y[0]));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, cimagf(y[0]));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.0f, crealf(y[1]));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.0f, cimagf(y[1]));
}
```

### For ZAXPY (complex double test)
```c
void test_zaxpy_basic_vectors() {
    int n = 3;
    int incx = 1, incy = 1;
    double _Complex alpha = 1.0 + 1.0 * I;  // (1 + i)
    
    double _Complex x[] = {1.0 + 0.0*I, 2.0 + 0.0*I, 3.0 + 0.0*I};
    double _Complex y[] = {1.0 + 0.0*I, 1.0 + 0.0*I, 1.0 + 0.0*I};
    
    zaxpy_(&n, &alpha, x, &incx, y, &incy);
    
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 2.0, creal(y[0]));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 2.0, cimag(y[0]));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 3.0, creal(y[1]));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 3.0, cimag(y[1]));
}
```

---

## Key Typing Conversions Reference

### Scalar Types

| Operation          | Single (S) | Double (D) | Complex Single (C) | Complex Double (Z) |
| ------------------ | ---------- | ---------- | ------------------ | ------------------ |
| **Scalar Type**    | `float`    | `double`   | `float _Complex`   | `double _Complex`  |
| **Vector Type**    | `float*`   | `double*`  | `float _Complex*`  | `double _Complex*` |
| **Zero Value**     | `0.0f`     | `0.0`      | `0.0f + 0.0f*I`    | `0.0 + 0.0*I`      |
| **Test Tolerance** | `1e-6`     | `1e-15`    | `1e-6`             | `1e-15`            |
| **Unary**          | `fabsf()`  | `fabs()`   | `cabsf()`          | `cabs()`           |
| **Real Part**      | N/A        | N/A        | `crealf()`         | `creal()`          |
| **Imag Part**      | N/A        | N/A        | `cimagf()`         | `cimag()`          |

### Complex Arithmetic (C23)

```c
// Defining complex numbers
float _Complex z1 = 1.0f + 2.0f * I;
double _Complex z2 = 3.0 + 4.0 * I;

// Arithmetic operations (compiler handles automatically)
float _Complex sum = z1 + z2;           // Addition
float _Complex prod = z1 * z2;          // Multiplication (includes cross terms)
float _Complex conj_z1 = conjf(z1);     // Conjugate (single)
double _Complex conj_z2 = conj(z2);     // Conjugate (double)

// Extracting real/imaginary parts
float real_part = crealf(z1);
float imag_part = cimagf(z1);
double real_part2 = creal(z2);
double imag_part2 = cimag(z2);

// Magnitude/phase
float mag = cabsf(z1);      // Magnitude
double mag2 = cabs(z2);
float phase = cargf(z1);    // Phase angle
double phase2 = carg(z2);
```

---

## Implementation Workflow: SAXPY → DAXPY → CAXPY → ZAXPY

**Total Time**: ~20 hours for all 4 variants with tests

**Week 1 Breakdown**:

| Day | Task                                 | Time  | Deliverable                               |
| --- | ------------------------------------ | ----- | ----------------------------------------- |
| 1-2 | SAXPY implementation + 7 tests       | 8 hrs | SAXPY complete                            |
| 2-3 | DAXPY implementation + 7 tests       | 6 hrs | DAXPY complete (nearly identical code)    |
| 3-4 | CAXPY implementation + 7 tests       | 4 hrs | CAXPY complete (add complex arithmetic)   |
| 4-5 | ZAXPY implementation + 7 tests       | 4 hrs | ZAXPY complete (same pattern as CAXPY)    |
| 5   | Code review & sign-off               | 2 hrs | All 4 functions ready for next operations |
| 5-6 | Integration testing (all 4 together) | 2 hrs | Cross-validation of variants              |

**Result**: 4 core Level 1 operations complete with 28 tests (4 operations × 7 tests each)

---

## File Structure for Multi-Precision Families

**BLAS Level 1 Example**:
```
src/blas/l1/
├── saxpy.c      (float version - implement first)
├── daxpy.c      (double version - copy pattern, change types)
├── caxpy.c      (complex single - add complex.h, _Complex)
├── zaxpy.c      (complex double - same as complex single, double precision)
├── sscal.c
├── dscal.c
├── cscal.c
├── zscal.c
├── scopy.c
├── dcopy.c
├── ccopy.c
├── zcopy.c
└── ... (14 more operations × 4 precision variants)
```

**Test Files Corresponding**:
```
tests/
├── test_blas_l1_saxpy.c    (tests all 4 AXPY variants)
├── test_blas_l1_scal.c     (tests all 4 SCAL variants)
├── test_blas_l1_copy.c     (tests all 4 COPY variants)
└── ... (one test file per operation family)
```

---

## Quick Checklist: Each Multi-Precision Operation

✅ Implement S variant with full edge case handling
✅ Write 7 test cases for S variant
✅ Copy S → D variant (change types)
✅ Verify D variant compiles without warnings
✅ Run D tests (should pass with 1e-15 tolerance)
✅ Copy S → C variant (add `#include <complex.h>`, `float _Complex`)
✅ Handle complex zero check correctly
✅ Run C tests with complex arithmetic
✅ Copy C → Z variant (change `float _Complex` → `double _Complex`)
✅ Run Z tests
✅ Cross-validate all 4 in integration test
✅ Mark operation family COMPLETE

---

## Common Mistakes in Multi-Precision Implementation

### ❌ Mistake 1: Type Mismatch in Tests
```c
// WRONG - testing double with float literal
double y = 1.5;
TEST_ASSERT_DOUBLE_WITHIN(1e-15, 1.5f, y);  // 1.5f is float!
```

✅ **Fix**:
```c
TEST_ASSERT_DOUBLE_WITHIN(1e-15, 1.5, y);  // 1.5 is double
```

### ❌ Mistake 2: Complex Zero Comparison
```c
// WRONG - can't compare complex to 0.0 directly
if (*alpha == 0.0f) return;  // Won't work for complex!
```

✅ **Fix**:
```c
if (*alpha == 0.0f + 0.0f * I) return;  // Complex zero
// Or use magnitude check:
if (cabsf(*alpha) == 0.0f) return;
```

### ❌ Mistake 3: Missing Complex Header
```c
// WRONG - forgot #include <complex.h>
void caxpy_(const int *n, const float _Complex *alpha, ...) {
    // Will fail: _Complex not defined
}
```

✅ **Fix**:
```c
#include <complex.h>  // Must include before using _Complex
void caxpy_(const int *n, const float _Complex *alpha, ...) {
    // Now _Complex is defined
}
```

### ❌ Mistake 4: Wrong Complex Math Functions
```c
// WRONG - using float functions on double complex
double _Complex z = 3.0 + 4.0 * I;
float mag = cabsf(z);  // cabsf expects float _Complex!
```

✅ **Fix**:
```c
double _Complex z = 3.0 + 4.0 * I;
double mag = cabs(z);  // Use cabs() for double complex
```

---

## Implementation Ready Template

Save this template for each multi-precision family:

```c
// TEMPLATE: src/blas/l1/OPERATION_PREFIX.c
#include <stdlib.h>
#include <complex.h>  // Only needed for C/Z variants

/**
 * OPERATION_PREFIX - Brief description
 * Precision: [SINGLE|DOUBLE|COMPLEX_SINGLE|COMPLEX_DOUBLE]
 * 
 * Computes: operation description
 * 
 * Parameters: [from standard BLAS documentation]
 */
void OPERATION_PREFIX_(const int *n, const SCALAR_TYPE *alpha, const VECTOR_TYPE *x, const int *incx,
                        VECTOR_TYPE *y, const int *incy) {
    
    // 1. Input validation
    if (!n || !alpha || !x || !y || !incx || !incy) return;
    if (*n <= 0) return;
    if (*alpha == ZERO_VALUE) return;  // Optimization: zero scalar
    
    // 2. Handle strides
    const int ix = (*incx > 0) ? 0 : (-(*n - 1) * *incx);
    const int iy = (*incy > 0) ? 0 : (-(*n - 1) * *incy);
    
    // 3. Core computation
    for (int i = 0; i < *n; ++i) {
        // operation: y += alpha * x
    }
}
```

---

## Success Verification

Each multi-precision operation is complete when:

✅ All 4 variants (S/D/C/Z) implemented  
✅ 7 test cases each × 4 variants = 28 tests minimum  
✅ 100% test pass rate on all variants  
✅ Code review approved  
✅ Memory clean (Valgrind)  
✅ Cross-platform verified (Windows/Linux/macOS)  
✅ Spreadsheet updated to COMPLETE  

**Example Complete State**:
```
saxpy | COMPLETE | 100% | 7/7 ✅
daxpy | COMPLETE | 100% | 7/7 ✅
caxpy | COMPLETE | 100% | 7/7 ✅
zaxpy | COMPLETE | 100% | 7/7 ✅
```
