# Developer Quick Reference - Phase 2 Implementation

**Print this or bookmark it!** Quick access to all essential information for implementing Tier 1 operations.

---

## 📋 The 7-Step Process (From PHASE2_START_GUIDE.md)

```
Step 1: Audit current stubs     (run audit_stubs_*.sh)
Step 2: Set up test framework   (cmake -B build && cmake --build build)
Step 3: Import tracking sheet   (MASTER_IMPLEMENTATION_SPREADSHEET_TIER1.csv)
Step 4: Create sequence plan    (dependency ordering)
Step 5: Establish commands      (build_and_test.sh / .ps1)
Step 6: Team kickoff meeting    (1 hour planning)
Step 7: Begin implementation    (SAXPY family, Week 1)
```

---

## 🚀 Quick Start (Week 1)

### For Developers

**Step 1: Get the template**
```c
// From MULTI_PRECISION_VARIANTS_GUIDE.md - Copy this structure:
void saxpy_(const int *n, const float *alpha, const float *x, const int *incx,
            float *y, const int *incy) {
    // 1. Input validation
    if (!n || !alpha || !x || !y || !incx || !incy) return;
    if (*n <= 0) return;
    if (*alpha == 0.0f) return;  // Fast path
    
    // 2. Calculate strides
    const int ix = (*incx > 0) ? 0 : (-(*n - 1) * *incx);
    const int iy = (*incy > 0) ? 0 : (-(*n - 1) * *incy);
    
    // 3. Implement core operation
    for (int i = 0; i < *n; ++i) {
        y[iy + i * *incy] += *alpha * x[ix + i * *incx];
    }
}
```

**Step 2: Implement S variant** (SAXPY with float)
- Copy template above
- Test with 7 test cases
- Time: 8 hours

**Step 3: Create D variant** (DAXPY with double)
- Copy SAXPY → DAXPY
- Change: `float` → `double`, `0.0f` → `0.0`, `saxpy_` → `daxpy_`
- Test tolerance: 1e-15 (not 1e-6)
- Time: 6 hours

**Step 4: Create C variant** (CAXPY with complex single)
- Add: `#include <complex.h>`
- Change: `float` → `float _Complex`
- Update zero check: `*alpha == 0.0f + 0.0f * I`
- Compiler handles complex math automatically
- Time: 4 hours

**Step 5: Create Z variant** (ZAXPY with complex double)
- Copy CAXPY → ZAXPY
- Change: `float _Complex` → `double _Complex`, `0.0f` → `0.0`
- Test tolerance: 1e-15 complex
- Time: 4 hours

**Total Week 1**: 22 hours (4 operations with 28 test cases)

---

## 🧪 Testing Quick Reference

### Test Structure (Use this for every operation)

```c
#include "unity.h"
#include "../reference_implementation.h"  // Include operation being tested
#
#include <complex.h>

// Define tolerance for this operation
#define TOLERANCE_S 1e-6   // Single precision
#define TOLERANCE_D 1e-15  // Double precision
#define TOLERANCE_C 1e-6   // Complex single
#define TOLERANCE_Z 1e-15  // Complex double

void setUp(void) {
    // Called before each test - allocate test data
}

void tearDown(void) {
    // Called after each test - free test data
}

void test_saxpy_basic_vectors(void) {
    // Arrange
    int n = 5;
    float alpha = 2.0f;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float expected[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    
    // Act
    saxpy_(&n, &alpha, x, &(int){1}, y, &(int){1});
    
    // Assert
    for (int i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(TOLERANCE_S, expected[i], y[i]);
    }
}

void test_saxpy_with_stride(void) {
    // Test with stride != 1
    int n = 3;
    float alpha = 1.0f;
    float x[] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f};  // stride 2
    float y[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float expected[] = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f};
    int incx = 2, incy = 2;
    
    saxpy_(&n, &alpha, x, &incx, y, &incy);
    
    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(TOLERANCE_S, expected[i], y[i]);
    }
}

void test_saxpy_zero_alpha(void) {
    // Alpha = 0 should not modify y
    int n = 5;
    float alpha = 0.0f;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float expected[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    
    saxpy_(&n, &alpha, x, &(int){1}, y, &(int){1});
    
    for (int i = 0; i < n; i++) {
        TEST_ASSERT_FLOAT_WITHIN(TOLERANCE_S, expected[i], y[i]);
    }
}

void test_caxpy_complex_single(void) {
    // Test complex arithmetic
    int n = 2;
    float _Complex alpha = 1.0f + 1.0f * I;
    float _Complex x[] = {1.0f + 0.0f * I, 0.0f + 1.0f * I};
    float _Complex y[] = {0.0f + 0.0f * I, 0.0f + 0.0f * I};
    
    caxpy_(&n, &alpha, x, &(int){1}, y, &(int){1});
    
    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE_C, crealf(alpha), crealf(y[0]));
    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE_C, cimagf(alpha), cimagf(y[0]));
}

// ... 3-4 more test cases per operation ...
```

### Test Compilation & Execution

```bash
# Build with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

# Run all tests
cd build && ctest --verbose

# Run single test
ctest -R test_saxpy_basic -V

# Run with Valgrind (memory check)
valgrind --leak-check=full ./bin/test_blas_l1_saxpy
```

---

## 🔍 Multi-Precision Checklist

Before marking operation COMPLETE in spreadsheet, verify:

### Single Precision (S) - float
- [ ] Function compiles without warnings
- [ ] Input validation (n <= 0, NULL pointers)
- [ ] Zero alpha optimization
- [ ] Stride handling (positive, negative)
- [ ] 7 test cases all pass
- [ ] Valgrind memory clean
- [ ] Tolerance: 1e-6

### Double Precision (D) - double
- [ ] Copy S variant (float→double, 0.0f→0.0)
- [ ] Function compiles without warnings
- [ ] All 7 test cases pass (double copy)
- [ ] Valgrind memory clean
- [ ] Tolerance: 1e-15

### Complex Single (C) - float _Complex
- [ ] Add `#include <complex.h>`
- [ ] Zero check: `*alpha == 0.0f + 0.0f * I`
- [ ] Compiler handles complex multiply/add automatically
- [ ] 7 test cases with complex arithmetic
- [ ] Use crealf/cimagf for component access
- [ ] Tolerance: 1e-6 per component

### Complex Double (Z) - double _Complex
- [ ] Copy C variant (float _Complex→double _Complex)
- [ ] Zero check: `*alpha == 0.0 + 0.0 * I`
- [ ] All 7 test cases pass (double complex)
- [ ] Use creal/cimag for component access
- [ ] Tolerance: 1e-15 per component

### Integration Test
- [ ] Test all 4 variants together
- [ ] Cross-validate results
- [ ] Mark operation family COMPLETE
- [ ] Update MASTER_IMPLEMENTATION_SPREADSHEET_TIER1.csv

---

## 🛠️ Essential Commands

### Build Test Framework
```bash
# First time setup
cd faster-blaster-reference/tests
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j4

# Run tests
cd build && ctest --verbose
```

### Compile Single Operation Test
```bash
# Windows (MSVC)
cl /std:c23 /W4 test_blas_l1_saxpy.c saxpy.c /link unity.lib

# Linux/macOS (GCC/Clang)
gcc -std=c23 -Wall -Wextra test_blas_l1_saxpy.c saxpy.c -lunity -o test_saxpy
./test_saxpy

# With Valgrind
valgrind --leak-check=full ./test_saxpy
```

### Update Tracking Spreadsheet
1. Open MASTER_IMPLEMENTATION_SPREADSHEET_TIER1.csv in Excel/Sheets
2. Find operation row (e.g., SAXPY)
3. Update columns:
   - status: QUEUED → IN_PROGRESS → TESTING → VERIFICATION → COMPLETE
   - impl_percent: 0 → 50 → 75 → 100
   - tests_written: Count of tests created
   - tests_passing: Count of passing tests
   - assigned_to: Your name
   - start_date: YYYY-MM-DD when you started
   - estimated_finish: YYYY-MM-DD target date

---

## 📚 File Structure

```
faster-blaster-reference/
  src/
    blas/
      l1/
        saxpy.c      ← Implement here
        daxpy.c      ← S variant copy
        caxpy.c      ← Complex single variant
        zaxpy.c      ← Complex double variant
  tests/
    test_blas_l1_saxpy.c  ← Tests for all 4 variants
    CMakeLists.txt        ← Test build config
```

---

## ⚠️ Common Mistakes (Fix These!)

### ❌ WRONG: Mixing types in complex test
```c
float _Complex z = 1.0 + 2.0 * I;  // BAD: 1.0 is double
```
### ✅ CORRECT: Consistent types
```c
float _Complex z = 1.0f + 2.0f * I;  // GOOD: 1.0f is float
```

---

### ❌ WRONG: Complex zero check
```c
if (*alpha == 0.0) return;  // BAD: compares magnitude only
```
### ✅ CORRECT: Complex zero check
```c
if (*alpha == 0.0f + 0.0f * I) return;  // GOOD: checks both parts
```

---

### ❌ WRONG: Missing complex.h
```c
void caxpy_(...) {
    float _Complex z = 1.0f + 2.0f * I;  // BAD: I not defined
}
```
### ✅ CORRECT: Include complex.h first
```c
#include <complex.h>
void caxpy_(...) {
    float _Complex z = 1.0f + 2.0f * I;  // GOOD: I is defined
}
```

---

### ❌ WRONG: Wrong tolerance for precision
```c
void test_daxpy_(...) {
    TEST_ASSERT_FLOAT_WITHIN(1e-6, expected, actual);  // BAD: 1e-6 for double!
}
```
### ✅ CORRECT: Appropriate tolerance per precision
```c
void test_daxpy_(...) {
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, expected, actual);  // GOOD: 1e-15 for double
}
```

---

### ❌ WRONG: Forgetting complex component extraction
```c
TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, expected, result);  // BAD: comparing complex as float
```
### ✅ CORRECT: Extract components for complex
```c
TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, crealf(expected), crealf(result));
TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, cimagf(expected), cimagf(result));
```

---

## 📞 Getting Help

1. **Stuck on implementation?** → Review MULTI_PRECISION_VARIANTS_GUIDE.md
2. **Stuck on testing?** → Copy test template above
3. **Stuck on build?** → Run `cmake -B build && cmake --build build`
4. **Stuck on complex math?** → Compiler handles it, just use `*alpha * x[i]`
5. **Stuck on tolerance?** → Use 1e-6 (S/C), 1e-15 (D/Z)

---

## ✅ Done! Mark as Complete

When all steps above pass:
1. Update MASTER_IMPLEMENTATION_SPREADSHEET_TIER1.csv
   - Set status to: COMPLETE
   - Set impl_percent to: 100
   - Set tests_written to: 8 (4 ops × 2 variants)
   - Set tests_passing to: 8
2. Commit code to git
3. Request code review from Tech Lead
4. Move to next operation family (SCAL, COPY, DOT, etc.)

---

**Happy implementing! 🚀**

*Questions? Check PHASE2_START_GUIDE.md or MULTI_PRECISION_VARIANTS_GUIDE.md*
