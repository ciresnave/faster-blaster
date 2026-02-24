# Phase 2 Implementation - Start Guide

**Date**: January 22, 2026  
**Status**: Ready to Begin  
**Objective**: Implement 603 Tier 1 operations with comprehensive test coverage

---

## Quick Start Checklist (Complete in Order)

### Step 1: Audit Current State (Est. 4 hours)

**Goal**: Understand exactly what's implemented vs. stub vs. partial

**Action**:
```bash
# Search for stub patterns
find src -name "*.c" -type f | xargs grep -l "TODO\|NOT_IMPLEMENTED\|STUB\|not yet\|not implemented" > stubs.txt

# Count implementation status
find src/blas -name "*.c" | wc -l  # Total files
grep -r "TODO" src/blas | wc -l   # With TODOs

# Generate inventory of empty functions
cd faster-blaster-reference
grep -r "^void.*{$" src | head -20  # Empty function bodies
```

**Output**: Create `IMPLEMENTATION_STATUS_AUDIT.csv` with columns:
```
operation,file,status,completeness,priority
saxpy,src/blas/l1/saxpy.c,EMPTY,0,1
daxpy,src/blas/l1/daxpy.c,EMPTY,0,1
sgemm,src/blas/l3/sgemm.c,PARTIAL,30,1
sgesv,src/lapack/drivers/sgesv.c,COMPLETE,100,2
```

### Step 2: Set Up Test Framework (Est. 6 hours)

**Goal**: Have testing infrastructure ready before implementation starts

**Action 1**: Download Unity test framework
```bash
cd faster-blaster-reference/tests
git clone https://github.com/ThrowTheSwitch/Unity.git unity_framework
```

**Action 2**: Create CMakeLists.txt for tests
```cmake
# tests/CMakeLists.txt
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/unity_framework/src)

add_library(unity unity_framework/src/unity.c)

# BLAS L1 Tests
add_executable(test_blas_l1 test_blas_l1.c)
target_link_libraries(test_blas_l1 faster_blaster_ref unity)
add_test(NAME blas_l1 COMMAND test_blas_l1)

# Add more test executables...
```

**Action 3**: Create test template file
```bash
# tests/test_template_operation.c
cat > test_template.c << 'EOF'
#include <stdlib.h>
#
#include <complex.h>
#include "unity.h"
#include "faster-blaster.h"

// =====  OPERATION_NAME Tests =====
void test_OPERATION_basic() {
    // TODO: Implement basic test
}

void test_OPERATION_edge_cases() {
    // TODO: Implement edge case tests
}

void test_OPERATION_precision() {
    // TODO: Implement precision test
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_OPERATION_basic);
    RUN_TEST(test_OPERATION_edge_cases);
    RUN_TEST(test_OPERATION_precision);
    return UNITY_END();
}
EOF
```

### Step 3: Prepare Master Implementation Spreadsheet (Est. 2 hours)

**Goal**: Single source of truth for tracking all 603 Tier 1 operations

**Create spreadsheet with columns**:
1. **Operation**: saxpy, daxpy, caxpy, etc.
2. **File**: src/blas/l1/saxpy.c
3. **Category**: L1, L2, L3, Driver, Computational, Auxiliary
4. **Status**: QUEUED, IN_PROGRESS, REVIEW, TESTING, COMPLETE
5. **Impl %**: 0-100% completion
6. **Tests**: e.g., "3/5" (3 tests written, 5 total needed)
7. **Notes**: Any blockers or special considerations
8. **Assigned To**: Developer name
9. **Start Date**: When work began
10. **Estimated Finish**: Target date

**Populate with all 603 Tier 1 operations**:
- 54 BLAS L1
- 30 BLAS L2 (core)
- 4 BLAS L3 (GEMM only)
- 24 LAPACK Drivers
- 40 LAPACK Computational
- 10 LAPACK Auxiliary
- 3 Unified Operations

**Share**: Make read-only version available to all team members for visibility

### Step 4: Create Implementation Priority Sequence (Est. 3 hours)

**Goal**: Define exact order to implement (manages dependencies)

**Dependency Graph**:
```
Phase 1: BLAS L1 (54 ops) - INDEPENDENT
  ↓
Phase 2: BLAS L2 (30 ops) - needs L1
  ↓
Phase 3: BLAS L3 (4 ops - GEMM only) - needs L1/L2
  ↓
Phase 4: LAPACK Computational (40 ops) - needs BLAS L1/L2/L3
  ↓
Phase 5: LAPACK Drivers (24 ops) - needs Computational
  ↓
Phase 6: LAPACK Auxiliary (10 ops) - independent
  ↓
Phase 7: Unified Operations (3 ops) - needs L1/L2/L3
```

**Create**: `IMPLEMENTATION_SEQUENCE.txt` with exact order and estimated hours
```
Week 1-2: BLAS L1 Operations
  1. saxpy (4 hours: impl + 5 tests)
  2. daxpy (4 hours)
  3. caxpy (4 hours)
  4. zaxpy (4 hours)
  [... continue for all 54 L1 ops ...]

Week 3: BLAS L2 Core Operations
  55. sgemv (6 hours: more complex than L1)
  56. dgemv (6 hours)
  [... continue for 30 L2 ops ...]

[... and so on ...]
```

### Step 5: Establish Build & Test Commands (Est. 1 hour)

**Goal**: Simple commands for developers to build and test

**Create**: `build_and_test.sh` (or `.ps1` for Windows)
```bash
#!/bin/bash
# Tier 1 Build & Test

echo "🔨 Building faster-blaster-reference with C23..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_STANDARD=23
cmake --build build -j$(nproc)

echo "🧪 Running Tier 1 tests..."
cd build
ctest -C Release --verbose --output-on-failure

echo "📊 Generating test report..."
ctest -C Release --verbose > ../test_results_$(date +%Y%m%d_%H%M%S).txt

echo "✅ Done! Check test_results_*.txt for details"
```

**Windows PowerShell variant**:
```powershell
# build_and_test.ps1
echo "🔨 Building faster-blaster-reference with C23..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_STANDARD=23
cmake --build build -j$env:NUMBER_OF_PROCESSORS

echo "🧪 Running Tier 1 tests..."
cd build
ctest -C Release --verbose --output-on-failure

echo "✅ Done! Check test results above"
```

### Step 6: Team Planning Session (Est. 1 hour)

**Agenda**:
1. Review audit results - understand current state
2. Review master spreadsheet - confirm priorities
3. Review timeline - confirm feasibility
4. Assign developers to phases
5. Set weekly sync-up schedule
6. Confirm resource availability

**Required**: All developers, tech lead, QA engineer

### Step 7: Week 1 Kickoff - Start with SAXPY (Est. ongoing)

**Week 1 Plan**:
```
Day 1: Setup & Preparation (2 hours)
  - Pull latest code
  - Build system verified
  - Test framework running
  - saxpy.c opened in editor

Day 2-3: Implement SAXPY (6 hours)
  - Write implementation with full comments
  - Verify C23 compliance
  - Compile with no warnings
  
Day 4-5: Write & Debug Tests (6 hours)
  - Write 5+ test cases
  - All tests pass
  - Edge cases verified
  - Code review ready

Day 6: Code Review & Sign-Off (2 hours)
  - Tech lead reviews implementation
  - Feedback addressed
  - SAXPY marked COMPLETE
  - Move to DAXPY

Result: SAXPY fully implemented & tested ✅
```

---

## Implementation Template - Ready to Copy

### For Each Operation (SAXPY Example)

**Step 1: Create Implementation File**
```bash
# File: src/blas/l1/saxpy.c
cat > src/blas/l1/saxpy.c << 'EOF'
#include <stdlib.h>

/**
 * saxpy - Single precision A*X Plus Y
 * 
 * Computes: y := alpha*x + y
 * 
 * Parameters:
 *   n     - Number of elements in vectors
 *   alpha - Scalar multiplier for x
 *   x     - Input vector x
 *   incx  - Stride for x (e.g., 1 for contiguous)
 *   y     - Input/Output vector y (modified in-place)
 *   incy  - Stride for y
 * 
 * Notes:
 *   - Accepts both positive and negative strides
 *   - Handles n=0 gracefully (no-op)
 *   - Handles alpha=0 gracefully (optimized)
 */
void saxpy_(const int *n, const float *alpha, const float *x, const int *incx,
            float *y, const int *incy) {
    // 1. Input validation
    if (!n || !alpha || !x || !y || !incx || !incy) {
        return;
    }
    
    if (*n <= 0) {
        return;
    }
    
    // 2. Trivial case: alpha = 0
    if (*alpha == 0.0f) {
        return;  // y is unchanged
    }
    
    // 3. Compute starting indices for strides
    //    If stride is negative, start at end of array
    const int ix = (*incx > 0) ? 0 : (-(*n - 1) * *incx);
    const int iy = (*incy > 0) ? 0 : (-(*n - 1) * *incy);
    
    // 4. Main loop: y[i] += alpha * x[i]
    for (int i = 0; i < *n; ++i) {
        y[iy + i * *incy] += *alpha * x[ix + i * *incx];
    }
}
EOF
```

**Step 2: Create Test File**
```bash
# File: tests/test_blas_l1_saxpy.c
cat > tests/test_blas_l1_saxpy.c << 'EOF'
#include <stdlib.h>
#include <string.h>
#
#include "unity.h"

// External function being tested
extern void saxpy_(const int *n, const float *alpha, const float *x, const int *incx,
                   float *y, const int *incy);

// Test helper
#define ASSERT_FLOAT_NEAR(expected, actual, tolerance) \
    TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, actual)

// ===== Basic Functionality =====
void test_saxpy_basic_vectors() {
    int n = 5;
    int incx = 1, incy = 1;
    float alpha = 2.0f;
    
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float expected[] = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};
    
    saxpy_(&n, &alpha, x, &incx, y, &incy);
    
    for (int i = 0; i < n; ++i) {
        ASSERT_FLOAT_NEAR(expected[i], y[i], 1e-6f);
    }
}

// ===== Edge Cases =====
void test_saxpy_zero_alpha() {
    int n = 5;
    float alpha = 0.0f;
    
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y_original[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float y[5];
    memcpy(y, y_original, sizeof(y));
    
    saxpy_(&n, &alpha, x, &one, y, &one);
    
    // y should be unchanged
    for (int i = 0; i < n; ++i) {
        ASSERT_FLOAT_NEAR(y_original[i], y[i], 1e-6f);
    }
}

void test_saxpy_zero_n() {
    int n = 0;
    float alpha = 2.0f;
    
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {1.0f, 1.0f, 1.0f};
    float y_original[] = {1.0f, 1.0f, 1.0f};
    
    saxpy_(&n, &alpha, x, &one, y, &one);
    
    // y should be unchanged
    for (int i = 0; i < 3; ++i) {
        ASSERT_FLOAT_NEAR(y_original[i], y[i], 1e-6f);
    }
}

// ===== Stride Tests =====
void test_saxpy_with_strides() {
    int n = 3;
    int incx = 2, incy = 3;
    float alpha = 1.5f;
    
    // x array: [1.0, XX, 2.0, XX, 3.0, XX] (stride 2)
    float x[] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f, 99.0f};
    
    // y array: [1.0, XX, XX, 2.0, XX, XX, 3.0] (stride 3)
    float y[] = {1.0f, 99.0f, 99.0f, 2.0f, 99.0f, 99.0f, 3.0f};
    
    saxpy_(&n, &alpha, x, &incx, y, &incy);
    
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f + 1.5f*1.0f, y[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f + 1.5f*2.0f, y[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.0f + 1.5f*3.0f, y[6]);
}

void test_saxpy_negative_stride() {
    int n = 3;
    int incx = -1, incy = -1;
    float alpha = 1.0f;
    
    float x[] = {3.0f, 2.0f, 1.0f};
    float y[] = {3.0f, 2.0f, 1.0f};
    
    saxpy_(&n, &alpha, x, &incx, y, &incy);
    
    // With negative stride, access is reversed
    // y[2] += x[0], y[1] += x[1], y[0] += x[2]
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, y[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, y[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, y[0]);
}

// ===== Precision Tests =====
void test_saxpy_large_values() {
    int n = 3;
    int incx = 1, incy = 1;
    float alpha = 1e30f;
    
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {1e30f, 1e30f, 1e30f};
    
    saxpy_(&n, &alpha, x, &incx, y, &incy);
    
    // Should not overflow or NaN
    TEST_ASSERT_FALSE(isnan(y[0]));
    TEST_ASSERT_FALSE(isinf(y[0]));
}

void test_saxpy_small_values() {
    int n = 3;
    int incx = 1, incy = 1;
    float alpha = 1e-30f;
    
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {0.0f, 0.0f, 0.0f};
    
    saxpy_(&n, &alpha, x, &incx, y, &incy);
    
    // Should not underflow to zero (unless system doesn't support denormals)
    for (int i = 0; i < 3; ++i) {
        TEST_ASSERT_TRUE(y[i] >= 0.0f);
    }
}

// ===== Test Runner =====
int main() {
    UNITY_BEGIN();
    
    RUN_TEST(test_saxpy_basic_vectors);
    RUN_TEST(test_saxpy_zero_alpha);
    RUN_TEST(test_saxpy_zero_n);
    RUN_TEST(test_saxpy_with_strides);
    RUN_TEST(test_saxpy_negative_stride);
    RUN_TEST(test_saxpy_large_values);
    RUN_TEST(test_saxpy_small_values);
    
    return UNITY_END();
}
EOF
```

**Step 3: Build and Test**
```bash
cd build
cmake ..
make test_blas_l1_saxpy
./bin/test_blas_l1_saxpy

# Output should show:
# test_saxpy_basic_vectors PASSED
# test_saxpy_zero_alpha PASSED
# test_saxpy_zero_n PASSED
# test_saxpy_with_strides PASSED
# test_saxpy_negative_stride PASSED
# test_saxpy_large_values PASSED
# test_saxpy_small_values PASSED
# 
# ===== 7 tests passed =====
```

**Step 4: Update Master Spreadsheet**
```
saxpy | COMPLETE | 100% | 7/7 ✅
```

---

## Common Pitfalls to Avoid

### 1. Fortran vs CBLAS Calling Convention
❌ WRONG:
```c
void saxpy(int n, float alpha, float *x, int incx, float *y, int incy) {
    // WRONG: Fortran expects pass-by-reference
}
```

✅ CORRECT:
```c
void saxpy_(const int *n, const float *alpha, const float *x, const int *incx,
            float *y, const int *incy) {
    // All parameters passed by reference, names end in underscore
}
```

### 2. Stride Handling
❌ WRONG:
```c
for (int i = 0; i < *n; ++i) {
    y[i * *incy] += *alpha * x[i * *incx];  // Assumes starting at index 0
}
```

✅ CORRECT:
```c
const int ix = (*incx > 0) ? 0 : (-(*n - 1) * *incx);
const int iy = (*incy > 0) ? 0 : (-(*n - 1) * *incy);

for (int i = 0; i < *n; ++i) {
    y[iy + i * *incy] += *alpha * x[ix + i * *incx];
}
```

### 3. Memory Management
❌ WRONG:
```c
float *temp = malloc(...);
// Forgot to free - memory leak!
```

✅ CORRECT:
```c
float *temp = malloc(...);
// ... use temp ...
free(temp);
```

### 4. Numerical Precision
❌ WRONG:
```c
TEST_ASSERT_EQUAL_FLOAT(expected, actual);  // Exact equality - fails with rounding
```

✅ CORRECT:
```c
TEST_ASSERT_FLOAT_WITHIN(1e-6, expected, actual);  // Tolerance for rounding
```

---

## Weekly Tracking Template

**Copy this to your project weekly**:

```markdown
# Week 1 Progress Report

**Period**: Jan 22-26, 2026  
**Team**: [Names]

## Summary Stats
- Operations Started: 1 (SAXPY)
- Operations Completed: 1 (SAXPY)
- Tests Written: 7
- Tests Passing: 7/7 (100%)
- Blocker Count: 0
- On Schedule: ✅ YES

## Operations Completed This Week
- ✅ saxpy (7 tests: basic, zero_alpha, zero_n, strides, negative_stride, large_values, small_values)

## Operations In Progress
- 🟡 daxpy (30% - implementation 50% done, tests not started)

## Blockers
- None at this time

## Next Week Plan
1. Complete DAXPY (impl + 7 tests)
2. Start CAXPY (complex precision variant)
3. Update master spreadsheet

## Notes
- No major issues
- All tests passing
- Code review: pending tech lead approval
- Ready to move to next operation after review
```

---

## Final Confirmation

Before starting implementation, confirm:

- [ ] Audit complete - understand current state
- [ ] Test framework installed and working
- [ ] Master spreadsheet created and shared
- [ ] Build command verified (builds with no warnings)
- [ ] Test command verified (tests run with no errors)
- [ ] First developer assignment confirmed
- [ ] SAXPY implementation template understood
- [ ] Team kickoff meeting scheduled

**Ready to Begin**: Once all items above are ✅, start SAXPY implementation.

**Expected Timeline**:
- Weeks 1-2: BLAS L1 (54 operations, ~400 tests)
- Weeks 3-4: BLAS L2 (30 operations, ~200 tests)
- Weeks 5-6: LAPACK Core (64 operations, ~450 tests)
- Weeks 7-12: Remaining Tier 1 (455 operations, ~2,500 tests)

**Total Tier 1**: 603 operations, ~3,500 tests, 12 weeks

---

**Status**: ✅ Ready for Phase 2 Implementation  
**Next**: Confirm audit, test framework, and team assignments
