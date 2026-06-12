# ACTION ITEMS: Reference Backend Integration - What To Do Next

## 📋 Immediate Next Steps (Today)

### Step 1: Generate All 1248 Wrapper Functions
**Time: 2-3 hours | Priority: CRITICAL BLOCKER**

```bash
cd c:\Users\cires\OneDrive\Documents\projects\faster-blaster

# Run the wrapper generator with enhanced operation database
python scripts/generate_wrappers.py > /tmp/generated_wrappers.c

# Append to implementation file
# NOTE: Need to replace OPERATIONS list in generate_wrappers.py with full 1248 ops first
# For now, this generates sample for 14 operations
```

**What needs to happen:**
1. Update `scripts/generate_wrappers.py` with complete operation database (1248 operations)
2. Run script to generate all wrappers
3. Append generated code to `src/backends/blas_lapack_reference_backend.c`
4. Verify file compiles: `cl /c src/backends/blas_lapack_reference_backend.c`

**Expected result:** blas_lapack_reference_backend.c grows from 282 → ~13,300 lines

---

## 🔗 Step 2: Register Reference Backend in Plugin System
**Time: 1-2 hours | Priority: HIGH | Depends on: Step 1**

### Modify: `src/backends/backend_registry.c`

Add to `fb_init()`:
```c
#include "blas_lapack_reference_backend.h"

void fb_init() {
    /* ... existing backend registrations ... */
    
    /* Register reference backend (ADDED) */
    fb_backend_descriptor_t ref_desc = {
        .backend_name = "faster-blaster-reference",
        .init_fn = fb_blr_init,
        .finalize_fn = fb_blr_finalize,
        .get_vtable_fn = fb_blr_get_vtable,
        .get_info_fn = fb_blr_get_info,
        .check_available_fn = fb_blr_is_available,
        .get_score_fn = fb_blr_get_hardware_score,  /* NEW FUNCTION */
    };
    fb_register_backend(&ref_desc);
    
    /* ... rest of backend registrations ... */
}
```

### Add to `blas_lapack_reference_backend.c`:
```c
static int fb_blr_get_hardware_score(int hardware_type) {
    /* Reference backend is available on all hardware */
    /* Return maximum score (correctness priority) */
    return 255;  /* Always available, highest priority for correctness */
}
```

**Verification:**
```bash
# After building, verify reference backend loads:
./build/test_backends | grep "faster-blaster-reference"
# Expected: "faster-blaster-reference: AVAILABLE"
```

---

## 🧪 Step 3: Replace Correctness Test Stubs
**Time: 2-3 hours | Priority: HIGH | Depends on: Step 2**

### Modify: `tests/test_correctness_with_reference.c`

**Current state:** Inline mini-implementations (simplified)
**Target state:** Real reference backend DLL calls

**Example transformation:**

**Before (inline):**
```c
void test_saxpy_correctness(test_suite_t* suite, const char* backend_name, int size, float alpha) {
    // ... setup ...
    
    /* Inline reference implementation - WRONG */
    for (int i = 0; i < size; i++) {
        y_ref[i] = alpha * x[i] + y_ref[i];
    }
    
    /* Test backend */
    memcpy(y_test, y_ref, size * sizeof(float));
    
    // ... compare ...
}
```

**After (DLL-based):**
```c
void test_saxpy_correctness(test_suite_t* suite, const char* backend_name, int size, float alpha) {
    // ... setup ...
    
    /* Reference backend - CORRECT */
    fb_backend_vtable_t* ref_vtable = fb_blr_get_vtable();
    ref_vtable->saxpy(size, alpha, x, 1, y_ref, 1);
    
    /* Other backend */
    fb_backend_vtable_t* test_vtable = fb_get_backend_vtable(backend_name);
    test_vtable->saxpy(size, alpha, x, 1, y_test, 1);
    
    // ... compare ...
}
```

**Expand test coverage:**
```c
/* Add test functions for key operations: */
test_sdot_correctness(suite, backend, SMALL_SIZE);
test_sgemv_correctness(suite, backend, MEDIUM_SIZE);
test_sgemm_correctness(suite, backend, LARGE_SIZE);
test_sgesv_correctness(suite, backend, MEDIUM_SIZE);
test_dgesv_correctness(suite, backend, LARGE_SIZE);
/* ... 95+ more operations ... */

/* Test with multiple backends: */
for each operation in critical_operations:
    for each backend in ["reference", "openblas", "aocl-blis", "mkl"]:
        for each size in [16, 64, 256, 1024]:
            test_operation(suite, backend, size)
```

**Verification:**
```bash
cd build
./test_correctness_with_reference
# Expected: correctness_results_with_reference.json with ~500+ test results
```

---

## 🎯 Step 4: Implement Consensus Violation Detection
**Time: 2-3 hours | Priority: MEDIUM | Depends on: Step 3**

### Add to `tests/test_correctness_with_reference.c`:

```c
typedef struct {
    char operation_name[64];
    char* backend_set;  /* e.g., "aocl-blis, openblas, mkl" */
    double reference_result;
    double consensus_result;
    double error;
} consensus_violation_t;

void detect_consensus_violations(test_suite_t* suite, 
                                  consensus_violation_t** violations,
                                  size_t* violation_count) {
    /* For each operation/size combination: */
    for (size_t i = 0; i < suite->count; i++) {
        const correctness_result_t* ref_result = find_reference_result(suite, 
                                                   result->operation_name, 
                                                   result->size);
        
        /* Get results from all non-reference backends */
        double first_backend_error = -1;
        int all_agree = 1;
        char agreeing_backends[256] = {0};
        
        for (size_t j = 0; j < suite->count; j++) {
            if (strcmp(suite->results[j].operation_name, ref_result->operation_name) == 0
                && suite->results[j].size == ref_result->size
                && strcmp(suite->results[j].backend_name, "faster-blaster-reference") != 0) {
                
                if (first_backend_error < 0) {
                    first_backend_error = suite->results[j].max_error;
                } else if (abs(suite->results[j].max_error - first_backend_error) > 1e-10) {
                    all_agree = 0;
                    break;
                }
                strcat(agreeing_backends, suite->results[j].backend_name);
                strcat(agreeing_backends, ", ");
            }
        }
        
        /* If all non-reference backends agree but differ from reference */
        if (all_agree && first_backend_error != ref_result->max_error) {
            consensus_violation_t* viol = malloc(sizeof(consensus_violation_t));
            strcpy(viol->operation_name, ref_result->operation_name);
            viol->backend_set = strdup(agreeing_backends);
            viol->reference_result = ref_result->max_error;
            viol->consensus_result = first_backend_error;
            viol->error = abs(viol->consensus_result - viol->reference_result);
            
            /* Record violation */
            (*violations)[(*violation_count)++] = *viol;
        }
    }
}
```

**JSON Output Format:**
```json
{
    "consensus_violations": [
        {
            "operation": "sgesv",
            "size": 256,
            "dtype": "FP32",
            "backend_set": ["aocl-blis", "openblas", "mkl"],
            "reference_result": 1.5e-6,
            "consensus_result": 2.1e-6,
            "error": 6.0e-7,
            "potential_issue": "All backends have same bug or tolerance issue"
        }
    ],
    "summary": {
        "total_consensus_violations": 3,
        "investigation_required": true
    }
}
```

---

## 🗑️ Step 5: Delete Obsolete Files
**Time: 30 minutes | Priority: MEDIUM | Depends on: Step 3**

### Files to Delete:
```
src/backends/reference.c                    # Old, ~2000 lines, incomplete
src/backends/reference_complete.c          # Level 1 only
src/backends/reference_level1.c             # Incomplete
src/backends/reference_level2.c             # Incomplete
src/backends/reference_level3.c             # Incomplete
```

### Update `CMakeLists.txt`:
```cmake
# REMOVE these from SOURCES:
${CMAKE_CURRENT_SOURCE_DIR}/src/backends/reference.c
${CMAKE_CURRENT_SOURCE_DIR}/src/backends/reference_complete.c
${CMAKE_CURRENT_SOURCE_DIR}/src/backends/reference_level1.c
${CMAKE_CURRENT_SOURCE_DIR}/src/backends/reference_level2.c
${CMAKE_CURRENT_SOURCE_DIR}/src/backends/reference_level3.c

# ADD these to SOURCES:
${CMAKE_CURRENT_SOURCE_DIR}/src/backends/blas_lapack_reference_backend.c
```

### Verify no other references:
```bash
grep -r "reference\.c\|reference_complete\|reference_level" . --include="*.c" --include="*.h" --include="CMakeLists.txt"
# Should return no results
```

---

## ✅ Step 6: Full Build & Test
**Time: 1-2 hours | Priority: CRITICAL | Depends on: Step 5**

### Build:
```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
# Expected: 0 errors, 0 warnings
```

### Run Tests:
```bash
ctest --output-on-failure
# Expected: All tests pass, reference backend available
```

### Verify JSON Output:
```bash
cat correctness_results_with_reference.json | jq '.summary'
# Expected:
# {
#   "total": 1200+,
#   "passed": 1150+,
#   "failed": <10,
#   "unavailable": 0,
#   "consensus_violations": <5
# }
```

### Check for Consensus Violations:
```bash
cat correctness_results_with_reference.json | jq '.consensus_violations[] | select(.error > 1e-6)'
# Investigate any violations found
```

---

## 📊 Completion Checklist

- [ ] **Phase 1 (Complete):**
  - [x] Create blas_lapack_reference_backend.h
  - [x] Create blas_lapack_reference_backend.c
  - [x] Create test_correctness_with_reference.c
  - [x] Create generate_wrappers.py
  - [x] Create documentation files

- [ ] **Phase 2 (In Progress):**
  - [ ] Update generate_wrappers.py with full operation database
  - [ ] Run generator for all 1248 operations
  - [ ] Append wrappers to implementation file
  - [ ] Verify compilation

- [ ] **Phase 3 (Pending):**
  - [ ] Modify backend_registry.c
  - [ ] Add registration call in fb_init()
  - [ ] Implement hardware scoring function
  - [ ] Test backend discovery

- [ ] **Phase 4 (Pending):**
  - [ ] Replace inline stubs with DLL calls
  - [ ] Expand test coverage to 100+ operations
  - [ ] Verify multiple size classes
  - [ ] Generate JSON output

- [ ] **Phase 5 (Pending):**
  - [ ] Implement consensus detection algorithm
  - [ ] Add violation JSON output
  - [ ] Test with multiple backends
  - [ ] Document findings

- [ ] **Phase 6 (Pending):**
  - [ ] Delete obsolete reference.c files
  - [ ] Update CMakeLists.txt
  - [ ] Verify no dangling references
  - [ ] Full build test

---

## 📚 Documentation to Read

### Quick Overview (10 min):
- `REFERENCE_BACKEND_DELIVERY_SUMMARY.md` - What was delivered

### Architecture Deep Dive (30 min):
- `ARCHITECTURE_REFERENCE_INTEGRATION.txt` - Layer-by-layer breakdown

### Implementation Guide (1 hour):
- `REFERENCE_BACKEND_INTEGRATION.md` - Step-by-step integration

### Executive Summary (20 min):
- `INTEGRATION_SUMMARY.md` - High-level overview with timeline

---

## 🚨 Critical Dependencies

| Task                | Depends On | Blocker |
| ------------------- | ---------- | ------- |
| Phase 2 (Wrappers)  | Phase 1    | No      |
| Phase 3 (Registry)  | Phase 2    | YES     |
| Phase 4 (Tests)     | Phase 3    | YES     |
| Phase 5 (Consensus) | Phase 4    | NO      |
| Phase 6 (Cleanup)   | Phase 4    | NO      |

**Critical Path:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → (5 & 6 parallel)

---

## 🎯 Success Definition

Integration is **COMPLETE** when:

1. ✅ All 1248 operations callable from reference backend
2. ✅ Correctness test runs without crashes
3. ✅ Test results saved as valid JSON
4. ✅ Consensus violations logged and reviewed
5. ✅ Old reference.c files deleted
6. ✅ Clean build (0 errors, 0 warnings)
7. ✅ All tests pass in CTest suite

---

## 💡 Tips & Tricks

### To view what was created:
```bash
ls -lah src/backends/blas_lapack_reference_backend.*
head -50 tests/test_correctness_with_reference.c
cat REFERENCE_BACKEND_DELIVERY_SUMMARY.md | grep -A 5 "What Was Delivered"
```

### To understand the pattern:
```bash
grep -A 15 "^static void fb_blr_saxpy" src/backends/blas_lapack_reference_backend.c
# Study this pattern for all 1248 operations
```

### To test the generator:
```bash
python scripts/generate_wrappers.py | head -50
# Should show typedefs and wrappers
```

### To check DLL path:
```bash
# Set environment variable
set FB_REFERENCE_BACKEND_PATH=C:\path\to\dll

# Or check Windows PATH
where blas_lapack_reference.dll
```

---

## 📞 Support References

### If DLL not found:
→ See REFERENCE_BACKEND_INTEGRATION.md "Troubleshooting" section

### If functions won't compile:
→ Check pattern against fb_blr_saxpy (most common reference)

### If tests fail:
→ Check TOLERANCE_FP32 and TOLERANCE_FP64 values

### If consensus violations found:
→ Investigate with ARCHITECTURE_REFERENCE_INTEGRATION.txt "Consensus Violation Detection" section

---

## ⏱️ Time Estimate: ~10-11 hours total

- Phase 2: 2-3 hours
- Phase 3: 1-2 hours  
- Phase 4: 2-3 hours
- Phase 5: 2-3 hours
- Phase 6: 1.5-2 hours
- **Total: 10-13 hours**

---

## Next Immediate Action

**RIGHT NOW:** Run this to see what was created:
```bash
cd c:\Users\cires\OneDrive\Documents\projects\faster-blaster
ls src/backends/blas_lapack_reference_backend.*
ls tests/test_correctness_with_reference.c
ls scripts/generate_wrappers.py
cat REFERENCE_BACKEND_DELIVERY_SUMMARY.md | head -100
```

**Then:** Begin Phase 2 (Generate All 1248 Wrappers) following the steps above.
