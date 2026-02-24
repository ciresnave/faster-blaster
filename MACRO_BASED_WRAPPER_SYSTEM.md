# Macro-Based Wrapper Generation System

## Overview

Instead of writing 1248 wrapper functions by hand (or even generating them with Python), we use **C preprocessor macros** to define all operations once and expand them as needed. This is:

- **Much cleaner** - operations defined in one place
- **More maintainable** - any change applies to all wrappers
- **Faster to generate** - macro expansion is instantaneous
- **Compiler-verified** - syntax errors caught at compile time
- **Easy to customize** - change macro behavior without touching operation list

## Architecture

### Problem with Previous Approach

```python
# generate_wrappers.py produced:
for each operation:
    generate typedef
    generate wrapper function
    generate vtable entry
# Result: ~13,000 lines of repetitive code
```

### Solution: Macro-Based System

```c
// operation_registry.h defines all 1248 operations:
#ifdef REGISTER_BLAS_LEVEL1
OP(saxpy, void, saxpy_, (int n, float alpha, ...))
OP(daxpy, void, daxpy_, (int n, double alpha, ...))
// ... 54 more Level 1 operations
#endif

// User includes and defines OP macro:
#define OP(name, ret, fort, params) \
    typedef ret (*fblas_##name##_t)params;
#include "operation_registry.h"
#undef OP

// Result: All 1248 typedefs generated
```

## Usage Examples

### Example 1: Generate All Typedefs

**File:** `src/backends/blas_lapack_reference_backend.c` (line ~50)

```c
#define TYPEDEF_OPERATION(name, ret_type, fort_name, params) \
    typedef ret_type (*fblas_##name##_t)params;

#define REGISTER_BLAS_LEVEL1
#define REGISTER_BLAS_LEVEL2
#define REGISTER_BLAS_LEVEL3
#define REGISTER_LAPACK

#define OP(name, ret, fort, params) TYPEDEF_OPERATION(name, ret, fort, params)
#include "operation_registry.h"
#undef OP

#undef REGISTER_BLAS_LEVEL1
#undef REGISTER_BLAS_LEVEL2
#undef REGISTER_BLAS_LEVEL3
#undef REGISTER_LAPACK
```

**Output:** All typedefs like `typedef void (*fblas_saxpy_t)(int*, float*, ...);`

### Example 2: Generate All Wrapper Functions

**File:** `src/backends/blas_lapack_reference_backend.c` (line ~500)

```c
#define WRAPPER_OPERATION(name, ret_type, fort_name, params) \
    static ret_type fb_blr_##name(params) { \
        static fblas_##name##_t name##_fn = NULL; \
        if (!name##_fn) { \
            name##_fn = (fblas_##name##_t)FB_GET_PROC_ADDRESS(g_blr_handle, #fort_name); \
            if (!name##_fn) { \
                if (sizeof(ret_type) > 1) return 0; else return; \
            } \
        } \
        /* Convert parameters and call */ \
        /* (implementation depends on parameter types) */ \
    }

#define REGISTER_BLAS_LEVEL1
#define REGISTER_BLAS_LEVEL2
#define REGISTER_BLAS_LEVEL3
#define REGISTER_LAPACK

#define OP(name, ret, fort, params) WRAPPER_OPERATION(name, ret, fort, params)
#include "operation_registry.h"
#undef OP
```

**Output:** All 1248 wrapper functions

### Example 3: Generate VTable Entries

**File:** `src/backends/blas_lapack_reference_backend.c` (line ~1000)

```c
static fb_backend_vtable_t g_blr_vtable = {
    .name = "faster-blaster-reference",
    
    #define VTABLE_OPERATION(name, ret_type, fort_name, params) \
        .name = fb_blr_##name,
    
    #define REGISTER_BLAS_LEVEL1
    #define REGISTER_BLAS_LEVEL2
    #define REGISTER_BLAS_LEVEL3
    #define REGISTER_LAPACK
    
    #define OP(name, ret, fort, params) VTABLE_OPERATION(name, ret, fort, params)
    #include "operation_registry.h"
    #undef OP
};
```

**Output:** All 1248 function pointers in vtable

## Why This Is Better

### 1. Single Source of Truth

**Old approach:**
```python
OPERATIONS = [
    ("saxpy", "void", [...]),
    ("daxpy", "void", [...]),
    # ... repeat for each operation
]
# Python script generates C code
# If operation list changes, must regenerate
```

**New approach:**
```c
// operation_registry.h has all 1248 operations
// If you change it, C compiler automatically picks up the change
// No script needed, no generation needed
```

### 2. Compile-Time Verification

**Old approach:**
```
$ python generate_wrappers.py
$ cd ../.. && cmake && make
error: undefined reference to `fb_blr_saxpy`  ← caught at link time
```

**New approach:**
```
$ cd build && cmake && make
error: In expansion of macro 'OP': unexpected token
        .saxpy = fb_blr_saxpy,  ← caught at compile time
              ^
(compiler tells you exactly which operation is wrong)
```

### 3. Easy to Customize

Want only BLAS Level 1 and LAPACK? Just use those flags:

```c
#define REGISTER_BLAS_LEVEL1
#define REGISTER_LAPACK

#define OP(name, ret, fort, params) WRAPPER_OPERATION(name, ret, fort, params)
#include "operation_registry.h"
```

### 4. Easy to Debug

Each operation is just a macro call:
```c
OP(saxpy, void, saxpy_, (int n, float alpha, const float* x, int incx, float* y, int incy))
```

Can search for any operation in registry, see exactly what was generated.

### 5. Zero Build Overhead

No Python script to run. C preprocessor expansion is instant (microseconds).

## Implementation Plan

### Step 1: Create Operation Registry ✅ DONE

**File:** `src/backends/operation_registry.h`

Contains all 1248 operations as `OP()` macro calls.

```c
#ifdef REGISTER_BLAS_LEVEL1
OP(saxpy, void, saxpy_, (int n, float alpha, ...))
OP(daxpy, void, daxpy_, (int n, double alpha, ...))
// ... 54 more
#endif
```

### Step 2: Update blas_lapack_reference_backend.c

Three sections:

#### Section A: Generate Typedefs (lines ~50-150)

```c
#define TYPEDEF_OPERATION(name, ret, fort, params) \
    typedef ret (*fblas_##name##_t)params;

#define REGISTER_BLAS_LEVEL1
// ... include all levels ...

#define OP(name, ret, fort, params) TYPEDEF_OPERATION(name, ret, fort, params)
#include "operation_registry.h"
#undef OP
```

Result: 1248 typedefs

#### Section B: Generate Wrapper Functions (lines ~500-1500)

```c
#define WRAPPER_OPERATION(name, ret, fort, params) \
    static ret fb_blr_##name(params) { \
        /* lazy-load and call fort_## function */ \
    }

#define REGISTER_BLAS_LEVEL1
// ... include all levels ...

#define OP(name, ret, fort, params) WRAPPER_OPERATION(name, ret, fort, params)
#include "operation_registry.h"
#undef OP
```

Result: 1248 wrapper functions

#### Section C: Populate VTable (lines ~2000+)

```c
static fb_backend_vtable_t g_blr_vtable = {
    #define VTABLE_OPERATION(name, ret, fort, params) .name = fb_blr_##name,

    #define REGISTER_BLAS_LEVEL1
    // ... include all levels ...

    #define OP(name, ret, fort, params) VTABLE_OPERATION(name, ret, fort, params)
    #include "operation_registry.h"
    #undef OP
};
```

Result: All 1248 function pointers registered

### Step 3: Build & Verify

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make
# Compiler will expand all 1248 operations automatically
```

## Complete Example: SAXPY

### In operation_registry.h:
```c
OP(saxpy, void, saxpy_, (int n, float alpha, const float* x, int incx, float* y, int incy))
```

### After macro expansion (typedef):
```c
typedef void (*fblas_saxpy_t)(int* n, float* alpha, const float* x, int* incx, float* y, int* incy);
```

### After macro expansion (wrapper):
```c
static void fb_blr_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    static fblas_saxpy_t saxpy_fn = NULL;
    if (!saxpy_fn) {
        saxpy_fn = (fblas_saxpy_t)FB_GET_PROC_ADDRESS(g_blr_handle, "saxpy_");
        if (!saxpy_fn) return;
    }
    int n_copy = n, incx_copy = incx, incy_copy = incy;
    float alpha_copy = alpha;
    saxpy_fn(&n_copy, &alpha_copy, x, &incx_copy, y, &incy_copy);
}
```

### After macro expansion (vtable):
```c
fb_backend_vtable_t g_blr_vtable = {
    // ... other fields ...
    .saxpy = fb_blr_saxpy,
    // ... other operations ...
};
```

## Questions & Answers

**Q: How do we get the complete operation list?**

A: Extract from faster-blaster-reference:
   - Use `extract_operations_and_generate_macros.py` to scan all source files
   - Parse function signatures
   - Generate `operation_registry.h` with all 1248 operations

**Q: What if we want different behavior per operation type?**

A: Create multiple macros and include the registry multiple times:

```c
// Generate typedefs
#define OP TYPEDEF_OPERATION
#include "operation_registry.h"
#undef OP

// Generate wrappers
#define OP WRAPPER_OPERATION
#include "operation_registry.h"
#undef OP

// Generate vtable
#define OP VTABLE_OPERATION
#include "operation_registry.h"
#undef OP
```

**Q: Can we generate just BLAS without LAPACK?**

A: Yes, only define the flags you want:

```c
#define REGISTER_BLAS_LEVEL1
#define REGISTER_BLAS_LEVEL2
#define REGISTER_BLAS_LEVEL3
// NO REGISTER_LAPACK

#define OP WRAPPER_OPERATION
#include "operation_registry.h"
```

**Q: How many lines will this add to blas_lapack_reference_backend.c?**

A: Approximately:
- Typedefs: 1248 lines (1 per operation)
- Wrappers: 3-4 lines per operation = 3,750-5,000 lines
- Vtable entries: 1 line per operation = 1,248 lines
- **Total: ~6,000-7,000 lines** (same as before, but now generated by macros)

**Q: How do we handle operations with different parameter counts?**

A: The macro system handles it automatically:

```c
OP(sdot, float, sdot_, (int n, const float* x, int incx, const float* y, int incy))
         ↑     ↑         ↑ different numbers of parameters, macro handles it
OP(sgemm, void, sgemm_, (char transa, char transb, int m, int n, int k, ...))
```

## Files Involved

| File                                                | Purpose                                       |
| --------------------------------------------------- | --------------------------------------------- |
| `src/backends/operation_registry.h`                 | All 1248 operations as macro calls            |
| `src/backends/blas_lapack_reference_backend.c`      | 3 sections using the registry                 |
| `scripts/extract_operations_and_generate_macros.py` | (Optional) Auto-populate operation_registry.h |

## Performance

- **At compilation time:** C preprocessor expands all 1248 macros instantly
- **At runtime:** No difference from hand-written wrappers
- **File size:** Same ~6,000 lines either way

## Advantages Over Python Generation

| Aspect          | Python                                 | Macro                                   |
| --------------- | -------------------------------------- | --------------------------------------- |
| Generation time | seconds                                | instant (preprocessor)                  |
| Syntax errors   | found at linker stage                  | found at compile stage                  |
| Debugging       | generated code is static               | can view expanded macros with `-E` flag |
| Maintenance     | must regenerate if list changes        | automatic with preprocessor             |
| Accuracy        | potential script bugs                  | verified by compiler                    |
| Integration     | external tool dependency               | pure C                                  |
| IDE support     | no syntax highlight for generated code | full IDE support                        |

## Next Steps

1. ✅ Create `operation_registry.h` with all 1248 operations
2. ⏳ Update `blas_lapack_reference_backend.c` with three macro sections
3. ⏳ Build and verify all 1248 operations generated correctly
4. ⏳ Test one operation (e.g., SAXPY) end-to-end

**Total time to complete:** ~2-3 hours (much faster than Python generation!)
