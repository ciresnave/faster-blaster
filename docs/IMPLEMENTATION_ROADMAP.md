# faster-blaster Implementation Roadmap

## Project Vision

A hardware-calibrated BLAS abstraction layer that automatically selects the fastest backend (OpenBLAS, MKL, cuBLAS, BLIS, Accelerate) for each operation on the current hardware through runtime benchmarking.

## Architecture Decision: Type-Specific CBLAS-Style Interface

**Chosen**: Option B - CBLAS-style type-specific functions (saxpy/daxpy/caxpy/zaxpy)

**Rationale**:
- Simpler for backend implementers (direct 1:1 mapping to existing BLAS libraries)
- More familiar to users (matches CBLAS, BLAS.jl, NumPy conventions)
- Type safety at compile time
- Zero runtime overhead for type dispatch
- Cleaner function signatures (no void* pointer casting)

**Trade-off accepted**: More function pointers in vtable (4x for s/d/c/z variants)

---

## Phase 1: Core Infrastructure (Foundation)

### 1.1: Complete Backend Interface Definition

**File**: `src/backends/backend_interface.h`

**Goal**: Define all 160+ BLAS function pointer types with full CBLAS signatures

**Operations to include**:

#### BLAS Level 1 (Vector-Vector) - 13 core operations × 4 types = 52 functions
- `ROTG` - Generate plane rotation (4 variants: s/d/c/z)
- `ROTMG` - Generate modified plane rotation (2 variants: s/d only)
- `ROT` - Apply plane rotation (4 variants)
- `ROTM` - Apply modified plane rotation (2 variants)
- `SWAP` - Exchange vectors (4 variants)
- `SCAL` - Scale vector (6 variants: s/d/c/z/cs/zd)
- `COPY` - Copy vector (4 variants)
- `AXPY` - y = alpha*x + y (4 variants)
- `DOT` - Dot product (8 variants: sdot, ddot, dsdot, sdsdot, cdotu, zdotu, cdotc, zdotc)
- `NRM2` - Euclidean norm (4 variants: snrm2, dnrm2, scnrm2, dznrm2)
- `ASUM` - Sum of absolute values (4 variants: sasum, dasum, scasum, dzasum)
- `IAMAX` - Index of max absolute value (4 variants)

**Total Level 1**: 52 function pointer types

#### BLAS Level 2 (Matrix-Vector) - 25 core operations × 4 types = ~70 functions
- `GEMV` - General matrix-vector multiply (4 variants)
- `GBMV` - Banded matrix-vector multiply (4 variants)
- `HEMV` - Hermitian matrix-vector multiply (2 variants: c/z only)
- `HBMV` - Hermitian banded matrix-vector multiply (2 variants)
- `HPMV` - Hermitian packed matrix-vector multiply (2 variants)
- `SYMV` - Symmetric matrix-vector multiply (2 variants: s/d only)
- `SBMV` - Symmetric banded matrix-vector multiply (2 variants)
- `SPMV` - Symmetric packed matrix-vector multiply (2 variants)
- `TRMV` - Triangular matrix-vector multiply (4 variants)
- `TBMV` - Triangular banded matrix-vector multiply (4 variants)
- `TPMV` - Triangular packed matrix-vector multiply (4 variants)
- `TRSV` - Triangular solve (4 variants)
- `TBSV` - Triangular banded solve (4 variants)
- `TPSV` - Triangular packed solve (4 variants)
- `GER` - Rank-1 update (2 variants: s/d)
- `GERU` - Unconjugated rank-1 update (2 variants: c/z)
- `GERC` - Conjugated rank-1 update (2 variants: c/z)
- `HER` - Hermitian rank-1 update (2 variants)
- `HPR` - Hermitian packed rank-1 update (2 variants)
- `HER2` - Hermitian rank-2 update (2 variants)
- `HPR2` - Hermitian packed rank-2 update (2 variants)
- `SYR` - Symmetric rank-1 update (2 variants)
- `SPR` - Symmetric packed rank-1 update (2 variants)
- `SYR2` - Symmetric rank-2 update (2 variants)
- `SPR2` - Symmetric packed rank-2 update (2 variants)

**Total Level 2**: 70 function pointer types

#### BLAS Level 3 (Matrix-Matrix) - 10 core operations × 4 types = ~30 functions
- `GEMM` - General matrix-matrix multiply (4 variants)
- `SYMM` - Symmetric matrix multiply (4 variants)
- `HEMM` - Hermitian matrix multiply (2 variants: c/z only)
- `SYRK` - Symmetric rank-k update (4 variants)
- `HERK` - Hermitian rank-k update (2 variants)
- `SYR2K` - Symmetric rank-2k update (4 variants)
- `HER2K` - Hermitian rank-2k update (2 variants)
- `TRMM` - Triangular matrix multiply (4 variants)
- `TRSM` - Triangular solve with multiple RHS (4 variants)

**Total Level 3**: 30 function pointer types

#### LAPACK Essential Subset - 20 operations × 4 types = ~60 functions

**Linear Systems**:
- `GESV` - General linear system solve (4 variants)
- `POSV` - Positive definite system solve (4 variants)
- `SYSV` - Symmetric indefinite system solve (2 variants)
- `HESV` - Hermitian indefinite system solve (2 variants)
- `GBSV` - Banded linear system solve (4 variants)
- `GTSV` - Tridiagonal system solve (4 variants)

**Factorizations**:
- `GETRF` - LU factorization (4 variants)
- `POTRF` - Cholesky factorization (4 variants)
- `SYTRF` - Symmetric indefinite factorization (2 variants)
- `HETRF` - Hermitian indefinite factorization (2 variants)
- `GBTRF` - Banded LU factorization (4 variants)
- `GTTRF` - Tridiagonal LU factorization (4 variants)

**Matrix Inversions**:
- `GETRI` - General matrix inversion (4 variants)
- `POTRI` - Positive definite matrix inversion (4 variants)
- `TRTRI` - Triangular matrix inversion (4 variants)

**Eigenvalue/SVD**:
- `GEEV` - General eigenvalue problem (4 variants)
- `SYEV` - Symmetric eigenvalue problem (2 variants)
- `HEEV` - Hermitian eigenvalue problem (2 variants)
- `GESVD` - Singular value decomposition (4 variants)

**QR/LQ**:
- `GEQRF` - QR factorization (4 variants)
- `ORGQR/UNGQR` - Generate Q from QR (4 variants)

**Total LAPACK Essentials**: 60 function pointer types

**Grand Total**: 212 function pointer types

**vtable structure**:
```c
typedef struct {
    /* Metadata */
    fb_backend_info_t info;
    
    /* Level 1 - 52 function pointers */
    fb_srotg_fn srotg;
    fb_drotg_fn drotg;
    // ... (all 52)
    
    /* Level 2 - 70 function pointers */
    fb_sgemv_fn sgemv;
    fb_dgemv_fn dgemv;
    // ... (all 70)
    
    /* Level 3 - 30 function pointers */
    fb_sgemm_fn sgemm;
    fb_dgemm_fn dgemm;
    // ... (all 30)
    
    /* LAPACK Essentials - 60 function pointers */
    fb_sgesv_fn sgesv;
    fb_dgesv_fn dgesv;
    // ... (all 60)
    
} fb_backend_vtable_t;
```

**Implementation steps**:
1. Define all typedefs for function pointers (212 total)
2. Define fb_backend_vtable_t struct with all fields
3. Add comprehensive documentation for each operation
4. Include parameter validation rules in comments
5. Document which operations are optional (can be NULL)

**MSVC Compatibility Requirements**:
- Use `fb_complex_float_t` and `fb_complex_double_t` instead of `float complex`/`double complex`
- Define as structs on MSVC, use C11 complex on other compilers
- Ensure all typedefs use `int64_t` for indices (not int)
- Use `const` correctly on all pointer parameters

---

### 1.2: Hardware Detection Module

**File**: `src/core/hardware_detect.h` and `hardware_detect.c`

**Current structure is adequate**:
```c
typedef struct {
    char cpu_vendor[64];        // "Intel", "AMD", "ARM", "Apple"
    char cpu_model[128];         // Full model string
    int cpu_cores;               // Physical cores
    int cpu_threads;             // Logical threads (with HT/SMT)
    uint64_t cpu_l1_cache;       // L1 cache size in bytes
    uint64_t cpu_l2_cache;       // L2 cache size in bytes
    uint64_t cpu_l3_cache;       // L3 cache size in bytes
    uint64_t ram_bytes;          // Total RAM
    
    bool has_gpu;                // GPU available
    char gpu_vendor[64];         // "NVIDIA", "AMD", "Intel", "Apple"
    char gpu_model[128];         // GPU model name
    int gpu_compute_units;       // CUs/SMs
    uint64_t gpu_memory_bytes;   // VRAM
    
    char fingerprint[64];        // Unique hardware ID for calibration cache
} fb_hardware_info_t;
```

**Action**: No changes needed, current structure is good.

---

### 1.3: Dispatch System

**Files**: `src/core/dispatch.h` and `dispatch.c`

**Already mostly complete**, needs:

1. **dispatch.h updates**:
   - ✅ Already has `fb_dispatch_policy_t` enum
   - ✅ Already has full `fb_dispatch_table_t` struct
   - Need to expand `fb_operation_id_t` enum to include all 212 operations

2. **dispatch.c updates**:
   - ✅ Backend registration works
   - ✅ Auto-selection logic is solid
   - Update to use `cpu_vendor` instead of `cpu.vendor` (already fixed)
   - Update to use `has_gpu` and `gpu_vendor` (already fixed)

**New operation enum**:
```c
typedef enum {
    /* Level 1 - Vector ops */
    FB_OP_SROTG, FB_OP_DROTG, FB_OP_CROTG, FB_OP_ZROTG,
    FB_OP_SROTMG, FB_OP_DROTMG,
    FB_OP_SROT, FB_OP_DROT, FB_OP_CROT, FB_OP_ZROT,
    // ... (all 52)
    
    /* Level 2 - Matrix-vector ops */
    FB_OP_SGEMV, FB_OP_DGEMV, FB_OP_CGEMV, FB_OP_ZGEMV,
    // ... (all 70)
    
    /* Level 3 - Matrix-matrix ops */
    FB_OP_SGEMM, FB_OP_DGEMM, FB_OP_CGEMM, FB_OP_ZGEMM,
    // ... (all 30)
    
    /* LAPACK */
    FB_OP_SGESV, FB_OP_DGESV, FB_OP_CGESV, FB_OP_ZGESV,
    // ... (all 60)
    
    FB_OP_COUNT  // = 212
} fb_operation_id_t;
```

---

### 1.4: Calibration System Redesign

**Files**: `src/core/calibration.h` and `calibration.c`

**Current issues**: References old generic `fb_dtype_t` system

**New design**: Type-specific calibration functions

```c
/* Remove old generic types, use operation-specific structs */

typedef struct {
    int64_t n;                    // Vector size
    int64_t incx, incy;           // Strides
    double execution_time_ns;     // Median time
    double gflops;                // Performance
    bool correctness_passed;      // Validation result
} fb_level1_calibration_result_t;

typedef struct {
    int64_t m, n;                 // Matrix dimensions
    int64_t lda, incx, incy;      // Leading dims and strides
    fb_transpose_t trans;         // Transpose mode
    double execution_time_ns;
    double gflops;
    double bandwidth_gbps;        // Memory bandwidth achieved
    bool correctness_passed;
} fb_level2_calibration_result_t;

typedef struct {
    int64_t m, n, k;              // Matrix dimensions
    int64_t lda, ldb, ldc;        // Leading dimensions
    fb_transpose_t transA, transB;
    double execution_time_ns;
    double gflops;
    double efficiency_percent;    // % of theoretical peak
    bool correctness_passed;
} fb_level3_calibration_result_t;

/* Calibration configuration */
typedef struct {
    size_t warmup_iterations;     // Warmup runs to prime caches
    size_t benchmark_iterations;  // Actual timed iterations
    size_t num_test_sizes;        // How many sizes to test
    int64_t *test_sizes;          // Array of sizes to benchmark
    double tolerance;             // Numerical tolerance for correctness
    bool validate_correctness;    // Whether to check results
    const char *output_json;      // JSON file path for results
} fb_calibration_config_t;

/* Public API */
int fb_calibrate_backend(
    const fb_backend_vtable_t *backend,
    const fb_calibration_config_t *config,
    fb_calibration_result_t ***results_out,  // Array of results per operation
    size_t *num_results_out
);

int fb_calibrate_operation(
    const fb_backend_vtable_t *backend,
    fb_operation_id_t operation,
    const fb_calibration_config_t *config,
    fb_calibration_result_t **results_out,
    size_t *num_results_out
);

/* Database management */
typedef struct fb_calibration_database_t fb_calibration_database_t;

fb_calibration_database_t* fb_calibration_db_create(const char *cache_dir);
void fb_calibration_db_destroy(fb_calibration_database_t *db);

int fb_calibration_db_store(
    fb_calibration_database_t *db,
    const fb_hardware_info_t *hw_info,
    const char *backend_name,
    fb_operation_id_t operation,
    const fb_calibration_result_t *results,
    size_t num_results
);

int fb_calibration_db_query(
    fb_calibration_database_t *db,
    const fb_hardware_info_t *hw_info,
    const char *backend_name,
    fb_operation_id_t operation,
    fb_calibration_result_t **results_out,
    size_t *num_results_out
);

int fb_calibration_db_export_json(
    fb_calibration_database_t *db,
    const char *filepath
);

int fb_calibration_db_import_json(
    fb_calibration_database_t *db,
    const char *filepath
);
```

**Implementation tasks**:
1. Remove all references to old `fb_dtype_t` generic types
2. Implement operation-specific benchmark harnesses
3. Build SQLite-based calibration database
4. Implement JSON import/export for sharing calibration data
5. Add automatic hardware fingerprinting for cache keys
6. Implement multi-backend comparison framework

---

### 1.5: Test Data Generation

**Files**: `src/core/test_data.h` and `test_data.c`

**Current implementation is good**, but needs:

1. Add complex number generation functions
2. Add matrix condition number control
3. Add special matrices (ill-conditioned, sparse, banded, triangular, etc.)

```c
/* Add to test_data.h */

/* Complex matrix generation */
int fb_test_generate_complex_matrix_float(
    fb_test_matrix_type_t type,
    int64_t rows,
    int64_t cols,
    fb_complex_float_t *matrix,
    uint64_t seed
);

int fb_test_generate_complex_matrix_double(
    fb_test_matrix_type_t type,
    int64_t rows,
    int64_t cols,
    fb_complex_double_t *matrix,
    uint64_t seed
);

/* Condition number control */
int fb_test_generate_matrix_conditioned(
    int64_t n,
    double condition_number,  // Target condition number
    double *matrix,
    uint64_t seed
);

/* Special structures */
int fb_test_generate_banded_matrix(
    int64_t n,
    int64_t kl,              // Lower bandwidth
    int64_t ku,              // Upper bandwidth
    double *matrix,
    uint64_t seed
);

int fb_test_generate_tridiagonal_matrix(
    int64_t n,
    double *dl,              // Lower diagonal
    double *d,               // Main diagonal
    double *du,              // Upper diagonal
    uint64_t seed
);
```

---

## Phase 2: Reference Backend Implementation

### 2.1: Complete Reference Backend

**File**: `src/backends/reference.c`

**Goal**: Implement ALL 212 operations in naive C for correctness baseline

**Current status**: Has 14 operations (saxpy, daxpy, scopy, dcopy, sdot, ddot, snrm2, dnrm2, sscal, dscal, sgemv, dgemv, sgemm, dgemm)

**Implementation approach**:

1. **Level 1 operations** (52 functions):
   - Straightforward loops over vectors
   - No complex optimizations
   - Focus on clarity and correctness
   - Example template:
   ```c
   void reference_saxpy(const int64_t n, const float alpha, 
                        const float *x, const int64_t incx,
                        float *y, const int64_t incy) {
       for (int64_t i = 0; i < n; i++) {
           y[i * incy] += alpha * x[i * incx];
       }
   }
   ```

2. **Level 2 operations** (70 functions):
   - Matrix-vector loops
   - Handle transpose, symmetric, hermitian, triangular cases
   - Example template:
   ```c
   void reference_sgemv(const fb_layout_t layout, const fb_transpose_t trans,
                        const int64_t m, const int64_t n,
                        const float alpha, const float *A, const int64_t lda,
                        const float *x, const int64_t incx,
                        const float beta, float *y, const int64_t incy) {
       // Scale y by beta first
       for (int64_t i = 0; i < m; i++) {
           y[i * incy] *= beta;
       }
       
       // Add alpha * A * x
       if (trans == FB_NO_TRANS) {
           for (int64_t i = 0; i < m; i++) {
               float sum = 0.0f;
               for (int64_t j = 0; j < n; j++) {
                   sum += A[i * lda + j] * x[j * incx];
               }
               y[i * incy] += alpha * sum;
           }
       } else {
           // Handle transpose...
       }
   }
   ```

3. **Level 3 operations** (30 functions):
   - Triple nested loops
   - Handle all transpose combinations
   - Symmetric/hermitian optimizations

4. **LAPACK operations** (60 functions):
   - **Linear systems**: Gaussian elimination with partial pivoting
   - **Factorizations**: Standard textbook algorithms
   - **Eigenvalues**: QR algorithm for symmetric, power iteration for general
   - **SVD**: Golub-Reinsch algorithm
   - Focus on numerical stability over performance

**Complex number operations**:
```c
/* Helper functions for complex arithmetic */
static inline fb_complex_float_t complex_float_add(
    fb_complex_float_t a, fb_complex_float_t b) {
    return (fb_complex_float_t){a.real + b.real, a.imag + b.imag};
}

static inline fb_complex_float_t complex_float_mul(
    fb_complex_float_t a, fb_complex_float_t b) {
    return (fb_complex_float_t){
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

static inline fb_complex_float_t complex_float_conj(
    fb_complex_float_t a) {
    return (fb_complex_float_t){a.real, -a.imag};
}
```

**Backend vtable initialization**:
```c
static const fb_backend_vtable_t reference_vtable = {
    .info = {
        .name = "reference",
        .version = "1.0.0",
        .vendor = "faster-blaster",
        .capabilities = FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | FB_CAP_CPU,
        .hw_type = FB_HW_UNKNOWN,
        .thread_safe = true,
        .min_efficient_size = 1,
        .max_matrix_size = 0  // No limit
    },
    
    /* Level 1 - All 52 functions */
    .srotg = reference_srotg,
    .drotg = reference_drotg,
    // ... populate all 52
    
    /* Level 2 - All 70 functions */
    .sgemv = reference_sgemv,
    .dgemv = reference_dgemv,
    // ... populate all 70
    
    /* Level 3 - All 30 functions */
    .sgemm = reference_sgemm,
    .dgemm = reference_dgemm,
    // ... populate all 30
    
    /* LAPACK - All 60 functions */
    .sgesv = reference_sgesv,
    .dgesv = reference_dgesv,
    // ... populate all 60
};

const fb_backend_vtable_t* reference_get_vtable(void) {
    return &reference_vtable;
}
```

**Estimated effort**: 40-60 hours (212 functions, ~15-20 minutes each)

---

## Phase 3: Public API Layer

### 3.1: Public Header Design

**File**: `include/faster_blaster.h`

**Goal**: User-facing CBLAS-compatible API

```c
/* BLAS Level 1 - All 52 operations */
void fb_saxpy(const int64_t n, const float alpha, const float *x, 
              const int64_t incx, float *y, const int64_t incy);
void fb_daxpy(const int64_t n, const double alpha, const double *x,
              const int64_t incx, double *y, const int64_t incy);
// ... all 52

/* BLAS Level 2 - All 70 operations */
void fb_sgemv(const fb_layout_t layout, const fb_transpose_t trans,
              const int64_t m, const int64_t n, const float alpha,
              const float *A, const int64_t lda, const float *x, const int64_t incx,
              const float beta, float *y, const int64_t incy);
// ... all 70

/* BLAS Level 3 - All 30 operations */
void fb_sgemm(const fb_layout_t layout, const fb_transpose_t transA,
              const fb_transpose_t transB, const int64_t m, const int64_t n, const int64_t k,
              const float alpha, const float *A, const int64_t lda,
              const float *B, const int64_t ldb, const float beta,
              float *C, const int64_t ldc);
// ... all 30

/* LAPACK - All 60 operations */
int fb_sgesv(const fb_layout_t layout, const int64_t n, const int64_t nrhs,
             float *A, const int64_t lda, int64_t *ipiv,
             float *B, const int64_t ldb);
// ... all 60
```

### 3.2: API Implementation Files

**Files**: `src/api/level1.c`, `level2.c`, `level3.c`, `lapack.c`

**Structure**: Each function dispatches to the selected backend

```c
/* Example: src/api/level1.c */

#include "faster_blaster.h"
#include "../core/dispatch.h"

void fb_saxpy(const int64_t n, const float alpha, const float *x,
              const int64_t incx, float *y, const int64_t incy) {
    const fb_backend_vtable_t *backend = fb_dispatch_get_backend();
    if (backend && backend->saxpy) {
        backend->saxpy(n, alpha, x, incx, y, incy);
    } else {
        /* Fallback error or reference implementation */
        fprintf(stderr, "faster-blaster: No backend available for saxpy\n");
        abort();
    }
}

// Repeat for all 52 Level 1 operations
```

**Implementation tasks**:
1. Create `level1.c` with all 52 Level 1 wrappers
2. Create `level2.c` with all 70 Level 2 wrappers
3. Create `level3.c` with all 30 Level 3 wrappers
4. Create `lapack.c` with all 60 LAPACK wrappers
5. Add error handling for NULL backend pointers
6. Add optional statistics collection (call counts, total time)

**Estimated effort**: 10-15 hours (mostly repetitive code generation)

---

## Phase 4: Optimized Backend Wrappers

### 4.1: OpenBLAS Backend

**Files**: `src/backends/openblas.c`, `openblas.h`

**Goal**: Wrap OpenBLAS as the primary performance baseline

**Implementation**:
```c
#include <cblas.h>
#include "backend_interface.h"

/* Wrapper functions - direct pass-through to OpenBLAS */

static void openblas_saxpy(const int64_t n, const float alpha,
                           const float *x, const int64_t incx,
                           float *y, const int64_t incy) {
    cblas_saxpy((int)n, alpha, x, (int)incx, y, (int)incy);
}

static void openblas_sgemm(const fb_layout_t layout, const fb_transpose_t transA,
                           const fb_transpose_t transB, const int64_t m,
                           const int64_t n, const int64_t k, const float alpha,
                           const float *A, const int64_t lda,
                           const float *B, const int64_t ldb,
                           const float beta, float *C, const int64_t ldc) {
    CBLAS_ORDER order = (layout == FB_LAYOUT_ROW_MAJOR) ? 
                        CblasRowMajor : CblasColMajor;
    CBLAS_TRANSPOSE transA_cb = (transA == FB_NO_TRANS) ? CblasNoTrans :
                                  (transA == FB_TRANS) ? CblasTrans : CblasConjTrans;
    CBLAS_TRANSPOSE transB_cb = (transB == FB_NO_TRANS) ? CblasNoTrans :
                                  (transB == FB_TRANS) ? CblasTrans : CblasConjTrans;
    
    cblas_sgemm(order, transA_cb, transB_cb, (int)m, (int)n, (int)k,
                alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
}

/* Populate vtable with all 212 operations */
static const fb_backend_vtable_t openblas_vtable = {
    .info = {
        .name = "openblas",
        .version = "0.3.x",  // Detect at runtime
        .vendor = "OpenBLAS",
        .capabilities = FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 | FB_CAP_CPU,
        .hw_type = FB_HW_UNKNOWN,  // Works on all CPUs
        .thread_safe = true,
        .min_efficient_size = 64,
        .max_matrix_size = 0
    },
    .saxpy = openblas_saxpy,
    .sgemm = openblas_sgemm,
    // ... all 212 functions
};
```

**Build integration**:
- Add `find_package(OpenBLAS)` to CMakeLists.txt
- Conditional compilation if OpenBLAS not found
- Auto-register backend if library is available

**Estimated effort**: 15-20 hours

---

### 4.2: Intel MKL Backend

**Files**: `src/backends/mkl.c`, `mkl.h`

**Similar to OpenBLAS but wrapping MKL**:
- Use MKL's CBLAS interface
- Take advantage of MKL-specific extensions where available
- Conditional compilation for Intel CPUs

**Estimated effort**: 15-20 hours

---

### 4.3: BLIS Backend

**Files**: `src/backends/blis.c`, `blis.h`

**Wrap BLIS (BLAS-like Library Instantiation Software)**:
- AMD-optimized BLAS alternative
- Similar interface to OpenBLAS

**Estimated effort**: 15-20 hours

---

### 4.4: Apple Accelerate Backend

**Files**: `src/backends/accelerate.c`, `accelerate.h`

**Wrap Apple's Accelerate framework** (macOS/iOS only):
- Use vecLib/BLAS component
- Optimized for Apple Silicon and Intel Macs
- Conditional compilation for macOS

**Estimated effort**: 15-20 hours

---

### 4.5: GPU Backends (Future)

**cuBLAS** (NVIDIA GPUs):
- Requires CUDA SDK
- Async operation support
- Device memory management

**rocBLAS** (AMD GPUs):
- Requires ROCm SDK
- Similar to cuBLAS interface

**oneMKL** (Intel GPUs):
- Requires Intel oneAPI
- Multi-device support

**Estimated effort per GPU backend**: 30-40 hours each

---

## Phase 5: Testing Infrastructure

### 5.1: Unit Tests

**Files**: `tests/test_level1.c`, `test_level2.c`, `test_level3.c`, `test_lapack.c`

**Test framework**: Use Catch2 or Google Test (C++ wrapper around C code)

**Test strategy**:
```c
/* For each operation, test: */

1. **Correctness**: Reference vs Optimized backends
   - Compare outputs within numerical tolerance
   - Test all data types (s/d/c/z)
   - Test edge cases (n=0, n=1, n=large)
   - Test stride variations (incx=1, incx=2, incx=-1)
   - Test transpose modes (NoTrans, Trans, ConjTrans)

2. **Numerical Stability**:
   - Test with ill-conditioned matrices
   - Test with extreme values (near overflow/underflow)
   - Test with NaN/Inf handling

3. **Parameter Validation**:
   - Test with invalid inputs
   - Verify error codes/handling

4. **Performance Regression**:
   - Benchmark each operation
   - Compare against baseline
   - Alert if >10% slowdown
```

**Test generator**: Create Python script to generate test code

```python
# scripts/generate_tests.py

OPERATIONS = [
    {'name': 'saxpy', 'level': 1, 'type': 'float', ...},
    {'name': 'daxpy', 'level': 1, 'type': 'double', ...},
    # ... all 212 operations
]

for op in OPERATIONS:
    generate_correctness_test(op)
    generate_edge_case_tests(op)
    generate_performance_test(op)
```

**Estimated effort**: 30-40 hours

---

### 5.2: Consensus-Based Correctness Testing

**File**: `tests/consensus_test.c`

**Concept**: Run operation on all available backends, compare results

```c
int fb_test_consensus(fb_operation_id_t operation,
                      void *inputs,
                      void **outputs_per_backend,
                      int num_backends,
                      double tolerance) {
    /* 
     * 1. Execute operation on all backends
     * 2. Compare all outputs pairwise
     * 3. Identify majority consensus
     * 4. Flag outliers
     * 5. For GPU backends, use relaxed tolerance
     * 6. Use reference backend as tiebreaker
     */
    
    int consensus_count = 0;
    void *consensus_output = NULL;
    
    // Find majority agreement
    for (int i = 0; i < num_backends; i++) {
        int agreement_count = 0;
        for (int j = 0; j < num_backends; j++) {
            if (outputs_match(outputs_per_backend[i],
                             outputs_per_backend[j],
                             tolerance)) {
                agreement_count++;
            }
        }
        if (agreement_count > consensus_count) {
            consensus_count = agreement_count;
            consensus_output = outputs_per_backend[i];
        }
    }
    
    // Report outliers
    for (int i = 0; i < num_backends; i++) {
        if (!outputs_match(outputs_per_backend[i],
                          consensus_output,
                          tolerance)) {
            fprintf(stderr, "Backend %d disagrees with consensus\n", i);
        }
    }
    
    return consensus_count >= (num_backends / 2) ? 0 : -1;
}
```

**GPU precision handling**:
- Use relaxed tolerance for GPU backends (1e-5 vs 1e-10)
- Document expected precision differences
- Track systematic biases

**Estimated effort**: 10-15 hours

---

### 5.3: Benchmark Suite

**File**: `benchmarks/benchmark_all.c`

**Benchmark each operation**:
- Multiple problem sizes (64, 128, 256, 512, 1024, 2048, 4096, 8192)
- All backends
- Generate performance plots
- Export to CSV/JSON

**Metrics**:
- Execution time (median, min, max, stddev)
- GFLOPS
- Memory bandwidth (GB/s)
- Efficiency (% of theoretical peak)

**Output format**:
```json
{
  "hardware": {
    "cpu": "Intel Core i9-13900K",
    "cores": 24,
    "threads": 32,
    "ram_gb": 64,
    "gpu": "NVIDIA RTX 4090"
  },
  "benchmarks": [
    {
      "operation": "sgemm",
      "size": {"m": 1024, "n": 1024, "k": 1024},
      "backends": {
        "reference": {"time_ns": 125000000, "gflops": 17.1},
        "openblas": {"time_ns": 2500000, "gflops": 858.9},
        "mkl": {"time_ns": 2100000, "gflops": 1021.4},
        "cublas": {"time_ns": 180000, "gflops": 11929.8}
      }
    }
  ]
}
```

**Estimated effort**: 15-20 hours

---

## Phase 6: Documentation

### 6.1: API Documentation

**Use Doxygen** to generate from inline comments

**Files to document**:
- `include/faster_blaster.h` - All 212 public functions
- `include/faster_blaster_config.h` - Configuration API
- All backend interfaces

**Documentation requirements per function**:
- Brief description
- Mathematical formula (LaTeX)
- Parameter descriptions
- Return value semantics
- Example usage
- Performance notes
- Numerical considerations

**Example**:
```c
/**
 * @brief Single precision general matrix multiply: C = alpha*op(A)*op(B) + beta*C
 * 
 * Computes:
 * \f[
 * C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
 * \f]
 * 
 * where op(X) is one of:
 * - op(X) = X (no transpose)
 * - op(X) = X^T (transpose)
 * - op(X) = X^H (conjugate transpose)
 * 
 * @param[in] layout Matrix storage layout (row-major or column-major)
 * @param[in] transA Transpose operation for matrix A
 * @param[in] transB Transpose operation for matrix B
 * @param[in] m Number of rows in op(A) and C
 * @param[in] n Number of columns in op(B) and C
 * @param[in] k Number of columns in op(A) and rows in op(B)
 * @param[in] alpha Scalar multiplier for A*B
 * @param[in] A Input matrix A (size m×k if no transpose)
 * @param[in] lda Leading dimension of A (stride between rows/columns)
 * @param[in] B Input matrix B (size k×n if no transpose)
 * @param[in] ldb Leading dimension of B
 * @param[in] beta Scalar multiplier for C
 * @param[in,out] C Input/output matrix C (size m×n), overwritten with result
 * @param[in] ldc Leading dimension of C
 * 
 * @note This operation is O(m*n*k) complexity
 * @note Performance is highly dependent on matrix size and cache behavior
 * @note Optimal performance typically requires m, n, k >= 128
 * 
 * @par Example:
 * @code
 * float A[1024], B[1024], C[1024];
 * // Initialize A and B...
 * fb_sgemm(FB_LAYOUT_ROW_MAJOR, FB_NO_TRANS, FB_NO_TRANS,
 *          32, 32, 32, 1.0f, A, 32, B, 32, 0.0f, C, 32);
 * @endcode
 * 
 * @see fb_dgemm, fb_cgemm, fb_zgemm
 */
void fb_sgemm(const fb_layout_t layout, const fb_transpose_t transA,
              const fb_transpose_t transB, const int64_t m, const int64_t n,
              const int64_t k, const float alpha, const float *A,
              const int64_t lda, const float *B, const int64_t ldb,
              const float beta, float *C, const int64_t ldc);
```

**Estimated effort**: 40-50 hours

---

### 6.2: User Guide

**File**: `docs/USER_GUIDE.md`

**Contents**:
1. **Installation**:
   - CMake build instructions
   - Dependency installation
   - Platform-specific notes

2. **Quick Start**:
   - Minimal example
   - Initialization
   - Basic operations

3. **Configuration**:
   - Backend selection strategies
   - Calibration process
   - Hardware detection

4. **Advanced Usage**:
   - Custom backends
   - Performance tuning
   - Multi-threading
   - GPU operations

5. **Troubleshooting**:
   - Common errors
   - Performance issues
   - Numerical stability

**Estimated effort**: 10-15 hours

---

### 6.3: Developer Guide

**File**: `docs/DEVELOPER_GUIDE.md`

**Contents**:
1. **Architecture Overview**:
   - Component diagram
   - Data flow
   - Design decisions

2. **Adding Operations**:
   - Step-by-step guide
   - Code generation templates
   - Testing requirements

3. **Backend Development**:
   - Interface requirements
   - Vtable population
   - Registration process
   - Calibration integration

4. **Contributing**:
   - Code style guidelines
   - PR process
   - Testing requirements

**Estimated effort**: 10-15 hours

---

## Phase 7: Build System & Packaging

### 7.1: CMake Improvements

**File**: `CMakeLists.txt`

**Enhancements needed**:

```cmake
cmake_minimum_required(VERSION 3.15)
project(faster-blaster VERSION 1.0.0 LANGUAGES C)

# Options
option(FB_BUILD_TESTS "Build test suite" ON)
option(FB_BUILD_BENCHMARKS "Build benchmarks" ON)
option(FB_BUILD_EXAMPLES "Build examples" ON)
option(FB_ENABLE_OPENBLAS "Enable OpenBLAS backend" ON)
option(FB_ENABLE_MKL "Enable Intel MKL backend" ON)
option(FB_ENABLE_BLIS "Enable BLIS backend" ON)
option(FB_ENABLE_ACCELERATE "Enable Apple Accelerate backend" ON)
option(FB_ENABLE_CUBLAS "Enable NVIDIA cuBLAS backend" OFF)
option(FB_ENABLE_ROCBLAS "Enable AMD rocBLAS backend" OFF)
option(FB_ENABLE_CALIBRATION "Enable calibration system" ON)

# C23 standard required
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Dependencies
find_package(Threads REQUIRED)
if(FB_ENABLE_OPENBLAS)
    find_package(OpenBLAS)
    if(OpenBLAS_FOUND)
        set(FB_HAS_OPENBLAS ON)
    endif()
endif()

# Similar for MKL, BLIS, etc.

# Library target
add_library(faster-blaster
    src/core/dispatch.c
    src/core/hardware_detect.c
    src/core/calibration.c
    src/core/timing.c
    src/core/test_data.c
    src/api/level1.c
    src/api/level2.c
    src/api/level3.c
    src/api/lapack.c
    src/backends/reference.c
    $<$<BOOL:${FB_HAS_OPENBLAS}>:src/backends/openblas.c>
    $<$<BOOL:${FB_HAS_MKL}>:src/backends/mkl.c>
    # ... conditional backend files
)

target_include_directories(faster-blaster
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(faster-blaster
    PUBLIC
        Threads::Threads
    PRIVATE
        $<$<BOOL:${FB_HAS_OPENBLAS}>:OpenBLAS::OpenBLAS>
        $<$<BOOL:${FB_HAS_MKL}>:MKL::MKL>
)

# Generate config header with enabled backends
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/faster_blaster_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/faster_blaster_config.h
)

# Install targets
install(TARGETS faster-blaster
    EXPORT faster-blaster-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# Export for find_package()
install(EXPORT faster-blaster-targets
    FILE faster-blaster-targets.cmake
    NAMESPACE faster-blaster::
    DESTINATION lib/cmake/faster-blaster
)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/faster-blaster-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/faster-blaster-config-version.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/faster-blaster-config.cmake"
    DESTINATION lib/cmake/faster-blaster
)
```

**Estimated effort**: 5-8 hours

---

### 7.2: Package Configuration

**Create distribution packages**:
- Debian/Ubuntu `.deb` packages
- RedHat/Fedora `.rpm` packages
- macOS Homebrew formula
- Windows MSVC packages
- Vcpkg port
- Conan package

**Estimated effort**: 10-15 hours

---

## Phase 8: Performance Optimization

### 8.1: Dispatch Optimization

**Goal**: Zero-overhead dispatch after initialization

**Current**: Function pointer lookup per call
**Optimized**: Direct function pointer table

```c
/* Global function pointer cache */
typedef struct {
    /* Level 1 - 52 direct pointers */
    fb_saxpy_fn saxpy;
    fb_daxpy_fn daxpy;
    // ... all 212
} fb_fast_dispatch_table_t;

static fb_fast_dispatch_table_t g_fast_dispatch = {0};

/* Initialize once */
void fb_init(void) {
    const fb_backend_vtable_t *backend = select_optimal_backend();
    
    /* Populate direct pointers */
    g_fast_dispatch.saxpy = backend->saxpy;
    g_fast_dispatch.daxpy = backend->daxpy;
    // ... all 212 operations
}

/* Zero overhead call */
void fb_saxpy(const int64_t n, const float alpha,
              const float *x, const int64_t incx,
              float *y, const int64_t incy) {
    g_fast_dispatch.saxpy(n, alpha, x, incx, y, incy);
}
```

**Result**: Single indirect call, no lookup, no branching

**Estimated effort**: 3-5 hours

---

### 8.2: Calibration Data Sharing

**Create community calibration database**:
- Web service for submitting calibration results
- Download pre-calibrated data for common hardware
- Crowdsourced performance optimization

**Estimated effort**: 20-30 hours (includes backend service)

---

## Time Estimates Summary

| Phase       | Task                                      | Estimated Hours |
| ----------- | ----------------------------------------- | --------------- |
| **Phase 1** | Core Infrastructure                       |                 |
| 1.1         | Complete backend interface (212 typedefs) | 8-12            |
| 1.2         | Hardware detection (no changes needed)    | 0               |
| 1.3         | Dispatch system updates                   | 3-5             |
| 1.4         | Calibration system redesign               | 15-20           |
| 1.5         | Test data generation updates              | 5-8             |
| **Phase 2** | Reference Backend                         |                 |
| 2.1         | Implement all 212 operations              | 40-60           |
| **Phase 3** | Public API                                |                 |
| 3.1         | Public header design                      | 3-5             |
| 3.2         | API implementation (212 wrappers)         | 10-15           |
| **Phase 4** | Optimized Backends                        |                 |
| 4.1         | OpenBLAS backend (212 wrappers)           | 15-20           |
| 4.2         | Intel MKL backend                         | 15-20           |
| 4.3         | BLIS backend                              | 15-20           |
| 4.4         | Apple Accelerate backend                  | 15-20           |
| 4.5         | GPU backends (future)                     | 30-40 each      |
| **Phase 5** | Testing                                   |                 |
| 5.1         | Unit tests (212 operations)               | 30-40           |
| 5.2         | Consensus testing framework               | 10-15           |
| 5.3         | Benchmark suite                           | 15-20           |
| **Phase 6** | Documentation                             |                 |
| 6.1         | API documentation (212 functions)         | 40-50           |
| 6.2         | User guide                                | 10-15           |
| 6.3         | Developer guide                           | 10-15           |
| **Phase 7** | Build & Packaging                         |                 |
| 7.1         | CMake improvements                        | 5-8             |
| 7.2         | Package configuration                     | 10-15           |
| **Phase 8** | Performance Optimization                  |                 |
| 8.1         | Dispatch optimization                     | 3-5             |
| 8.2         | Calibration data sharing                  | 20-30           |
| **TOTAL**   | **Minimum**                               | **278 hours**   |
| **TOTAL**   | **Maximum**                               | **423 hours**   |

**Realistic estimate with buffer**: **350-450 hours** (~9-11 weeks full-time or ~18-22 weeks part-time)

---

## Implementation Order

### Sprint 1 (Weeks 1-2): Foundation
1. Complete backend_interface.h with all 212 typedefs
2. Update dispatch.h with expanded operation enum
3. Redesign calibration.h (remove old generic types)
4. Fix all compilation errors in existing code

**Deliverable**: Clean compilation of existing 14 operations

---

### Sprint 2 (Weeks 3-5): Reference Backend
1. Implement all Level 1 operations (52 functions)
2. Implement all Level 2 operations (70 functions)
3. Implement all Level 3 operations (30 functions)
4. Start LAPACK operations (linear systems first)

**Deliverable**: Complete reference backend with basic testing

---

### Sprint 3 (Weeks 6-7): Public API
1. Design final public header
2. Implement all 212 API wrappers
3. Add basic error handling
4. Create simple examples

**Deliverable**: Working public API that compiles and links

---

### Sprint 4 (Weeks 8-10): First Optimized Backend
1. Implement OpenBLAS backend (all 212 operations)
2. Add backend auto-detection
3. Basic calibration for GEMM operations
4. Performance comparison framework

**Deliverable**: Measurable performance improvement over reference

---

### Sprint 5 (Weeks 11-13): Additional Backends
1. Implement Intel MKL backend
2. Implement BLIS backend
3. Implement Apple Accelerate backend (if on macOS)
4. Cross-platform testing

**Deliverable**: Multiple working backends with auto-selection

---

### Sprint 6 (Weeks 14-16): Testing Infrastructure
1. Generate unit tests for all 212 operations
2. Implement consensus testing framework
3. Create comprehensive benchmark suite
4. Add continuous integration

**Deliverable**: Full test coverage and performance baselines

---

### Sprint 7 (Weeks 17-19): Calibration & Optimization
1. Complete calibration system implementation
2. SQLite database for calibration results
3. JSON import/export
4. Dispatch optimization (zero overhead)
5. Multi-backend calibration runs

**Deliverable**: Working calibration system with persistent cache

---

### Sprint 8 (Weeks 20-22): Documentation & Polish
1. Write comprehensive API documentation
2. Create user guide
3. Create developer guide
4. Add examples for common use cases
5. Performance tuning
6. Package creation

**Deliverable**: Production-ready library v1.0.0

---

## Future Extensions (Post v1.0)

### GPU Support
- cuBLAS backend (NVIDIA)
- rocBLAS backend (AMD)
- oneMKL backend (Intel)
- Device memory management
- Async operations
- Multi-GPU support

### Advanced Features
- Batched operations
- Strided batched operations
- Mixed precision support (FP16, BF16, INT8)
- Sparse matrix operations
- Extended LAPACK (eigensolvers, SVD)
- Automatic algorithm selection per operation

### Ecosystem Integration
- Python bindings (NumPy compatibility)
- Julia bindings
- Rust bindings
- Language-specific high-level APIs

---

## Success Criteria

### Functional Requirements
- ✅ All 212 operations implemented
- ✅ At least 3 optimized backends working
- ✅ Reference backend for correctness
- ✅ Automatic backend selection
- ✅ Calibration system operational
- ✅ Cross-platform (Linux, macOS, Windows)

### Performance Requirements
- ✅ Zero overhead dispatch (<1ns per call)
- ✅ Within 5% of native backend performance
- ✅ Automatic selection matches manual optimal 90% of the time

### Quality Requirements
- ✅ >95% test coverage
- ✅ All operations numerically validated
- ✅ No memory leaks (Valgrind clean)
- ✅ Thread-safe operation
- ✅ Comprehensive documentation

---

## Risk Mitigation

### Technical Risks
1. **Complex number support on MSVC**: Use struct-based implementation
2. **Performance overhead**: Optimize dispatch table, profile carefully
3. **Numerical precision issues**: Extensive testing with ill-conditioned matrices
4. **Backend availability**: Graceful fallback to reference implementation

### Project Risks
1. **Scope creep**: Stick to BLAS/LAPACK subset, defer exotic operations
2. **Backend API changes**: Version detection and compatibility layers
3. **Platform differences**: Comprehensive CI testing on all platforms
4. **Performance regression**: Automated benchmarking in CI

---

This roadmap provides a complete, methodical path to building a production-quality BLAS abstraction layer with hardware-specific optimization through calibration. No corners cut, no simplifications—just solid engineering.
