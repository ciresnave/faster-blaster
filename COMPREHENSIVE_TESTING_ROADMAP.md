# Comprehensive Testing Roadmap for faster-blaster

**Status**: Comprehensive plan for testing all 1662 standard operations + 1175 extensions  
**Last Updated**: 2026-02-14  
**Scope**: Complete BLAS/LAPACK/Extensions testing strategy across all backends

## Overview

This document outlines the complete testing strategy for faster-blaster v1.0, covering:
- **Phase 1 (Core)**: 1662 standard BLAS/LAPACK operations
- **Phase 2 (Extensions)**: 1175 extended operations (Sparse, FFT, Tensor, etc.)
- **Testing Tiers**: Unit → Integration → Performance → Cross-Platform

---

## Phase 1: Standard BLAS/LAPACK (1662 operations)

### BLAS Level 1 (54 operations) ✅ IN PROGRESS

**Completed Tests**:
- ✅ SAXPY (single, vector-vector operation)
- ✅ DAXPY (double precision)
- ✅ CAXPY (complex single)
- ✅ ZAXPY (complex double)

**Pending** (50 remaining):
- Rotation: SROTG, DROTG, CROTG, ZROTG, SROTM, DROTM, SROT, DROT, CROT, ZROT, CSROT, ZDROT (12)
- Vector Ops: SSCAL, DSCAL, CSCAL, ZSCAL, CSSCAL, ZDSCAL, SCOPY, DCOPY, CCOPY, ZCOPY, SSWAP, DSWAP, CSWAP, ZSWAP (14)
- Reductions: SDOT, DDOT, CDOTU, ZDOTU, CDOTC, ZDOTC, SNRM2, DNRM2, SCNRM2, DZNRM2, SASUM, DASUM, SCASUM, DZASUM (14)
- Indexing: ISAMAX, IDAMAX, ICAMAX, IZAMAX (4)
- Extended: SDSDOT, DSDOT (2)
- Special: SROTMG, DROTMG (2)
- Blas-Like: SBMV, DBMV, SHER, DHER, SHER2, DHER2, SHBMV, DHBMV, SHPMV, DHPMV (10)

**Testing Strategy**:
1. **Unit tests** per operation (edge cases, special values)
2. **Cross-backend validation** (MKL vs OpenBLAS vs BLIS vs Reference)
3. **Precision variants** (SDOT → DDOT → CDOTU → CDOTC → ZDOTU → ZDOTC)
4. **Numerical properties** (associativity, commutativity where applicable)

**Test Matrix**:
```
54 ops × 3 backends (MKL, OpenBLAS, Reference) × 5 test cases = 810 test points
+ 4 precision variants per operation = 216 precision-specific tests
= 1026 test assertions
```

---

### BLAS Level 2 (90 operations) ⏳ PLANNED

**Categories**:

#### 1. General Matrix-Vector (14 operations)
- `?GEMV`: SGEMV, DGEMV, CGEMV, ZGEMV (4) ← **PRIORITY 1**
- `?GBMV`: SGBMV, DGBMV, CGBMV, ZGBMV (4)
- `?GER`: SGER, DGER, CGER, ZGER (implied by SGER)
- `?GERU`: CGERU, ZGERU (2) ← Complex rank-1 unconjugated
- `?GERC`: CGERC, ZGERC (2) ← Complex rank-1 conjugated

**Test Coverage**:
```c
// Example test: SGEMV
fb_status_t test_sgemv() {
    // Setup: A[m×n], x[n], y[m]
    // Operation: y := α*A*x + β*y
    // Variants: trans={N,T,C}, layout={RowMajor, ColMajor}
    // Cross-check: Reference BLAS vs MKL vs OpenBLAS
    // Tolerances: FP32 (1e-5), FP64 (1e-14)
}
```

#### 2. Symmetric/Hermitian Matrix-Vector (26 operations)
- `?SYMV`: SSYMV, DSYMV, CSYMV, ZSYMV (4) ← Symmetric
- `?SYR`: SSYR, DSYR, CSYR, ZSYR (4) ← Symmetric rank-1 update
- `?SYR2`: SSYR2, DSYR2, CSYR2, ZSYR2 (4) ← Symmetric rank-2 update
- `?SPMV`: SSPMV, DSPMV, CSPMV, ZSPMV (4) ← Packed storage
- `?SPR`: SSPR, DSPR, CSPR, ZSPR (4) ← Packed rank-1
- `?SPR2`: SSPR2, DSPR2, CSPR2, ZSPR2 (4) ← Packed rank-2
- `?SSBMV`: SSBMV, DSBMV, CSBMV, ZSBMV (4) ← Banded
- `?CHEMV`: CHEMV, ZHEMV (2) ← Hermitian (complex only)
- `?CHER`: CHER, ZHER (2) ← Hermitian rank-1
- `?CHER2`: CHER2, ZHER2 (2) ← Hermitian rank-2
- `?CHBMV`: CHBMV, ZHBMV (2) ← Hermitian banded
- `?CHPMV`: CHPMV, ZHPMV (2) ← Hermitian packed
- `?CHPR`: CHPR, ZHPR (2) ← Hermitian packed rank-1
- `?CHPR2`: CHPR2, ZHPR2 (2) ← Hermitian packed rank-2

#### 3. Triangular Matrix-Vector (24 operations)
- `?TRMV`: STRMV, DTRMV, CTRMV, ZTRMV (4) ← General triangular
- `?TRSV`: STRSV, DTRSV, CTRSV, ZTRSV (4) ← Triangular solve
- `?TBMV`: STBMV, DTBMV, CTBMV, ZTBMV (4) ← Banded triangular
- `?TBSV`: STBSV, DTBSV, CTBSV, ZTBSV (4) ← Banded triangular solve
- `?TPMV`: STPMV, DTPMV, CTPMV, ZTPMV (4) ← Packed triangular
- `?TPSV`: STPSV, DTPSV, CTPSV, ZTPSV (4) ← Packed triangular solve

**Total Level 2 Test Points**: 90 ops × 3 backends × 8 test cases = 2,160 assertions

**Timeline**: 
- Week 1: GEMV, GER (6 ops) + infrastructure
- Week 2: SYM* operations (26 ops)
- Week 3: Triangle* operations (24 ops)
- Week 4: Edge cases, performance profiling

---

### BLAS Level 3 (30 operations) ⏳ PLANNED

#### 1. General Matrix-Matrix (4 operations) ← **CRITICAL**
- `SGEMM`, `DGEMM`, `CGEMM`, `ZGEMM` (4)

**Unified Implementation Testing** (Section 6 of spec):
```c
// Tests verify that both paths produce identical results:
// 1. Direct: sgemm(...) → vtable->sgemm(...) → backend
// 2. Unified: sgemm(...) → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_SINGLE, 1, FB_FUSION_NONE, ...) → vtable->gemm_unified(...)
// Both must produce identical results with <1 ULP difference
```

#### 2. Symmetric/Hermitian Multiply (6 operations)
- `?SYMM`: SSYMM, DSYMM, CSYMM, ZSYMM (4)
- `?HEMM`: CHEMM, ZHEMM (2)

#### 3. Rank-K Updates (10 operations)
- `?SYRK`: SSYRK, DSYRK, CSYRK, ZSYRK (4) ← Symmetric rank-K
- `?HERK`: CHERK, ZHERK (2) ← Hermitian rank-K
- `?SYR2K`: SSYR2K, DSYR2K, CSYR2K, ZSYR2K (4) ← Symmetric rank-2K
- `?HER2K`: CHER2K, ZHER2K (2) ← Hermitian rank-2K

#### 4. Triangular Multiply & Solve (10 operations)
- `?TRMM`: STRMM, DTRMM, CTRMM, ZTRMM (4) ← Triangular multiply
- `?TRSM`: STRSM, DTRSM, CTRSM, ZTRSM (4) ← Triangular solve
- `?GEMM3M`: For complex (fused multiply using 3 real GEMMs)

**Test Coverage**:
```c
// For SGEMM:
// - Dimensions: square (8×8), tall (100×50), wide (50×100), large (1000×1000)
// - Layouts: RowMajor, ColMajor, mixed
// - Transpose: No-op, Transpose A, Transpose B, Transpose both
// - Scalars: α={0, ±1, 2.5}, β={0, ±1}
// - Numerical stability: Hilbert matrices, Pascal matrices, random
// - Batching: Single, array (8×), strided (8×)
// - Mixed-precision: FP32, BF16 (aliased to unified)
// - Fusions: relu, gelu, bias, residual (aliased to unified)
```

**Total Level 3 Test Points**: 30 ops × 3 backends × 12 test cases = 1,080 assertions

---

### LAPACK: Driver Routines (264 operations) ⏳ PLANNED

#### 1. Linear System Solvers (86 operations)
```
GESV (General): 12 ops
GBSV (Banded): 12 ops  
GTSV (Tridiagonal): 8 ops
POSV (Positive Definite): 12 ops
PPSV (Packed PD): 8 ops
PBSV (Banded PD): 8 ops
PTSV (Tridiagonal PD): 8 ops
SYSV/HESV (Symmetric/Hermitian Indefinite): 14 ops
SPSV/HPSV (Packed Indefinite): 8 ops
```

**Testing Strategy**:
- Generate well-conditioned systems (κ ≈ 10)
- Generate ill-conditioned systems (κ ≈ 10^6) → test error bounds
- Verify solution accuracy: ||Ax - b|| / ||A|| ||x|| < tolerance
- Check error bound estimates (expert drivers with ERRBND)

#### 2. Eigenvalue Problems (108 operations)
```
SYEV/HEEV (Dense): 16 ops
SPEV/HPEV (Packed): 12 ops
SBEV/HBEV (Banded): 12 ops
STEV (Tridiagonal): 8 ops
SYGV/HEGV (Generalized): 36 ops (12 dense + 12 packed + 12 banded)
```

**Numerical Properties to Verify**:
- Eigenvalue ordering: λ₁ ≤ λ₂ ≤ ... ≤ λₙ
- Orthonormality: V^T V = I (or V^H V = I for Hermitian)
- Reconstruction: A = V Λ V^T (with residual bounds)
- Generalized: Verify A V = B V Λ

#### 3. Singular Value Decomposition (8 operations)
```
GESVD (QR-based): 4 ops
GESDD (Divide-and-conquer): 4 ops
```

**Verification**:
- Singular values: σ₁ ≥ σ₂ ≥ ... ≥ σₚ ≥ 0
- Orthonormality: U^T U = I, V^T V = I
- Reconstruction: A ≈ U Σ V^T (with bounds)
- Rank determination: count σᵢ > ε

#### 4. Least Squares (16 operations)
```
GELS (QR): 4 ops
GELSY (Complete orthogonal): 4 ops
GELSS (SVD-based): 4 ops
GELSD (Divide-and-conquer SVD): 4 ops
```

#### 5. Generalized Eigenvalues (62 operations)
```
GGES/GGESX (Schur): 8 ops
GGEV/GGEVX (Eigenvalues): 8 ops
GGSVD (Generalized SVD): 4 ops
+ Generalized symmetric/Hermitian variants (42 ops)
```

**Priority Ranking**:
1. **TIER 1** (most common): GESV, POSV, GELS, GEEV, SYEV, GESVD (24 ops)
2. **TIER 2** (scientific): SYGV, GGES, GGEV (16 ops)
3. **TIER 3** (specialized): Band, packed, tridiagonal variants (224 ops)

---

### LAPACK: Computational Routines (840 operations) ⏳ PLANNED

**Categories** (by frequency of use):
1. **Factorizations** (LU, QR, Cholesky): ~120 ops ← **PRIORITY 1**
2. **Reductions** (Hessenberg, tridiagonal, bidiagonal): ~60 ops
3. **Eigenvalue drivers** (QR, RRR, etc): ~160 ops
4. **Orthogonal matrix generation**: ~80 ops
5. **Triangular system solvers**: ~40 ops
6. **Condition estimation**: ~40 ops
7. **Equilibration**: ~20 ops
8. **Other auxiliary**: ~320 ops

**Factorization Testing Example**:
```c
test_getrf_accuracy() {
    // Test LU factorization: A = P L U
    // 1. Generate matrix A
    // 2. Compute LU decomposition
    // 3. Verify: A ≈ P^T L U  (residual ||A - P^T LU|| / ||A|| < ε)
    // 4. Test GETRS solve: Ax = b using LU factors
    // 5. Verify: ||Ax - b|| / ||A|| ||x|| < tolerance
    // 6. Test GETRI inversion: verify A A^-1 ≈ I
}
```

---

### LAPACK: Auxiliary Routines (384 operations) ⏳ PLANNED

**High-Value Targets**:
- `LANxx` (52 ops): Matrix norms → used by condition estimators
- `LARF*` (36 ops): Householder reflections → used in QR/Hessenberg
- `LARFG` (4 ops): Householder vector generation → fundamental
- `LAQXX` (28 ops): Scaling/equilibration → improve numerical stability

---

## Phase 2: Extended Operations (1175 operations)

### A. Sparse BLAS + Preconditioners (236 operations)

**Testing Approach**:
```
Test Matrices:
- Small: 100×100, 5% sparsity
- Medium: 10K×10K, 0.01% sparsity  
- Large: 1M×1M, 0.001% sparsity (distributed)

Inspector-Executor Phases:
1. Create matrix (format: CSR, COO, CSC)
2. Optimize (backend selects best kernel)
3. Execute operation (SpMV, SpMM, iterative solver)
4. Verify result (compare to dense BLAS)
```

**Priority**:
1. SpMV (most common): 4 ops
2. Iterative solvers (GMRES, CG): 8 ops
3. Preconditioners (ILU, Jacobi): 12 ops
4. Format conversions: 8 ops

### B. Tensor Operations (40 operations)

**Testing Focus**:
- Multi-dimensional contractions (einsum notation)
- Permutations across device boundaries
- Fusion validation (single kernel vs multi-pass)

**Example**:
```c
test_tensor_contract() {
    // Einstein notation: D[i,j,k,l] = A[i,j,m,n] * B[m,n,k,l]
    // Compare: 
    //   - cuTENSOR implementation
    //   - Fallback GEMM-based implementation
    // Verify: ||D_tensor - D_gemm|| / ||D_tensor|| < 1e-5
}
```

### C. FFT Operations (68 operations)

**Testing Plan**:
```
Transforms:
- 1D, 2D, 3D
- Complex↔Complex
- Real↔Complex  
- Batched versions

Test Cases:
- Power-of-2 sizes (32 to 2^20)
- Mixed-radix sizes (powers of 2,3,5)
- Prime sizes (worst case)
- In-place vs out-of-place
- Forward then backward (round-trip accuracy)
```

**Numerical Validation**:
- Parseval's theorem: Σ|x(n)|² = Σ|X(k)|² / N
- Shift property: FFT(x[n-m]) = e^(-2πikm/N) X[k]
- Linearity: FFT(αx + βy) = α FFT(x) + β FFT(y)

### D. Deep Learning Primitives (100 operations)

**Testing Coverage**:
```
Convolution:
- Batch size: 1, 8, 64
- Kernel: 1×1, 3×3, 5×5, 7×7
- Padding: valid, same, full
- Stride: 1, 2, 4
- Dilation: 1, 2, 4
- Precision: FP32, FP16, BF16, INT8

Activation Functions:
- ReLU, LeakyReLU, ELU, GELU, Swish, Mish
- Range: x ∈ [-2, 2]
- Test derivative computation

Batch Normalization:
- Training vs Inference modes
- Verify running statistics updates
- Check gradient computation
```

**Cross-Backend Validation**:
- cuDNN vs MIOpen vs oneDNN vs ZenDNN
- Tolerance by precision:
  - FP32: 1e-5
  - FP16: 1e-3 (with slower convergence accepted)
  - INT8: Exact match on quantized output

### E. Other Extensions

**Random Number Generation** (48 ops):
- Statistical tests (Kolmogorov-Smirnov, χ²)
- Seed reproducibility
- Distribution accuracy (mean, variance, skewness)

**Extended Math** (120 ops):
- Compare against reference implementations (GSL, MPFR)
- Test special function properties (zeros, poles, recurrence relations)

**Data Fitting & Statistics** (48 ops):
- Spline interpolation accuracy on known functions
- Statistical correctness (mean, covariance, correlation)

---

## Testing Infrastructure

### Test Framework

**Location**: `tests/`

**Structure**:
```
tests/
├── blas_l1/              # BLAS Level 1 tests
│   ├── test_saxpy.c     # Single precision AXPY
│   ├── test_daxpy.c     # Double precision AXPY
│   └── ...
├── blas_l2/              # BLAS Level 2 tests
├── blas_l3/              # BLAS Level 3 tests
├── lapack_drivers/       # LAPACK driver routine tests
├── lapack_computational/ # LAPACK computational tests
├── sparse/               # Sparse BLAS tests
├── tensor/               # Tensor operation tests
├── common/               # Shared test utilities
│   ├── reference_blas.h  # Link to reference backend
│   ├── test_helpers.h    # Tolerance, error checking
│   ├── matrix_generator.h # Random matrix generation
│   └── performance.h     # Timing utilities
└── CMakeLists.txt        # Test configuration
```

### Test Template

```c
// tests/blas_l2/test_sgemv.c
#include <stdio.h>
#
#include <faster-blaster/faster_blaster.h>
#include "../common/test_helpers.h"

// Test SGEMV: y := α*A*x + β*y
void test_sgemv_basic() {
    const int m = 100, n = 50;
    float *A = malloc(m * n * sizeof(float));
    float *x = malloc(n * sizeof(float));
    float *y = malloc(m * sizeof(float));
    
    // Generate test data
    generate_random_matrix(A, m, n);
    generate_random_vector(x, n);
    generate_random_vector(y, m);
    
    float alpha = 2.5, beta = -1.3;
    float *y_expected = malloc(m * sizeof(float));
    memcpy(y_expected, y, m * sizeof(float));
    
    // Compute expected result (reference backend)
    fb_sgemv(FB_LAYOUT_COL_MAJOR, FB_TRANSPOSE_NO, 
             m, n, alpha, A, m, x, 1, beta, y_expected, 1);
    
    // Compute with optimized backend
    fb_sgemv(FB_LAYOUT_COL_MAJOR, FB_TRANSPOSE_NO, 
             m, n, alpha, A, m, x, 1, beta, y, 1);
    
    // Verify
    assert_vectors_near(y, y_expected, m, 1e-5, "SGEMV basic");
    
    free(A); free(x); free(y); free(y_expected);
}

void test_sgemv_transpose() {
    // Test with transposed A: y := α*A^T*x + β*y
    // ...
}

void test_sgemv_ill_conditioned() {
    // Generate ill-conditioned matrix (e.g., Hilbert matrix)
    // Verify still accurate
}
```

### Cross-Backend Validation

**Compare Against**:
1. **Reference Backend** (faster-blaster-reference) - pure C23
2. **Intel MKL** (if available)
3. **OpenBLAS** (open-source CPU)
4. **BLIS** (research BLAS)
5. **System BLAS** (Accelerate on macOS, etc.)

**Tolerance Strategy**:
```c
#define TOLERANCE_FP64 1e-14  // Double: machine epsilon ≈ 2.2e-16
#define TOLERANCE_FP32 1e-5   // Single: machine epsilon ≈ 1.2e-7
#define TOLERANCE_FP16 1e-3   // Half: machine epsilon ≈ 4.9e-4

// Adjusted by operation complexity
float actual_tol = base_tol * log10(problem_size);
```

---

## Test Execution Strategy

### Tier 1: Unit Tests (Local, Fast)

**Run**: `ctest -L unit`  
**Time**: < 5 minutes  
**Scope**: Single operations, small matrices (≤ 100×100)  
**Coverage**: 80% of total test points

```bash
cd build && ctest -L unit --output-on-failure
```

### Tier 2: Integration Tests (Cross-Backend)

**Run**: `ctest -L integration`  
**Time**: 15-30 minutes  
**Scope**: Multi-operation sequences (e.g., GETRF → GETRS)  
**Coverage**: Operation combinations

### Tier 3: Performance Tests (Optional)

**Run**: `ctest -L performance`  
**Time**: Variable (can take 1+ hour)  
**Scope**: Large matrices (100K+ elements)  
**Metrics**: GFLOPs, memory bandwidth, cache efficiency

### Tier 4: Platform-Specific Tests (CI/CD)

**Run on**:
- Linux (x86-64, ARM)
- Windows (Visual Studio, Clang)
- macOS (Intel, Apple Silicon)
- GPU (NVIDIA, AMD)

---

## Current Status & Next Steps

### ✅ Completed
- [x] BLAS Level 1: AXPY operations (4 ops)
- [x] Test infrastructure (framework, helpers)
- [x] Cross-backend validation setup

### 🔄 In Progress
- [ ] Complete BLAS Level 1 (50 remaining)
- [ ] Begin BLAS Level 2 (90 ops)

### ⏳ Planned (Q1 2026)
- [ ] BLAS Level 3 (30 ops) - **CRITICAL**
- [ ] LAPACK Drivers (264 ops) - prioritize Tier 1
- [ ] Sparse BLAS (236 ops)

### 📋 Deferred (Post-v1.0)
- [ ] ScaLAPACK distributed tests (requires MPI cluster)
- [ ] GPU-specific tests (requires NVIDIA/AMD hardware)
- [ ] Performance benchmarking suite

---

## Success Criteria

**Phase 1 Complete When**:
- ✅ 100% BLAS operations pass cross-backend validation
- ✅ LAPACK Tier 1 drivers (24 ops) tested and verified
- ✅ < 2 ULP error on all operations
- ✅ All precision variants working (S/D/C/Z)
- ✅ Edge cases handled (NaN, Inf, zero, very small/large)

**Minimum for v1.0 Release**:
- ✅ All BLAS (174 ops)
- ✅ LAPACK drivers (264 ops)
- ✅ LAPACK computational (core: ~200 ops)
- ✅ LAPACK auxiliary (core: ~100 ops)
- ✅ Sparse BLAS Inspector-Executor (132 ops)

---

## Execution Timeline

```
Week 1:  BLAS Level 1 completion (50 ops)
Week 2:  BLAS Level 2 start (GEMV, SYM*) (40 ops)
Week 3:  BLAS Level 2 finish (TRI*) (24 ops)
Week 4:  BLAS Level 3 (30 ops)
Week 5:  LAPACK Tier 1 drivers (24 ops)
Week 6:  LAPACK Tier 1 computational (40 ops)
Week 7:  LAPACK Tier 2 drivers (remaining 240 ops)
Week 8:  Sparse BLAS + Extensions (236+ ops)
Week 9:  Performance profiling & optimization
Week 10: Final validation, cross-platform testing
```

**Expected Completion**: Q2 2026

