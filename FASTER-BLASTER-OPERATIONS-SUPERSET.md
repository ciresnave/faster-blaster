# Complete BLAS + LAPACK Superset Design

## Overview

This document enumerates the complete interface for faster-blaster v1.0, including all BLAS and LAPACK operations across all data types. Most pointers will be NULL initially; this defines the complete design surface for eventual implementation.

**Design Principle**: Operations are identified by (operation_name, data_type) pairs. The vtable must have entries for all combinations, even if initial implementations only support a subset.

**Canonical Surface Note (Judge-Aligned, 2026-04)**: The authoritative concrete operation namespace for faster-blaster is the `3054` judge-tracked names in `src/judge/judge_op_ids.h` (IDs `0..3053`). Appendix A is the exact sorted expansion of that namespace. The family sections below remain grouped by mathematical domain for readability and may discuss unified implementations, aliases, or planning-oriented subtotals; when those differ from a concrete name count, `judge_op_ids.h` and Appendix A win.

---

## BLAS LEVEL 1: Vector Operations

### Naming Convention: {prefix}{operation}

- **Prefix**: S (float), D (double), C (complex single), Z (complex double)
- **Prefix Variants for Special Cases**: SC (single complex), DZ (double complex) for certain operations
- **Example**: `srotg` (single), `drotg` (double), `crotg` (complex single), `zrotg` (complex double)

### Level 1 Operations (16 base operations = 52 total variants)

**Complete Enumerated List (52 operations):**

#### Rotation Operations (14 operations)

1. `SROTG` - Generate Givens rotation (single precision)
2. `DROTG` - Generate Givens rotation (double precision)
3. `CROTG` - Generate Givens rotation (complex single)
4. `ZROTG` - Generate Givens rotation (complex double)
5. `SROTMG` - Generate modified Givens rotation (single)
6. `DROTMG` - Generate modified Givens rotation (double)
7. `SROT` - Apply Givens rotation (single)
8. `DROT` - Apply Givens rotation (double)
9. `CROT` - Apply Givens rotation (complex single)
10. `ZROT` - Apply Givens rotation (complex double)
11. `CSROT` - Apply plane rotation (real cosine/sine, complex vectors)
12. `ZDROT` - Apply plane rotation (real cosine/sine, complex double vectors)
13. `SROTM` - Apply modified Givens rotation (single)
14. `DROTM` - Apply modified Givens rotation (double)

#### Vector Operations (32 operations)

1. `SSWAP` - Swap vectors (single)
2. `DSWAP` - Swap vectors (double)
3. `CSWAP` - Swap vectors (complex single)
4. `ZSWAP` - Swap vectors (complex double)
5. `SSCAL` - Scale vector (single)
6. `DSCAL` - Scale vector (double)
7. `CSCAL` - Scale vector (complex single)
8. `ZSCAL` - Scale vector (complex double)
9. `CSSCAL` - Scale complex vector by real scalar (single)
10. `ZDSCAL` - Scale complex vector by real scalar (double)
11. `SCOPY` - Copy vector (single)
12. `DCOPY` - Copy vector (double)
13. `CCOPY` - Copy vector (complex single)
14. `ZCOPY` - Copy vector (complex double)
15. `SAXPY` - Vector sum Y := a*X + Y (single)
16. `DAXPY` - Vector sum Y := a*X + Y (double)
17. `CAXPY` - Vector sum Y := a*X + Y (complex single)
18. `ZAXPY` - Vector sum Y := a*X + Y (complex double)
19. `SDOT` - Dot product (single)
20. `DDOT` - Dot product (double)
21. `CDOTU` - Dot product unconjugated (complex single)
22. `ZDOTU` - Dot product unconjugated (complex double)
23. `CDOTC` - Dot product conjugated (complex single)
24. `ZDOTC` - Dot product conjugated (complex double)
25. `SNRM2` - Euclidean norm (single)
26. `DNRM2` - Euclidean norm (double)
27. `SCNRM2` / `CNRM2` - Euclidean norm (complex single, returns real)
28. `DZNRM2` / `ZNRM2` - Euclidean norm (complex double, returns real)
29. `SASUM` - Sum of absolute values (single)
30. `DASUM` - Sum of absolute values (double)
31. `SCASUM` - Sum of absolute values (complex single, returns real)
32. `DZASUM` - Sum of absolute values (complex double, returns real)

#### Indexing Operations (4 operations)

1. `ISAMAX` - Index of maximum absolute value (single)
2. `IDAMAX` - Index of maximum absolute value (double)
3. `ICAMAX` - Index of maximum absolute value (complex single)
4. `IZAMAX` - Index of maximum absolute value (complex double)

#### Extended Precision Operations (6 operations)

1. `SDSDOT` - Single precision dot with double accumulation
2. `DSDOT` - Double precision dot with single inputs
3. `SROTMG` - Generate modified Givens (counted in Rotation Operations)
4. `DROTMG` - Generate modified Givens (counted in Rotation Operations)
5. `SROTM` - Apply modified Givens (counted in Rotation Operations)
6. `DROTM` - Apply modified Givens (counted in Rotation Operations)

**Total Level 1**: 52 operations

---

## BLAS LEVEL 2: Matrix-Vector Operations

### Naming Convention: {prefix}{matrix_type}{operation}

- **Prefix**: S, D, C, Z
- **Matrix Types**:
  - `GE` = General
  - `SY` = Symmetric (real) / Hermitian (complex)
  - `HE` = Hermitian (complex only)
  - `TR` = Triangular
  - `TB` = Triangular Banded
  - `GB` = General Banded
  - `SB` = Symmetric Banded
  - `HB` = Hermitian Banded (complex only)
  - `SP` = Symmetric Packed
  - `HP` = Hermitian Packed (complex only)
  - `TP` = Triangular Packed
  - `PB` = Packed Banded (ambiguous; may be covered under SB/HB/TB)

### Level 2 Operations (32 base operations = 90 total variants)

**Complete Enumerated List (90 operations):**

#### General Matrix Operations (14 operations)

1. `SGEMV` - General matrix-vector multiply (single)
2. `DGEMV` - General matrix-vector multiply (double)
3. `CGEMV` - General matrix-vector multiply (complex single)
4. `ZGEMV` - General matrix-vector multiply (complex double)
5. `SGBMV` - General banded matrix-vector multiply (single)
6. `DGBMV` - General banded matrix-vector multiply (double)
7. `CGBMV` - General banded matrix-vector multiply (complex single)
8. `ZGBMV` - General banded matrix-vector multiply (complex double)
9. `SGER` - General rank-1 update (single)
10. `DGER` - General rank-1 update (double)
11. `CGERU` - General rank-1 update, unconjugated (complex single)
12. `ZGERU` - General rank-1 update, unconjugated (complex double)
13. `CGERC` - General rank-1 update, conjugated (complex single)
14. `ZGERC` - General rank-1 update, conjugated (complex double)

#### Symmetric Matrix Operations (18 operations)

1. `SSYMV` - Symmetric matrix-vector multiply (single)
2. `DSYMV` - Symmetric matrix-vector multiply (double)
3. `CSYMV` - Symmetric matrix-vector multiply (complex single)
4. `ZSYMV` - Symmetric matrix-vector multiply (complex double)
5. `SSYR` - Symmetric rank-1 update (single)
6. `DSYR` - Symmetric rank-1 update (double)
7. `CSYR` - Symmetric rank-1 update (complex single)
8. `ZSYR` - Symmetric rank-1 update (complex double)
9. `SSYR2` - Symmetric rank-2 update (single)
10. `DSYR2` - Symmetric rank-2 update (double)
11. `CSYR2` - Symmetric rank-2 update (complex single)
12. `ZSYR2` - Symmetric rank-2 update (complex double)
13. `SSBMV` - Symmetric banded matrix-vector multiply (single)
14. `DSBMV` - Symmetric banded matrix-vector multiply (double)
15. `CSBMV` - Symmetric banded matrix-vector multiply (complex single)
16. `ZSBMV` - Symmetric banded matrix-vector multiply (complex double)
17. `SSPMV` - Symmetric packed matrix-vector multiply (single)
18. `DSPMV` - Symmetric packed matrix-vector multiply (double)
19. `CSPMV` - Symmetric packed matrix-vector multiply (complex single)
20. `ZSPMV` - Symmetric packed matrix-vector multiply (complex double)
21. `SSPR` - Symmetric packed rank-1 update (single)
22. `DSPR` - Symmetric packed rank-1 update (double)
23. `CSPR` - Symmetric packed rank-1 update (complex single)
24. `ZSPR` - Symmetric packed rank-1 update (complex double)
25. `SSPR2` - Symmetric packed rank-2 update (single)
26. `DSPR2` - Symmetric packed rank-2 update (double)
27. `CSPR2` - Symmetric packed rank-2 update (complex single)
28. `ZSPR2` - Symmetric packed rank-2 update (complex double)

#### Hermitian Matrix Operations (12 operations - complex only)

1. `CHEMV` - Hermitian matrix-vector multiply (complex single)
2. `ZHEMV` - Hermitian matrix-vector multiply (complex double)
3. `CHER` - Hermitian rank-1 update (complex single)
4. `ZHER` - Hermitian rank-1 update (complex double)
5. `CHER2` - Hermitian rank-2 update (complex single)
6. `ZHER2` - Hermitian rank-2 update (complex double)
7. `CHBMV` - Hermitian banded matrix-vector multiply (complex single)
8. `ZHBMV` - Hermitian banded matrix-vector multiply (complex double)
9. `CHPMV` - Hermitian packed matrix-vector multiply (complex single)
10. `ZHPMV` - Hermitian packed matrix-vector multiply (complex double)
11. `CHPR` - Hermitian packed rank-1 update (complex single)
12. `ZHPR` - Hermitian packed rank-1 update (complex double)
13. `CHPR2` - Hermitian packed rank-2 update (complex single)
14. `ZHPR2` - Hermitian packed rank-2 update (complex double)

**Real Hermitian-Equivalent Operations (10 operations - real precision variants):**
1. `SBMV` - Symmetric banded matrix-vector multiply (single)
2. `DBMV` - Symmetric banded matrix-vector multiply (double)
3. `SHER` - Symmetric rank-1 update (single)
4. `DHER` - Symmetric rank-1 update (double)
5. `SHER2` - Symmetric rank-2 update (single)
6. `DHER2` - Symmetric rank-2 update (double)
7. `SHBMV` - Symmetric/Hermitian banded matrix-vector (single)
8. `DHBMV` - Symmetric/Hermitian banded matrix-vector (double)
9. `SHPMV` - Symmetric/Hermitian packed matrix-vector (single)
10. `DHPMV` - Symmetric/Hermitian packed matrix-vector (double)

#### Triangular Matrix Operations (24 operations)

1. `STRMV` - Triangular matrix-vector multiply (single)
2. `DTRMV` - Triangular matrix-vector multiply (double)
3. `CTRMV` - Triangular matrix-vector multiply (complex single)
4. `ZTRMV` - Triangular matrix-vector multiply (complex double)
5. `STRSV` - Triangular system solve (single)
6. `DTRSV` - Triangular system solve (double)
7. `CTRSV` - Triangular system solve (complex single)
8. `ZTRSV` - Triangular system solve (complex double)
9. `STBMV` - Triangular banded matrix-vector multiply (single)
10. `DTBMV` - Triangular banded matrix-vector multiply (double)
11. `CTBMV` - Triangular banded matrix-vector multiply (complex single)
12. `ZTBMV` - Triangular banded matrix-vector multiply (complex double)
13. `STBSV` - Triangular banded system solve (single)
14. `DTBSV` - Triangular banded system solve (double)
15. `CTBSV` - Triangular banded system solve (complex single)
16. `ZTBSV` - Triangular banded system solve (complex double)
17. `STPMV` - Triangular packed matrix-vector multiply (single)
18. `DTPMV` - Triangular packed matrix-vector multiply (double)
19. `CTPMV` - Triangular packed matrix-vector multiply (complex single)
20. `ZTPMV` - Triangular packed matrix-vector multiply (complex double)
21. `STPSV` - Triangular packed system solve (single)
22. `DTPSV` - Triangular packed system solve (double)
23. `CTPSV` - Triangular packed system solve (complex single)
24. `ZTPSV` - Triangular packed system solve (complex double)

**Total Level 2**: 90 operations (includes 10 real Hermitian-equivalent operations)

---

## BLAS LEVEL 3: Matrix-Matrix Operations

### Naming Convention: {prefix}{operation}

- **Prefix**: S, D, C, Z

### Level 3 Operations (9 base operations = 30 total variants)

**Complete Enumerated List (30 operations):**

#### General Matrix Multiply (4 aliases)

**SGEMM** - Single-precision matrix multiply: C := α*A*B + β*C
```c
void sgemm(...)  // Alias → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_SINGLE, 1, FB_FUSION_NONE, FB_FUSION_NONE, ...)
```

**DGEMM** - Double-precision matrix multiply: C := α*A*B + β*C
```c
void dgemm(...)  // Alias → fb_gemm_unified(FB_PREC_FP64, FB_BATCH_SINGLE, 1, FB_FUSION_NONE, FB_FUSION_NONE, ...)
```

**CGEMM** - Complex single-precision matrix multiply: C := α*A*B + β*C
```c
void cgemm(...)  // Alias → fb_gemm_unified(FB_PREC_C32, FB_BATCH_SINGLE, 1, FB_FUSION_NONE, FB_FUSION_NONE, ...)
```

**ZGEMM** - Complex double-precision matrix multiply: C := α*A*B + β*C
```c
void zgemm(...)  // Alias → fb_gemm_unified(FB_PREC_C64, FB_BATCH_SINGLE, 1, FB_FUSION_NONE, FB_FUSION_NONE, ...)
```

⚠️ **Performance Note**: These are zero-cost convenience aliases. For batching, mixed-precision, or fused operations, use `fb_gemm_unified()` directly.

✅ **Migration Path**:
- **Drop-in replacement**: No code changes needed - aliases provide backward compatibility
- **Batched operations**: `fb_gemm_unified(FB_PREC_FP32, FB_BATCH_STRIDED, batch_count, ...)`
- **Fused activation**: `fb_gemm_unified(FB_PREC_FP32, FB_BATCH_SINGLE, 1, FB_FUSION_RELU, ...)`
- **Mixed precision**: `fb_gemm_unified(FB_PREC_BF16, ...)`

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

#### Symmetric Matrix Multiply (4 operations)

1. `SSYMM` - Symmetric matrix multiply C := α*A*B + β*C (single)
2. `DSYMM` - Symmetric matrix multiply C := α*A*B + β*C (double)
3. `CSYMM` - Symmetric matrix multiply C := α*A*B + β*C (complex single)
4. `ZSYMM` - Symmetric matrix multiply C := α*A*B + β*C (complex double)

#### Hermitian Matrix Multiply (2 operations - complex only)

1. `CHEMM` - Hermitian matrix multiply C := α*A*B + β*C (complex single)
2. `ZHEMM` - Hermitian matrix multiply C := α*A*B + β*C (complex double)

#### Symmetric Rank-K Update (4 operations)

1. `SSYRK` - Symmetric rank-k update C := α*A*A^T + β*C (single)
2. `DSYRK` - Symmetric rank-k update C := α*A*A^T + β*C (double)
3. `CSYRK` - Symmetric rank-k update C := α*A*A^T + β*C (complex single)
4. `ZSYRK` - Symmetric rank-k update C := α*A*A^T + β*C (complex double)

#### Hermitian Rank-K Update (2 operations - complex only)

1. `CHERK` - Hermitian rank-k update C := α*A*A^H + β*C (complex single)
2. `ZHERK` - Hermitian rank-k update C := α*A*A^H + β*C (complex double)

#### Symmetric Rank-2K Update (4 operations)

1. `SSYR2K` - Symmetric rank-2k update C := α*A*B^T + α*B*A^T + β*C (single)
2. `DSYR2K` - Symmetric rank-2k update C := α*A*B^T + α*B*A^T + β*C (double)
3. `CSYR2K` - Symmetric rank-2k update C := α*A*B^T + α*B*A^T + β*C (complex single)
4. `ZSYR2K` - Symmetric rank-2k update C := α*A*B^T + α*B*A^T + β*C (complex double)

#### Hermitian Rank-2K Update (2 operations - complex only)

1. `CHER2K` - Hermitian rank-2k update C := α*A*B^H + conj(α)*B*A^H + β*C (complex single)
2. `ZHER2K` - Hermitian rank-2k update C := α*A*B^H + conj(α)*B*A^H + β*C (complex double)

#### Triangular Matrix Multiply (4 operations)

1. `STRMM` - Triangular matrix multiply B := α*op(A)*B (single)
2. `DTRMM` - Triangular matrix multiply B := α*op(A)*B (double)
3. `CTRMM` - Triangular matrix multiply B := α*op(A)*B (complex single)
4. `ZTRMM` - Triangular matrix multiply B := α*op(A)*B (complex double)

#### Triangular Solve (4 operations)

1. `STRSM` - Triangular solve op(A)*X = α*B (single)
2. `DTRSM` - Triangular solve op(A)*X = α*B (double)
3. `CTRSM` - Triangular solve op(A)*X = α*B (complex single)
4. `ZTRSM` - Triangular solve op(A)*X = α*B (complex double)

**Total Level 3**: 30 operations

---

## BLAS SUMMARY

| Level          | Base Operations | Variants | Total Slots |
| -------------- | --------------- | -------- | ----------- |
| 1              | 16              | varies   | 54          |
| 2              | 32              | varies   | 90          |
| 3              | 9               | varies   | 30          |
| **BLAS Total** |                 |          | **174**     |

---

## LAPACK: Linear Algebra Package

LAPACK is significantly larger. Operations are organized by category and typically exist in S, D, C, Z variants (LAPACK 3.12.1).

### Naming Convention: {prefix}{category}{operation}

- **Prefix**: S (float), D (double), C (complex float), Z (complex double)
- **Category** (2-3 letters after prefix):
  - `GE` = General matrix
  - `SY` = Symmetric / `HE` = Hermitian
  - `OR` = Orthogonal / `UN` = Unitary
  - `TR` = Triangular
  - `PO` = Positive definite
  - `GB` = General banded
  - `ST` = Symmetric tridiagonal
  - etc.

---

### LAPACK DRIVER ROUTINES (246 operations)

**Complete Enumerated List:**

#### Linear System Solvers (86 operations)

**General Linear System (GESV family - 12 operations):**

1. `SGESV` - General linear system solve Ax=b (single)
2. `DGESV` - General linear system solve Ax=b (double)
3. `CGESV` - General linear system solve Ax=b (complex single)
4. `ZGESV` - General linear system solve Ax=b (complex double)
5. `SGESVX` - Expert driver with error bounds (single)
6. `DGESVX` - Expert driver with error bounds (double)
7. `CGESVX` - Expert driver with error bounds (complex single)
8. `ZGESVX` - Expert driver with error bounds (complex double)
9. `SGESVXX` - Extra-precise expert driver with improved error bounds (single)
10. `DGESVXX` - Extra-precise expert driver with improved error bounds (double)
11. `CGESVXX` - Extra-precise expert driver with improved error bounds (complex single)
12. `ZGESVXX` - Extra-precise expert driver with improved error bounds (complex double)

**General Banded (GBSV family - 12 operations):**
9. `SGBSV` - Banded linear system solve (single)
10. `DGBSV` - Banded linear system solve (double)
11. `CGBSV` - Banded linear system solve (complex single)
12. `ZGBSV` - Banded linear system solve (complex double)
13. `SGBSVX` - Expert banded solver (single)
14. `DGBSVX` - Expert banded solver (double)
15. `CGBSVX` - Expert banded solver (complex single)
16. `ZGBSVX` - Expert banded solver (complex double)

**General Tridiagonal (GTSV family - 8 operations):**
25. `SGTSV` - Tridiagonal linear system solve (single)
26. `DGTSV` - Tridiagonal linear system solve (double)
27. `CGTSV` - Tridiagonal linear system solve (complex single)
28. `ZGTSV` - Tridiagonal linear system solve (complex double)
29. `SGTSVX` - Expert tridiagonal solver (single)
30. `DGTSVX` - Expert tridiagonal solver (double)
31. `CGTSVX` - Expert tridiagonal solver (complex single)
32. `ZGTSVX` - Expert tridiagonal solver (complex double)

**Positive Definite (POSV family - 12 operations):**
33. `SPOSV` - Symmetric positive definite solver (single)
34. `DPOSV` - Symmetric positive definite solver (double)
35. `CPOSV` - Hermitian positive definite solver (complex single)
36. `ZPOSV` - Hermitian positive definite solver (complex double)
37. `SPOSVX` - Expert positive definite solver (single)
38. `DPOSVX` - Expert positive definite solver (double)
39. `CPOSVX` - Expert positive definite solver (complex single)
40. `ZPOSVX` - Expert positive definite solver (complex double)
41. `SPOSVXX` - Extra-precise expert positive definite solver (single)
42. `DPOSVXX` - Extra-precise expert positive definite solver (double)
43. `CPOSVXX` - Extra-precise expert positive definite solver (complex single)
44. `ZPOSVXX` - Extra-precise expert positive definite solver (complex double)

**Positive Definite Packed (PPSV family - 8 operations):**
45. `SPPSV` - Packed positive definite solver (single)
46. `DPPSV` - Packed positive definite solver (double)
47. `CPPSV` - Packed positive definite solver (complex single)
48. `ZPPSV` - Packed positive definite solver (complex double)
49. `SPPSVX` - Expert packed positive definite solver (single)
50. `DPPSVX` - Expert packed positive definite solver (double)
51. `CPPSVX` - Expert packed positive definite solver (complex single)
52. `ZPPSVX` - Expert packed positive definite solver (complex double)

**Positive Definite Banded (PBSV family - 8 operations):**
53. `SPBSV` - Banded positive definite solver (single)
54. `DPBSV` - Banded positive definite solver (double)
55. `CPBSV` - Banded positive definite solver (complex single)
56. `ZPBSV` - Banded positive definite solver (complex double)
57. `SPBSVX` - Expert banded positive definite solver (single)
58. `DPBSVX` - Expert banded positive definite solver (double)
59. `CPBSVX` - Expert banded positive definite solver (complex single)
60. `ZPBSVX` - Expert banded positive definite solver (complex double)

**Positive Definite Tridiagonal (PTSV family - 8 operations):**
61. `SPTSV` - Tridiagonal positive definite solver (single)
62. `DPTSV` - Tridiagonal positive definite solver (double)
63. `CPTSV` - Tridiagonal positive definite solver (complex single)
64. `ZPTSV` - Tridiagonal positive definite solver (complex double)
65. `SPTSVX` - Expert tridiagonal positive definite solver (single)
66. `DPTSVX` - Expert tridiagonal positive definite solver (double)
67. `CPTSVX` - Expert tridiagonal positive definite solver (complex single)
68. `ZPTSVX` - Expert tridiagonal positive definite solver (complex double)

**Symmetric Indefinite (SYSV/HESV family - 14 operations):**
69. `SSYSV` - Symmetric indefinite solver (single)
70. `DSYSV` - Symmetric indefinite solver (double)
71. `CSYSV` - Complex symmetric solver (complex single)
72. `ZSYSV` - Complex symmetric solver (complex double)
73. `SSYSVX` - Expert symmetric solver (single)
74. `DSYSVX` - Expert symmetric solver (double)
75. `CSYSVX` - Expert complex symmetric solver (complex single)
76. `ZSYSVX` - Expert complex symmetric solver (complex double)
77. `SSYSVXX` - Extra-precise expert symmetric solver (single)
78. `DSYSVXX` - Extra-precise expert symmetric solver (double)
79. `CSYSVXX` - Extra-precise expert complex symmetric solver (complex single)
80. `ZSYSVXX` - Extra-precise expert complex symmetric solver (complex double)

**Hermitian Indefinite (HESV family - 6 operations):**
81. `CHESV` - Hermitian indefinite solver (complex single)
82. `ZHESV` - Hermitian indefinite solver (complex double)
83. `CHESVX` - Expert Hermitian solver (complex single)
84. `ZHESVX` - Expert Hermitian solver (complex double)
85. `CHESVXX` - Extra-precise expert Hermitian solver (complex single)
86. `ZHESVXX` - Extra-precise expert Hermitian solver (complex double)

**Symmetric/Hermitian Packed Indefinite (SPSV/HPSV family - 8 operations):**
87. `SSPSV` - Packed symmetric indefinite solver (single)
88. `DSPSV` - Packed symmetric indefinite solver (double)
89. `CSPSV` - Packed complex symmetric solver (complex single)
90. `ZSPSV` - Packed complex symmetric solver (complex double)
91. `CHPSV` - Packed Hermitian solver (complex single)
92. `ZHPSV` - Packed Hermitian solver (complex double)
93. `SSPSVX` - Expert packed symmetric solver (single)
94. `DSPSVX` - Expert packed symmetric solver (double)

#### Least Squares Solvers (16 operations)

**Least Squares (GELS family - 4 operations):**
95. `SGELS` - Overdetermined/underdetermined linear system (single)
96. `DGELS` - Overdetermined/underdetermined linear system (double)
97. `CGELS` - Overdetermined/underdetermined linear system (complex single)
98. `ZGELS` - Overdetermined/underdetermined linear system (complex double)

**Complete Orthogonal Factorization (GELSY family - 4 operations):**
83. `SGELSY` - Least squares using complete orthogonal factorization (single)
84. `DGELSY` - Least squares using complete orthogonal factorization (double)
85. `CGELSY` - Least squares using complete orthogonal factorization (complex single)
86. `ZGELSY` - Least squares using complete orthogonal factorization (complex double)

**SVD-Based Least Squares (GELSS family - 4 operations):**
87. `SGELSS` - Least squares using SVD (single)
88. `DGELSS` - Least squares using SVD (double)
89. `CGELSS` - Least squares using SVD (complex single)
90. `ZGELSS` - Least squares using SVD (complex double)

**Divide-and-Conquer SVD Least Squares (GELSD family - 4 operations):**
91. `SGELSD` - Least squares using divide-and-conquer SVD (single)
92. `DGELSD` - Least squares using divide-and-conquer SVD (double)
93. `CGELSD` - Least squares using divide-and-conquer SVD (complex single)
94. `ZGELSD` - Least squares using divide-and-conquer SVD (complex double)

#### Generalized Linear Least Squares (8 operations)

**Generalized LSE (GGLSE family - 4 operations):**
95. `SGGLSE` - Linear equality-constrained least squares (single)
96. `DGGLSE` - Linear equality-constrained least squares (double)
97. `CGGLSE` - Linear equality-constrained least squares (complex single)
98. `ZGGLSE` - Linear equality-constrained least squares (complex double)

**Generalized GLM (GGGLM family - 4 operations):**
99. `SGGGLM` - Generalized linear model (single)
100. `DGGGLM` - Generalized linear model (double)
101. `CGGGLM` - Generalized linear model (complex single)
102. `ZGGGLM` - Generalized linear model (complex double)

#### Symmetric/Hermitian Eigenvalue Solvers (52 operations)

**Full Dense (SYEV/HEEV families - 16 operations):**
103. `SSYEV` - All eigenvalues (single)
104. `DSYEV` - All eigenvalues (double)
105. `CHEEV` - All eigenvalues (complex single)
106. `ZHEEV` - All eigenvalues (complex double)
107. `SSYEVX` - Selected eigenvalues (single)
108. `DSYEVX` - Selected eigenvalues (double)
109. `CHEEVX` - Selected eigenvalues (complex single)
110. `ZHEEVX` - Selected eigenvalues (complex double)
111. `SSYEVD` - Divide-and-conquer (single)
112. `DSYEVD` - Divide-and-conquer (double)
113. `CHEEVD` - Divide-and-conquer (complex single)
114. `ZHEEVD` - Divide-and-conquer (complex double)
115. `SSYEVR` - RRR algorithm (single)
116. `DSYEVR` - RRR algorithm (double)
117. `CHEEVR` - RRR algorithm (complex single)
118. `ZHEEVR` - RRR algorithm (complex double)

**Packed Storage (SPEV/HPEV families - 12 operations):**
119. `SSPEV` - Packed all eigenvalues (single)
120. `DSPEV` - Packed all eigenvalues (double)
121. `CHPEV` - Packed all eigenvalues (complex single)
122. `ZHPEV` - Packed all eigenvalues (complex double)
123. `SSPEVX` - Packed selected eigenvalues (single)
124. `DSPEVX` - Packed selected eigenvalues (double)
125. `CHPEVX` - Packed selected eigenvalues (complex single)
126. `ZHPEVX` - Packed selected eigenvalues (complex double)
127. `SSPEVD` - Packed divide-and-conquer (single)
128. `DSPEVD` - Packed divide-and-conquer (double)
129. `CHPEVD` - Packed divide-and-conquer (complex single)
130. `ZHPEVD` - Packed divide-and-conquer (complex double)

**Banded Storage (SBEV/HBEV families - 12 operations):**
129. `SSBEV` - Banded all eigenvalues (single)
130. `DSBEV` - Banded all eigenvalues (double)
131. `CHBEV` - Banded all eigenvalues (complex single)
132. `ZHBEV` - Banded all eigenvalues (complex double)
133. `SSBEVX` - Banded selected eigenvalues (single)
134. `DSBEVX` - Banded selected eigenvalues (double)
135. `CHBEVX` - Banded selected eigenvalues (complex single)
136. `ZHBEVX` - Banded selected eigenvalues (complex double)
137. `SSBEVD` - Banded divide-and-conquer (single)
138. `DSBEVD` - Banded divide-and-conquer (double)
139. `CHBEVD` - Banded divide-and-conquer (complex single)
140. `ZHBEVD` - Banded divide-and-conquer (complex double)

**Tridiagonal (STEV family - 8 operations):**
141. `SSTEV` - Tridiagonal all eigenvalues (single)
142. `DSTEV` - Tridiagonal all eigenvalues (double)
143. `SSTEVX` - Tridiagonal selected eigenvalues (single)
144. `DSTEVX` - Tridiagonal selected eigenvalues (double)
145. `SSTEVD` - Tridiagonal divide-and-conquer (single)
146. `DSTEVD` - Tridiagonal divide-and-conquer (double)
147. `SSTEVR` - Tridiagonal RRR algorithm (single)
148. `DSTEVR` - Tridiagonal RRR algorithm (double)

**Symmetric Tridiagonal Additional (4 operations):**
149. `SSTEQR` - Symmetric tridiagonal eigenvalues by QR (single)
150. `DSTEQR` - Symmetric tridiagonal eigenvalues by QR (double)
151. `CSTEQR` - Symmetric tridiagonal eigenvalues by QR (complex single)
152. `ZSTEQR` - Symmetric tridiagonal eigenvalues by QR (complex double)

#### Non-Symmetric Eigenvalue Solvers (16 operations)

**Schur Factorization (GEES family - 4 operations):**
153. `SGEES` - Schur factorization (single)
154. `DGEES` - Schur factorization (double)
155. `CGEES` - Schur factorization (complex single)
156. `ZGEES` - Schur factorization (complex double)

**Schur with Condition (GEESX family - 4 operations):**
157. `SGEESX` - Schur with condition estimates (single)
158. `DGEESX` - Schur with condition estimates (double)
159. `CGEESX` - Schur with condition estimates (complex single)
160. `ZGEESX` - Schur with condition estimates (complex double)

**Eigenvalues and Eigenvectors (GEEV family - 4 operations):**
161. `SGEEV` - Eigenvalues and eigenvectors (single)
162. `DGEEV` - Eigenvalues and eigenvectors (double)
163. `CGEEV` - Eigenvalues and eigenvectors (complex single)
164. `ZGEEV` - Eigenvalues and eigenvectors (complex double)

**Eigenvalues with Condition (GEEVX family - 4 operations):**
165. `SGEEVX` - Eigenvalues with condition estimates (single)
166. `DGEEVX` - Eigenvalues with condition estimates (double)
167. `CGEEVX` - Eigenvalues with condition estimates (complex single)
168. `ZGEEVX` - Eigenvalues with condition estimates (complex double)

#### Singular Value Decomposition (8 operations)

**SVD Standard (GESVD family - 4 operations):**
169. `SGESVD` - Singular value decomposition (single)
170. `DGESVD` - Singular value decomposition (double)
171. `CGESVD` - Singular value decomposition (complex single)
172. `ZGESVD` - Singular value decomposition (complex double)

**SVD Divide-and-Conquer (GESDD family - 4 operations):**
173. `SGESDD` - SVD using divide-and-conquer (single)
174. `DGESDD` - SVD using divide-and-conquer (double)
175. `CGESDD` - SVD using divide-and-conquer (complex single)
176. `ZGESDD` - SVD using divide-and-conquer (complex double)

#### Generalized Symmetric Definite Eigenvalue (36 operations)

**Full Dense (SYGV/HEGV families - 12 operations):**
177. `SSYGV` - Generalized symmetric eigenvalue (single)
178. `DSYGV` - Generalized symmetric eigenvalue (double)
179. `CHEGV` - Generalized Hermitian eigenvalue (complex single)
180. `ZHEGV` - Generalized Hermitian eigenvalue (complex double)
181. `SSYGVX` - Generalized selective eigenvalues (single)
182. `DSYGVX` - Generalized selective eigenvalues (double)
183. `CHEGVX` - Generalized selective eigenvalues (complex single)
184. `ZHEGVX` - Generalized selective eigenvalues (complex double)
185. `SSYGVD` - Generalized divide-and-conquer (single)
186. `DSYGVD` - Generalized divide-and-conquer (double)
187. `CHEGVD` - Generalized divide-and-conquer (complex single)
188. `ZHEGVD` - Generalized divide-and-conquer (complex double)

**Packed Storage (SPGV/HPGV families - 12 operations):**
189. `SSPGV` - Generalized packed eigenvalue (single)
190. `DSPGV` - Generalized packed eigenvalue (double)
191. `CHPGV` - Generalized packed eigenvalue (complex single)
192. `ZHPGV` - Generalized packed eigenvalue (complex double)
193. `SSPGVX` - Generalized packed selective (single)
194. `DSPGVX` - Generalized packed selective (double)
195. `CHPGVX` - Generalized packed selective (complex single)
196. `ZHPGVX` - Generalized packed selective (complex double)
197. `SSPGVD` - Generalized packed divide-and-conquer (single)
198. `DSPGVD` - Generalized packed divide-and-conquer (double)
199. `CHPGVD` - Generalized packed divide-and-conquer (complex single)
200. `ZHPGVD` - Generalized packed divide-and-conquer (complex double)

**Banded Storage (SBGV/HBGV families - 12 operations):**
201. `SSBGV` - Generalized banded eigenvalue (single)
202. `DSBGV` - Generalized banded eigenvalue (double)
203. `CHBGV` - Generalized banded eigenvalue (complex single)
204. `ZHBGV` - Generalized banded eigenvalue (complex double)
205. `SSBGVX` - Generalized banded selective (single)
206. `DSBGVX` - Generalized banded selective (double)
207. `CHBGVX` - Generalized banded selective (complex single)
208. `ZHBGVX` - Generalized banded selective (complex double)
209. `SSBGVD` - Generalized banded divide-and-conquer (single)
210. `DSBGVD` - Generalized banded divide-and-conquer (double)
211. `CHBGVD` - Generalized banded divide-and-conquer (complex single)
212. `ZHBGVD` - Generalized banded divide-and-conquer (complex double)

**Complex Symmetric Generalized Eigenvalue (6 operations):**
213. `CSYGV` - Generalized complex symmetric eigenvalue (complex single)
214. `ZSYGV` - Generalized complex symmetric eigenvalue (complex double)
215. `CSYGVD` - Generalized complex symmetric divide-and-conquer (complex single)
216. `ZSYGVD` - Generalized complex symmetric divide-and-conquer (complex double)
217. `CSYGVX` - Generalized complex symmetric selective (complex single)
218. `ZSYGVX` - Generalized complex symmetric selective (complex double)

#### Generalized Non-Symmetric Eigenvalue (16 operations)

**Generalized Schur (GGES family - 4 operations):**
219. `SGGES` - Generalized Schur factorization (single)
220. `DGGES` - Generalized Schur factorization (double)
221. `CGGES` - Generalized Schur factorization (complex single)
222. `ZGGES` - Generalized Schur factorization (complex double)

**Generalized Schur with Condition (GGESX family - 4 operations):**
223. `SGGESX` - Generalized Schur with condition (single)
224. `DGGESX` - Generalized Schur with condition (double)
225. `CGGESX` - Generalized Schur with condition (complex single)
226. `ZGGESX` - Generalized Schur with condition (complex double)

**Generalized Eigenvalue (GGEV family - 4 operations):**
227. `SGGEV` - Generalized eigenvalues and eigenvectors (single)
228. `DGGEV` - Generalized eigenvalues and eigenvectors (double)
229. `CGGEV` - Generalized eigenvalues and eigenvectors (complex single)
230. `ZGGEV` - Generalized eigenvalues and eigenvectors (complex double)

**Generalized Eigenvalue with Condition (GGEVX family - 4 operations):**
231. `SGGEVX` - Generalized eigenvalues with condition (single)
232. `DGGEVX` - Generalized eigenvalues with condition (double)
233. `CGGEVX` - Generalized eigenvalues with condition (complex single)
234. `ZGGEVX` - Generalized eigenvalues with condition (complex double)

#### Generalized Singular Value Decomposition (4 operations)

**Generalized SVD (GGSVD family - 4 operations):**
235. `SGGSVD` - Generalized SVD (single)
236. `DGGSVD` - Generalized SVD (double)
237. `CGGSVD` - Generalized SVD (complex single)
238. `ZGGSVD` - Generalized SVD (complex double)

#### SVD Support/Decomposition (8 operations)

**Bidiagonal SVD Divide-and-Conquer (BDSDC family - 4 operations):**
239. `SBDSDC` - Bidiagonal SVD divide-and-conquer (single)
240. `DBDSDC` - Bidiagonal SVD divide-and-conquer (double)
241. `CBDSDC` - Bidiagonal SVD divide-and-conquer (complex single)
242. `ZBDSDC` - Bidiagonal SVD divide-and-conquer (complex double)

**Hermitian Tridiagonal Reduction (4 operations):**
243. `SHETD2` - Hermitian tridiagonal reduction (single)
244. `DHETD2` - Hermitian tridiagonal reduction (double)

**Note**: Operations 243-244 are reduction operations that are also listed as driver routines but primarily used internally.

**DRIVER ROUTINES TOTAL: 264 operations**
Note: Includes standard LAPACK drivers (172) plus extended operations found in reference implementation (74).

---

### LAPACK COMPUTATIONAL ROUTINES (840 operations)

**Complete Enumerated List:**

#### Linear Equations - General Matrices (24 operations)

1. `SGETRF` - LU factorization (single)
2. `DGETRF` - LU factorization (double)
3. `CGETRF` - LU factorization (complex single)
4. `ZGETRF` - LU factorization (complex double)
5. `SGETRS` - Solve using LU factorization (single)
6. `DGETRS` - Solve using LU factorization (double)
7. `CGETRS` - Solve using LU factorization (complex single)
8. `ZGETRS` - Solve using LU factorization (complex double)
9. `SGECON` - Estimate condition number (single)
10. `DGECON` - Estimate condition number (double)
11. `CGECON` - Estimate condition number (complex single)
12. `ZGECON` - Estimate condition number (complex double)
13. `SGERFS` - Refine solution (single)
14. `DGERFS` - Refine solution (double)
15. `CGERFS` - Refine solution (complex single)
16. `ZGERFS` - Refine solution (complex double)
17. `SGETRI` - Invert using LU factorization (single)
18. `DGETRI` - Invert using LU factorization (double)
19. `CGETRI` - Invert using LU factorization (complex single)
20. `ZGETRI` - Invert using LU factorization (complex double)
21. `SGEEQU` - Equilibrate matrix (single)
22. `DGEEQU` - Equilibrate matrix (double)
23. `CGEEQU` - Equilibrate matrix (complex single)
24. `ZGEEQU` - Equilibrate matrix (complex double)

#### Linear Equations - General Banded (20 operations)

1. `SGBTRF` - LU factorization banded (single)
2. `DGBTRF` - LU factorization banded (double)
3. `CGBTRF` - LU factorization banded (complex single)
4. `ZGBTRF` - LU factorization banded (complex double)
5. `SGBTRS` - Solve using banded LU (single)
6. `DGBTRS` - Solve using banded LU (double)
7. `CGBTRS` - Solve using banded LU (complex single)
8. `ZGBTRS` - Solve using banded LU (complex double)
9. `SGBCON` - Condition number banded (single)
10. `DGBCON` - Condition number banded (double)
11. `CGBCON` - Condition number banded (complex single)
12. `ZGBCON` - Condition number banded (complex double)
13. `SGBRFS` - Refine solution banded (single)
14. `DGBRFS` - Refine solution banded (double)
15. `CGBRFS` - Refine solution banded (complex single)
16. `ZGBRFS` - Refine solution banded (complex double)
17. `SGBEQU` - Equilibrate banded (single)
18. `DGBEQU` - Equilibrate banded (double)
19. `CGBEQU` - Equilibrate banded (complex single)
20. `ZGBEQU` - Equilibrate banded (complex double)

#### Linear Equations - General Tridiagonal (16 operations)

1. `SGTTRF` - LU factorization tridiagonal (single)
2. `DGTTRF` - LU factorization tridiagonal (double)
3. `CGTTRF` - LU factorization tridiagonal (complex single)
4. `ZGTTRF` - LU factorization tridiagonal (complex double)
5. `SGTTRS` - Solve tridiagonal (single)
6. `DGTTRS` - Solve tridiagonal (double)
7. `CGTTRS` - Solve tridiagonal (complex single)
8. `ZGTTRS` - Solve tridiagonal (complex double)
9. `SGTCON` - Condition number tridiagonal (single)
10. `DGTCON` - Condition number tridiagonal (double)
11. `CGTCON` - Condition number tridiagonal (complex single)
12. `ZGTCON` - Condition number tridiagonal (complex double)
13. `SGTRFS` - Refine tridiagonal (single)
14. `DGTRFS` - Refine tridiagonal (double)
15. `CGTRFS` - Refine tridiagonal (complex single)
16. `ZGTRFS` - Refine tridiagonal (complex double)

#### Linear Equations - Positive Definite (24 operations)

1. `SPOTRF` - Cholesky factorization (single)
2. `DPOTRF` - Cholesky factorization (double)
3. `CPOTRF` - Cholesky factorization (complex single)
4. `ZPOTRF` - Cholesky factorization (complex double)
5. `SPOTRS` - Solve using Cholesky (single)
6. `DPOTRS` - Solve using Cholesky (double)
7. `CPOTRS` - Solve using Cholesky (complex single)
8. `ZPOTRS` - Solve using Cholesky (complex double)
9. `SPOTRI` - Invert using Cholesky (single)
10. `DPOTRI` - Invert using Cholesky (double)
11. `CPOTRI` - Invert using Cholesky (complex single)
12. `ZPOTRI` - Invert using Cholesky (complex double)
13. `SPOCON` - Condition number positive definite (single)
14. `DPOCON` - Condition number positive definite (double)
15. `CPOCON` - Condition number positive definite (complex single)
16. `ZPOCON` - Condition number positive definite (complex double)
17. `SPORFS` - Refine positive definite (single)
18. `DPORFS` - Refine positive definite (double)
19. `CPORFS` - Refine positive definite (complex single)
20. `ZPORFS` - Refine positive definite (complex double)
21. `SPOEQU` - Equilibrate positive definite (single)
22. `DPOEQU` - Equilibrate positive definite (double)
23. `CPOEQU` - Equilibrate positive definite (complex single)
24. `ZPOEQU` - Equilibrate positive definite (complex double)

#### Linear Equations - Positive Definite Packed (24 operations)

1. `SPPTRF` - Cholesky packed (single)
2. `DPPTRF` - Cholesky packed (double)
3. `CPPTRF` - Cholesky packed (complex single)
4. `ZPPTRF` - Cholesky packed (complex double)
5. `SPPTRS` - Solve packed (single)
6. `DPPTRS` - Solve packed (double)
7. `CPPTRS` - Solve packed (complex single)
8. `ZPPTRS` - Solve packed (complex double)
9. `SPPTRI` - Invert packed (single)
10. `DPPTRI` - Invert packed (double)
11. `CPPTRI` - Invert packed (complex single)
12. `ZPPTRI` - Invert packed (complex double)
13. `SPPCON` - Condition packed (single)
14. `DPPCON` - Condition packed (double)
15. `CPPCON` - Condition packed (complex single)
16. `ZPPCON` - Condition packed (complex double)
17. `SPPRFS` - Refine packed (single)
18. `DPPRFS` - Refine packed (double)
19. `CPPRFS` - Refine packed (complex single)
20. `ZPPRFS` - Refine packed (complex double)
21. `SPPEQU` - Equilibrate packed (single)
22. `DPPEQU` - Equilibrate packed (double)
23. `CPPEQU` - Equilibrate packed (complex single)
24. `ZPPEQU` - Equilibrate packed (complex double)

#### Linear Equations - Positive Definite Banded (20 operations)

1. `SPBTRF` - Cholesky banded (single)
2. `DPBTRF` - Cholesky banded (double)
3. `CPBTRF` - Cholesky banded (complex single)
4. `ZPBTRF` - Cholesky banded (complex double)
5. `SPBTRS` - Solve banded (single)
6. `DPBTRS` - Solve banded (double)
7. `CPBTRS` - Solve banded (complex single)
8. `ZPBTRS` - Solve banded (complex double)
9. `SPBCON` - Condition banded (single)
10. `DPBCON` - Condition banded (double)
11. `CPBCON` - Condition banded (complex single)
12. `ZPBCON` - Condition banded (complex double)
13. `SPBRFS` - Refine banded (single)
14. `DPBRFS` - Refine banded (double)
15. `CPBRFS` - Refine banded (complex single)
16. `ZPBRFS` - Refine banded (complex double)
17. `SPBEQU` - Equilibrate banded (single)
18. `DPBEQU` - Equilibrate banded (double)
19. `CPBEQU` - Equilibrate banded (complex single)
20. `ZPBEQU` - Equilibrate banded (complex double)

#### Linear Equations - Positive Definite Tridiagonal (16 operations)

1. `SPTTRF` - Cholesky tridiagonal (single)
2. `DPTTRF` - Cholesky tridiagonal (double)
3. `CPTTRF` - Cholesky tridiagonal (complex single)
4. `ZPTTRF` - Cholesky tridiagonal (complex double)
5. `SPTTRS` - Solve tridiagonal (single)
6. `DPTTRS` - Solve tridiagonal (double)
7. `CPTTRS` - Solve tridiagonal (complex single)
8. `ZPTTRS` - Solve tridiagonal (complex double)
9. `SPTCON` - Condition tridiagonal (single)
10. `DPTCON` - Condition tridiagonal (double)
11. `CPTCON` - Condition tridiagonal (complex single)
12. `ZPTCON` - Condition tridiagonal (complex double)
13. `SPTRFS` - Refine tridiagonal (single)
14. `DPTRFS` - Refine tridiagonal (double)
15. `CPTRFS` - Refine tridiagonal (complex single)
16. `ZPTRFS` - Refine tridiagonal (complex double)

#### Linear Equations - Symmetric/Hermitian Indefinite (33 operations)

1. `SSYTRF` - Bunch-Kaufman factorization symmetric (single)
2. `DSYTRF` - Bunch-Kaufman factorization symmetric (double)
3. `CSYTRF` - Bunch-Kaufman factorization complex symmetric (complex single)
4. `ZSYTRF` - Bunch-Kaufman factorization complex symmetric (complex double)
5. `CHETRF` - Bunch-Kaufman factorization Hermitian (complex single)
6. `ZHETRF` - Bunch-Kaufman factorization Hermitian (complex double)
7. `SSYTRS` - Solve symmetric (single)
8. `DSYTRS` - Solve symmetric (double)
9. `CSYTRS` - Solve complex symmetric (complex single)
10. `ZSYTRS` - Solve complex symmetric (complex double)
11. `CHETRS` - Solve Hermitian (complex single)
12. `ZHETRS` - Solve Hermitian (complex double)
13. `SSYTRI` - Invert symmetric (single)
14. `DSYTRI` - Invert symmetric (double)
15. `CSYTRI` - Invert complex symmetric (complex single)
16. `ZSYTRI` - Invert complex symmetric (complex double)
17. `CHETRI` - Invert Hermitian (complex single)
18. `ZHETRI` - Invert Hermitian (complex double)
19. `SSYCON` - Condition symmetric (single)
20. `DSYCON` - Condition symmetric (double)
21. `CSYCON` - Condition complex symmetric (complex single)
22. `ZSYCON` - Condition complex symmetric (complex double)
23. `CHECON` - Condition Hermitian (complex single)
24. `ZHECON` - Condition Hermitian (complex double)
25. `SSYRFS` - Refine symmetric (single)
26. `DSYRFS` - Refine symmetric (double)
27. `CSYRFS` - Refine complex symmetric (complex single)
28. `ZSYRFS` - Refine complex symmetric (complex double)
29. `CHERFS` - Refine Hermitian (complex single)
30. `ZHERFS` - Refine Hermitian (complex double)
31. `SSYTRI2` - Invert symmetric variant (single)
32. `DSYTRI2` - Invert symmetric variant (double)
33. `CSYTRI2` - Invert complex symmetric variant (complex single)

#### Linear Equations - Symmetric/Hermitian Indefinite Packed (37 operations)

1. `SSPTRF` - Factor symmetric packed (single)
2. `DSPTRF` - Factor symmetric packed (double)
3. `CSPTRF` - Factor complex symmetric packed (complex single)
4. `ZSPTRF` - Factor complex symmetric packed (complex double)
5. `CHPTRF` - Factor Hermitian packed (complex single)
6. `ZHPTRF` - Factor Hermitian packed (complex double)
7. `SSPTRS` - Solve symmetric packed (single)
8. `DSPTRS` - Solve symmetric packed (double)
9. `CSPTRS` - Solve complex symmetric packed (complex single)
10. `ZSPTRS` - Solve complex symmetric packed (complex double)
11. `CHPTRS` - Solve Hermitian packed (complex single)
12. `ZHPTRS` - Solve Hermitian packed (complex double)
13. `SSPTRI` - Invert symmetric packed (single)
14. `DSPTRI` - Invert symmetric packed (double)
15. `CSPTRI` - Invert complex symmetric packed (complex single)
16. `ZSPTRI` - Invert complex symmetric packed (complex double)
17. `CHPTRI` - Invert Hermitian packed (complex single)
18. `ZHPTRI` - Invert Hermitian packed (complex double)
19. `SSPCON` - Condition symmetric packed (single)
20. `DSPCON` - Condition symmetric packed (double)
21. `CSPCON` - Condition complex symmetric packed (complex single)
22. `ZSPCON` - Condition complex symmetric packed (complex double)
23. `CHPCON` - Condition Hermitian packed (complex single)
24. `ZHPCON` - Condition Hermitian packed (complex double)
25. `SSPRFS` - Refine symmetric packed (single)
26. `DSPRFS` - Refine symmetric packed (double)
27. `CSPRFS` - Refine complex symmetric packed (complex single)
28. `ZSPRFS` - Refine complex symmetric packed (complex double)
29. `CHPRFS` - Refine Hermitian packed (complex single)
30. `ZHPRFS` - Refine Hermitian packed (complex double)
31. `SSYTRF_ROOK` - Rook pivoting symmetric (single)
32. `DSYTRF_ROOK` - Rook pivoting symmetric (double)
33. `CSYTRF_ROOK` - Rook pivoting complex symmetric (complex single)
34. `ZSYTRF_ROOK` - Rook pivoting complex symmetric (complex double)
35. `CHETRF_ROOK` - Rook pivoting Hermitian (complex single)
36. `ZHETRF_ROOK` - Rook pivoting Hermitian (complex double)
37. `SSYTRF_AA` - Aasen's algorithm (single)

#### Linear Equations - Symmetric/Hermitian Indefinite Banded (4 operations)

1. `SSBTRF` - Factor symmetric banded (single)
2. `DSBTRF` - Factor symmetric banded (double)
3. `CHBTRF` - Factor Hermitian banded (complex single)
4. `ZHBTRF` - Factor Hermitian banded (complex double)

#### Linear Equations - Triangular (16 operations)

1. `STRTRI` - Invert triangular (single)
2. `DTRTRI` - Invert triangular (double)
3. `CTRTRI` - Invert triangular (complex single)
4. `ZTRTRI` - Invert triangular (complex double)
5. `STRTRS` - Solve triangular (single)
6. `DTRTRS` - Solve triangular (double)
7. `CTRTRS` - Solve triangular (complex single)
8. `ZTRTRS` - Solve triangular (complex double)
9. `STRCON` - Condition triangular (single)
10. `DTRCON` - Condition triangular (double)
11. `CTRCON` - Condition triangular (complex single)
12. `ZTRCON` - Condition triangular (complex double)
13. `STRRFS` - Refine triangular (single)
14. `DTRRFS` - Refine triangular (double)
15. `CTRRFS` - Refine triangular (complex single)
16. `ZTRRFS` - Refine triangular (complex double)

#### Linear Equations - Triangular Packed (16 operations)

1. `STPTRI` - Invert triangular packed (single)
2. `DTPTRI` - Invert triangular packed (double)
3. `CTPTRI` - Invert triangular packed (complex single)
4. `ZTPTRI` - Invert triangular packed (complex double)
5. `STPTRS` - Solve triangular packed (single)
6. `DTPTRS` - Solve triangular packed (double)
7. `CTPTRS` - Solve triangular packed (complex single)
8. `ZTPTRS` - Solve triangular packed (complex double)
9. `STPCON` - Condition triangular packed (single)
10. `DTPCON` - Condition triangular packed (double)
11. `CTPCON` - Condition triangular packed (complex single)
12. `ZTPCON` - Condition triangular packed (complex double)
13. `STPRFS` - Refine triangular packed (single)
14. `DTPRFS` - Refine triangular packed (double)
15. `CTPRFS` - Refine triangular packed (complex single)
16. `ZTPRFS` - Refine triangular packed (complex double)

#### Linear Equations - Triangular Banded (12 operations)

1. `STBTRS` - Solve triangular banded (single)
2. `DTBTRS` - Solve triangular banded (double)
3. `CTBTRS` - Solve triangular banded (complex single)
4. `ZTBTRS` - Solve triangular banded (complex double)
5. `STBCON` - Condition triangular banded (single)
6. `DTBCON` - Condition triangular banded (double)
7. `CTBCON` - Condition triangular banded (complex single)
8. `ZTBCON` - Condition triangular banded (complex double)
9. `STBRFS` - Refine triangular banded (single)
10. `DTBRFS` - Refine triangular banded (double)
11. `CTBRFS` - Refine triangular banded (complex single)
12. `ZTBRFS` - Refine triangular banded (complex double)

#### QR Factorization (20 operations)

1. `SGEQRF` - QR factorization (single)
2. `DGEQRF` - QR factorization (double)
3. `CGEQRF` - QR factorization (complex single)
4. `ZGEQRF` - QR factorization (complex double)
5. `SORGQR` - Generate orthogonal Q (single)
6. `DORGQR` - Generate orthogonal Q (double)
7. `CUNGQR` - Generate unitary Q (complex single)
8. `ZUNGQR` - Generate unitary Q (complex double)
9. `SORMQR` - Multiply by orthogonal Q (single)
10. `DORMQR` - Multiply by orthogonal Q (double)
11. `CUNMQR` - Multiply by unitary Q (complex single)
12. `ZUNMQR` - Multiply by unitary Q (complex double)
13. `SGEQP3` - QR with column pivoting (single)
14. `DGEQP3` - QR with column pivoting (double)
15. `CGEQP3` - QR with column pivoting (complex single)
16. `ZGEQP3` - QR with column pivoting (complex double)
17. `SGEQPF` - QR with pivoting deprecated (single)
18. `DGEQPF` - QR with pivoting deprecated (double)
19. `CGEQPF` - QR with pivoting deprecated (complex single)
20. `ZGEQPF` - QR with pivoting deprecated (complex double)

#### LQ Factorization (20 operations)

1. `SGELQF` - LQ factorization (single)
2. `DGELQF` - LQ factorization (double)
3. `CGELQF` - LQ factorization (complex single)
4. `ZGELQF` - LQ factorization (complex double)
5. `SORGLQ` - Generate orthogonal Q (single)
6. `DORGLQ` - Generate orthogonal Q (double)
7. `CUNGLQ` - Generate unitary Q (complex single)
8. `ZUNGLQ` - Generate unitary Q (complex double)
9. `SORMLQ` - Multiply by orthogonal Q (single)
10. `DORMLQ` - Multiply by orthogonal Q (double)
11. `CUNMLQ` - Multiply by unitary Q (complex single)
12. `ZUNMLQ` - Multiply by unitary Q (complex double)
13. `SGELQ2` - LQ unblocked (single)
14. `DGELQ2` - LQ unblocked (double)
15. `CGELQ2` - LQ unblocked (complex single)
16. `ZGELQ2` - LQ unblocked (complex double)
17. `SORGL2` - Generate Q unblocked (single)
18. `DORGL2` - Generate Q unblocked (double)
19. `CUNGL2` - Generate Q unblocked (complex single)
20. `ZUNGL2` - Generate Q unblocked (complex double)

#### RQ Factorization (20 operations)

1. `SGERQF` - RQ factorization (single)
2. `DGERQF` - RQ factorization (double)
3. `CGERQF` - RQ factorization (complex single)
4. `ZGERQF` - RQ factorization (complex double)
5. `SORGRQ` - Generate orthogonal Q (single)
6. `DORGRQ` - Generate orthogonal Q (double)
7. `CUNGRQ` - Generate unitary Q (complex single)
8. `ZUNGRQ` - Generate unitary Q (complex double)
9. `SORMRQ` - Multiply by orthogonal Q (single)
10. `DORMRQ` - Multiply by orthogonal Q (double)
11. `CUNMRQ` - Multiply by unitary Q (complex single)
12. `ZUNMRQ` - Multiply by unitary Q (complex double)
13. `SGERQ2` - RQ unblocked (single)
14. `DGERQ2` - RQ unblocked (double)
15. `CGERQ2` - RQ unblocked (complex single)
16. `ZGERQ2` - RQ unblocked (complex double)
17. `SORGR2` - Generate Q unblocked (single)
18. `DORGR2` - Generate Q unblocked (double)
19. `CUNGR2` - Generate Q unblocked (complex single)
20. `ZUNGR2` - Generate Q unblocked (complex double)

#### QL Factorization (20 operations)

1. `SGEQLF` - QL factorization (single)
2. `DGEQLF` - QL factorization (double)
3. `CGEQLF` - QL factorization (complex single)
4. `ZGEQLF` - QL factorization (complex double)
5. `SORGQL` - Generate orthogonal Q (single)
6. `DORGQL` - Generate orthogonal Q (double)
7. `CUNGQL` - Generate unitary Q (complex single)
8. `ZUNGQL` - Generate unitary Q (complex double)
9. `SORMQL` - Multiply by orthogonal Q (single)
10. `DORMQL` - Multiply by orthogonal Q (double)
11. `CUNMQL` - Multiply by unitary Q (complex single)
12. `ZUNMQL` - Multiply by unitary Q (complex double)
13. `SGEQL2` - QL unblocked (single)
14. `DGEQL2` - QL unblocked (double)
15. `CGEQL2` - QL unblocked (complex single)
16. `ZGEQL2` - QL unblocked (complex double)
17. `SORG2L` - Generate Q unblocked (single)
18. `DORG2L` - Generate Q unblocked (double)
19. `CUNG2L` - Generate Q unblocked (complex single)
20. `ZUNG2L` - Generate Q unblocked (complex double)

#### RZ Factorization (12 operations)

1. `STZRZF` - RZ factorization (single)
2. `DTZRZF` - RZ factorization (double)
3. `CTZRZF` - RZ factorization (complex single)
4. `ZTZRZF` - RZ factorization (complex double)
5. `SORMRZ` - Multiply by Z (single)
6. `DORMRZ` - Multiply by Z (double)
7. `CUNMRZ` - Multiply by Z (complex single)
8. `ZUNMRZ` - Multiply by Z (complex double)
9. `SORMR3` - Multiply by R (single)
10. `DORMR3` - Multiply by R (double)
11. `CUNMR3` - Multiply by R (complex single)
12. `ZUNMR3` - Multiply by R (complex double)

#### Symmetric/Hermitian Eigenvalue - Reduction (6 operations)

1. `SSYTRD` - Reduce to tridiagonal symmetric (single)
2. `DSYTRD` - Reduce to tridiagonal symmetric (double)
3. `CSYTRD` - Reduce to tridiagonal complex symmetric (complex single)
4. `ZSYTRD` - Reduce to tridiagonal complex symmetric (complex double)
5. `CHETRD` - Reduce to tridiagonal Hermitian (complex single)
6. `ZHETRD` - Reduce to tridiagonal Hermitian (complex double)

#### Symmetric/Hermitian Eigenvalue - Tridiagonal Solver (27 operations)

1. `SSTEQR` - Eigenvalues via QR (single)
2. `DSTEQR` - Eigenvalues via QR (double)
3. `CSTEQR` - Eigenvalues via QR (complex single)
4. `ZSTEQR` - Eigenvalues via QR (complex double)
5. `SSTEDC` - Eigenvalues divide-and-conquer (single)
6. `DSTEDC` - Eigenvalues divide-and-conquer (double)
7. `CSTEDC` - Eigenvalues divide-and-conquer (complex single)
8. `ZSTEDC` - Eigenvalues divide-and-conquer (complex double)
9. `SSTEGR` - Eigenvalues RRR algorithm (single)
10. `DSTEGR` - Eigenvalues RRR algorithm (double)
11. `CSTEGR` - Eigenvalues RRR algorithm (complex single)
12. `ZSTEGR` - Eigenvalues RRR algorithm (complex double)
13. `SSTEIN` - Eigenvectors by inverse iteration (single)
14. `DSTEIN` - Eigenvectors by inverse iteration (double)
15. `CSTEIN` - Eigenvectors by inverse iteration (complex single)
16. `ZSTEIN` - Eigenvectors by inverse iteration (complex double)
17. `SSTEVD` - Symmetric tridiagonal eigenvalues (single)
18. `DSTEVD` - Symmetric tridiagonal eigenvalues (double)
19. `SSTEVR` - Symmetric tridiagonal RRR (single)
20. `DSTEVR` - Symmetric tridiagonal RRR (double)
21. `SSTEVX` - Symmetric tridiagonal selected (single)
22. `DSTEVX` - Symmetric tridiagonal selected (double)
23. `SSTERF` - Eigenvalues root-free QR (single)
24. `DSTERF` - Eigenvalues root-free QR (double)
25. `SSTEBZ` - Eigenvalues by bisection (single)
26. `DSTEBZ` - Eigenvalues by bisection (double)
27. `SPTEQR` - Positive definite tridiagonal (single)
28. `DPTEQR` - Positive definite tridiagonal (double)
29. `CPTEQR` - Positive definite tridiagonal (complex single)
30. `ZPTEQR` - Positive definite tridiagonal (complex double)

#### Non-Symmetric Eigenvalue - Hessenberg Reduction (20 operations)

1. `SGEHRD` - Reduce to Hessenberg (single)
2. `DGEHRD` - Reduce to Hessenberg (double)
3. `CGEHRD` - Reduce to Hessenberg (complex single)
4. `ZGEHRD` - Reduce to Hessenberg (complex double)
5. `SORGHR` - Generate orthogonal Q (single)
6. `DORGHR` - Generate orthogonal Q (double)
7. `CUNGHR` - Generate unitary Q (complex single)
8. `ZUNGHR` - Generate unitary Q (complex double)
9. `SORMHR` - Multiply by orthogonal Q (single)
10. `DORMHR` - Multiply by orthogonal Q (double)
11. `CUNMHR` - Multiply by unitary Q (complex single)
12. `ZUNMHR` - Multiply by unitary Q (complex double)
13. `SGEHD2` - Hessenberg unblocked (single)
14. `DGEHD2` - Hessenberg unblocked (double)
15. `CGEHD2` - Hessenberg unblocked (complex single)
16. `ZGEHD2` - Hessenberg unblocked (complex double)
17. `SGEBD2` - Bidiagonal reduction (single)
18. `DGEBD2` - Bidiagonal reduction (double)
19. `CGEBD2` - Bidiagonal reduction (complex single)
20. `ZGEBD2` - Bidiagonal reduction (complex double)

#### Non-Symmetric Eigenvalue - Schur Form (12 operations)

1. `SHSEQR` - Schur factorization (single)
2. `DHSEQR` - Schur factorization (double)
3. `CHSEQR` - Schur factorization (complex single)
4. `ZHSEQR` - Schur factorization (complex double)
5. `STREVC` - Eigenvectors from Schur (single)
6. `DTREVC` - Eigenvectors from Schur (double)
7. `CTREVC` - Eigenvectors from Schur (complex single)
8. `ZTREVC` - Eigenvectors from Schur (complex double)
9. `STGEVC` - Generalized eigenvectors (single)
10. `DTGEVC` - Generalized eigenvectors (double)
11. `CTGEVC` - Generalized eigenvectors (complex single)
12. `ZTGEVC` - Generalized eigenvectors (complex double)

#### Non-Symmetric Eigenvalue - Balancing (8 operations)

1. `SGEBAL` - Balance matrix (single)
2. `DGEBAL` - Balance matrix (double)
3. `CGEBAL` - Balance matrix (complex single)
4. `ZGEBAL` - Balance matrix (complex double)
5. `SGEBAK` - Transform eigenvectors back (single)
6. `DGEBAK` - Transform eigenvectors back (double)
7. `CGEBAK` - Transform eigenvectors back (complex single)
8. `ZGEBAK` - Transform eigenvectors back (complex double)

#### SVD Computational (6 operations)

1. `SBDSQR` - Bidiagonal SVD (single)
2. `DBDSQR` - Bidiagonal SVD (double)
3. `CBDSQR` - Bidiagonal SVD (complex single)
4. `ZBDSQR` - Bidiagonal SVD (complex double)
5. `SBDSDC` - Bidiagonal SVD divide-and-conquer (single)
6. `DBDSDC` - Bidiagonal SVD divide-and-conquer (double)

#### Generalized Eigenvalue - Reduction (6 operations)

1. `SSYGST` - Reduce generalized symmetric (single)
2. `DSYGST` - Reduce generalized symmetric (double)
3. `CSYGST` - Reduce generalized complex symmetric (complex single)
4. `ZSYGST` - Reduce generalized complex symmetric (complex double)
5. `CHEGST` - Reduce generalized Hermitian (complex single)
6. `ZHEGST` - Reduce generalized Hermitian (complex double)

#### Generalized Eigenvalue - Hessenberg (4 operations)

1. `SGGHRD` - Generalized Hessenberg (single)
2. `DGGHRD` - Generalized Hessenberg (double)
3. `CGGHRD` - Generalized Hessenberg (complex single)
4. `ZGGHRD` - Generalized Hessenberg (complex double)

#### Generalized Eigenvalue - QZ Algorithm (4 operations)

1. `SHGEQZ` - QZ algorithm (single)
2. `DHGEQZ` - QZ algorithm (double)
3. `CHGEQZ` - QZ algorithm (complex single)
4. `ZHGEQZ` - QZ algorithm (complex double)

#### Generalized Eigenvalue - Reordering (24 operations)

1. `STGEXC` - Reorder Schur factorization (single)
2. `DTGEXC` - Reorder Schur factorization (double)
3. `CTGEXC` - Reorder Schur factorization (complex single)
4. `ZTGEXC` - Reorder Schur factorization (complex double)
5. `STGSEN` - Reorder with condition (single)
6. `DTGSEN` - Reorder with condition (double)
7. `CTGSEN` - Reorder with condition (complex single)
8. `ZTGSEN` - Reorder with condition (complex double)
9. `STGSNA` - Condition numbers (single)
10. `DTGSNA` - Condition numbers (double)
11. `CTGSNA` - Condition numbers (complex single)
12. `ZTGSNA` - Condition numbers (complex double)
13. `STGSYL` - Sylvester equation (single)
14. `DTGSYL` - Sylvester equation (double)
15. `CTGSYL` - Sylvester equation (complex single)
16. `ZTGSYL` - Sylvester equation (complex double)
17. `STGSJA` - Jacobi-type SVD (single)
18. `DTGSJA` - Jacobi-type SVD (double)
19. `CTGSJA` - Jacobi-type SVD (complex single)
20. `ZTGSJA` - Jacobi-type SVD (complex double)
21. `STGSY2` - Sylvester level-2 (single)
22. `DTGSY2` - Sylvester level-2 (double)
23. `CTGSY2` - Sylvester level-2 (complex single)
24. `ZTGSY2` - Sylvester level-2 (complex double)

#### Bidiagonal Reduction for SVD (20 operations)

1. `SGEBRD` - Reduce to bidiagonal (single)
2. `DGEBRD` - Reduce to bidiagonal (double)
3. `CGEBRD` - Reduce to bidiagonal (complex single)
4. `ZGEBRD` - Reduce to bidiagonal (complex double)
5. `SORGBR` - Generate orthogonal matrices (single)
6. `DORGBR` - Generate orthogonal matrices (double)
7. `CUNGBR` - Generate unitary matrices (complex single)
8. `ZUNGBR` - Generate unitary matrices (complex double)
9. `SORMBR` - Multiply by orthogonal matrices (single)
10. `DORMBR` - Multiply by orthogonal matrices (double)
11. `CUNMBR` - Multiply by unitary matrices (complex single)
12. `ZUNMBR` - Multiply by unitary matrices (complex double)
13. `SLABRD` - Band reduction utilities (single)
14. `DLABRD` - Band reduction utilities (double)
15. `CLABRD` - Band reduction utilities (complex single)
16. `ZLABRD` - Band reduction utilities (complex double)
17. `SORGTR` - Generate orthogonal from tridiagonal (single)
18. `DORGTR` - Generate orthogonal from tridiagonal (double)
19. `CUNGTR` - Generate unitary from tridiagonal (complex single)
20. `ZUNGTR` - Generate unitary from tridiagonal (complex double)

**COMPUTATIONAL ROUTINES TOTAL: 840 operations**

---

### LAPACK AUXILIARY ROUTINES (380 operations)

**Complete Enumerated List:**

#### Matrix Norms and Condition Estimation (52 operations)

1. `SLANGE` - Matrix norm general (single)
2. `DLANGE` - Matrix norm general (double)
3. `CLANGE` - Matrix norm general (complex single)
4. `ZLANGE` - Matrix norm general (complex double)
5. `SLANGB` - Matrix norm banded (single)
6. `DLANGB` - Matrix norm banded (double)
7. `CLANGB` - Matrix norm banded (complex single)
8. `ZLANGB` - Matrix norm banded (complex double)
9. `SLANGT` - Matrix norm tridiagonal (single)
10. `DLANGT` - Matrix norm tridiagonal (double)
11. `CLANGT` - Matrix norm tridiagonal (complex single)
12. `ZLANGT` - Matrix norm tridiagonal (complex double)
13. `SLANHS` - Matrix norm Hessenberg (single)
14. `DLANHS` - Matrix norm Hessenberg (double)
15. `CLANHS` - Matrix norm Hessenberg (complex single)
16. `ZLANHS` - Matrix norm Hessenberg (complex double)
17. `SLANSB` - Matrix norm symmetric banded (single)
18. `DLANSB` - Matrix norm symmetric banded (double)
19. `CLANSB` - Matrix norm symmetric banded (complex single)
20. `ZLANSB` - Matrix norm symmetric banded (complex double)
21. `SLANSP` - Matrix norm symmetric packed (single)
22. `DLANSP` - Matrix norm symmetric packed (double)
23. `CLANSP` - Matrix norm symmetric packed (complex single)
24. `ZLANSP` - Matrix norm symmetric packed (complex double)
25. `SLANSY` - Matrix norm symmetric (single)
26. `DLANSY` - Matrix norm symmetric (double)
27. `CLANSY` - Matrix norm symmetric (complex single)
28. `ZLANSY` - Matrix norm symmetric (complex double)
29. `CLANHP` - Matrix norm Hermitian packed (complex single)
30. `ZLANHP` - Matrix norm Hermitian packed (complex double)
31. `CLANHE` - Matrix norm Hermitian (complex single)
32. `ZLANHE` - Matrix norm Hermitian (complex double)
33. `CLANHB` - Matrix norm Hermitian banded (complex single)
34. `ZLANHB` - Matrix norm Hermitian banded (complex double)
35. `SLANTB` - Matrix norm triangular banded (single)
36. `DLANTB` - Matrix norm triangular banded (double)
37. `CLANTB` - Matrix norm triangular banded (complex single)
38. `ZLANTB` - Matrix norm triangular banded (complex double)
39. `SLANTP` - Matrix norm triangular packed (single)
40. `DLANTP` - Matrix norm triangular packed (double)
41. `CLANTP` - Matrix norm triangular packed (complex single)
42. `ZLANTP` - Matrix norm triangular packed (complex double)
43. `SLANTR` - Matrix norm triangular (single)
44. `DLANTR` - Matrix norm triangular (double)
45. `CLANTR` - Matrix norm triangular (complex single)
46. `ZLANTR` - Matrix norm triangular (complex double)
47. `SLANST` - Matrix norm symmetric tridiagonal (single)
48. `DLANST` - Matrix norm symmetric tridiagonal (double)
49. `SLACON` - Condition number estimation (single)
50. `DLACON` - Condition number estimation (double)
51. `CLACON` - Condition number estimation (complex single)
52. `ZLACON` - Condition number estimation (complex double)

#### Householder Reflections and Rotations (36 operations)

1. `SLARF` - Apply Householder reflection (single)
2. `DLARF` - Apply Householder reflection (double)
3. `CLARF` - Apply Householder reflection (complex single)
4. `ZLARF` - Apply Householder reflection (complex double)
5. `SLARFB` - Apply block reflector (single)
6. `DLARFB` - Apply block reflector (double)
7. `CLARFB` - Apply block reflector (complex single)
8. `ZLARFB` - Apply block reflector (complex double)
9. `SLARFG` - Generate Householder reflection (single)
10. `DLARFG` - Generate Householder reflection (double)
11. `CLARFG` - Generate Householder reflection (complex single)
12. `ZLARFG` - Generate Householder reflection (complex double)
13. `SLARFT` - Form triangular factor of reflector (single)
14. `DLARFT` - Form triangular factor of reflector (double)
15. `CLARFT` - Form triangular factor of reflector (complex single)
16. `ZLARFT` - Form triangular factor of reflector (complex double)
17. `SLARFX` - Apply elementary reflector (single)
18. `DLARFX` - Apply elementary reflector (double)
19. `CLARFX` - Apply elementary reflector (complex single)
20. `ZLARFX` - Apply elementary reflector (complex double)
21. `SLARTG` - Generate plane rotation (single)
22. `DLARTG` - Generate plane rotation (double)
23. `CLARTG` - Generate plane rotation (complex single)
24. `ZLARTG` - Generate plane rotation (complex double)
25. `SLARTV` - Apply vector of plane rotations (single)
26. `DLARTV` - Apply vector of plane rotations (double)
27. `CLARTV` - Apply vector of plane rotations (complex single)
28. `ZLARTV` - Apply vector of plane rotations (complex double)
29. `SLASR` - Apply sequence of plane rotations (single)
30. `DLASR` - Apply sequence of plane rotations (double)
31. `CLACGV` - Conjugate vector (complex single)
32. `ZLACGV` - Conjugate vector (complex double)
33. `CROT` - Apply plane rotation (complex single)
34. `ZROT` - Apply plane rotation (complex double)
35. `CSROT` - Apply real rotation to complex vectors (complex single)
36. `ZDROT` - Apply real rotation to complex vectors (complex double)

#### Matrix Scaling and Equilibration (28 operations)

1. `SLASCL` - Scale matrix (single)
2. `DLASCL` - Scale matrix (double)
3. `CLASCL` - Scale matrix (complex single)
4. `ZLASCL` - Scale matrix (complex double)
5. `SLAQGE` - Equilibrate general matrix (single)
6. `DLAQGE` - Equilibrate general matrix (double)
7. `CLAQGE` - Equilibrate general matrix (complex single)
8. `ZLAQGE` - Equilibrate general matrix (complex double)
9. `SLAQGB` - Equilibrate banded matrix (single)
10. `DLAQGB` - Equilibrate banded matrix (double)
11. `CLAQGB` - Equilibrate banded matrix (complex single)
12. `ZLAQGB` - Equilibrate banded matrix (complex double)
13. `SLAQSB` - Equilibrate symmetric banded (single)
14. `DLAQSB` - Equilibrate symmetric banded (double)
15. `CLAQSB` - Equilibrate symmetric banded (complex single)
16. `ZLAQSB` - Equilibrate symmetric banded (complex double)
17. `SLAQSP` - Equilibrate symmetric packed (single)
18. `DLAQSP` - Equilibrate symmetric packed (double)
19. `CLAQSP` - Equilibrate symmetric packed (complex single)
20. `ZLAQSP` - Equilibrate symmetric packed (complex double)
21. `SLAQSY` - Equilibrate symmetric (single)
22. `DLAQSY` - Equilibrate symmetric (double)
23. `CLAQSY` - Equilibrate symmetric (complex single)
24. `ZLAQSY` - Equilibrate symmetric (complex double)
25. `CLAQHP` - Equilibrate Hermitian packed (complex single)
26. `ZLAQHP` - Equilibrate Hermitian packed (complex double)
27. `CLAQHE` - Equilibrate Hermitian (complex single)
28. `ZLAQHE` - Equilibrate Hermitian (complex double)

#### Error Bound Auxiliaries (4 operations)

1. `SLA_GBAMV` / `SGBMVX` - Banded matrix-vector for error bounds (single)
2. `DLA_GBAMV` / `DGBMVX` - Banded matrix-vector for error bounds (double)
3. `CLA_GBAMV` / `CGBMVX` - Banded matrix-vector for error bounds (complex single)
4. `ZLA_GBAMV` / `ZGBMVX` - Banded matrix-vector for error bounds (complex double)

#### Matrix Copying and Transposition (16 operations)

1. `SLACPY` - Copy matrix (single)
2. `DLACPY` - Copy matrix (double)
3. `CLACPY` - Copy matrix (complex single)
4. `ZLACPY` - Copy matrix (complex double)
5. `SLASET` - Initialize matrix (single)
6. `DLASET` - Initialize matrix (double)
7. `CLASET` - Initialize matrix (complex single)
8. `ZLASET` - Initialize matrix (complex double)
9. `SLASWP` - Perform row interchanges (single)
10. `DLASWP` - Perform row interchanges (double)
11. `CLASWP` - Perform row interchanges (complex single)
12. `ZLASWP` - Perform row interchanges (complex double)
13. `SLAPMT` - Permute columns (single)
14. `DLAPMT` - Permute columns (double)
15. `CLAPMT` - Permute columns (complex single)
16. `ZLAPMT` - Permute columns (complex double)

#### Tridiagonal and Banded Matrix Utilities (32 operations)

1. `SLAGTF` - Factor tridiagonal (single)
2. `DLAGTF` - Factor tridiagonal (double)
3. `CLAGTF` - Factor tridiagonal (complex single)
4. `ZLAGTF` - Factor tridiagonal (complex double)
5. `SLAGTM` - Matrix multiply tridiagonal (single)
6. `DLAGTM` - Matrix multiply tridiagonal (double)
7. `CLAGTM` - Matrix multiply tridiagonal (complex single)
8. `ZLAGTM` - Matrix multiply tridiagonal (complex double)
9. `SLAGTS` - Solve tridiagonal system (single)
10. `DLAGTS` - Solve tridiagonal system (double)
11. `CLAGTS` - Solve tridiagonal system (complex single)
12. `ZLAGTS` - Solve tridiagonal system (complex double)
13. `SLAPY2` - Safe sqrt(x^2 + y^2) (single)
14. `DLAPY2` - Safe sqrt(x^2 + y^2) (double)
15. `SLAPY3` - Safe sqrt(x^2 + y^2 + z^2) (single)
16. `DLAPY3` - Safe sqrt(x^2 + y^2 + z^2) (double)
17. `SLABAD` - Adjust machine parameters (single)
18. `DLABAD` - Adjust machine parameters (double)
19. `SLAMCH` - Machine parameters (single)
20. `DLAMCH` - Machine parameters (double)
21. `SLASSQ` - Update sum of squares (single)
22. `DLASSQ` - Update sum of squares (double)
23. `CLASSQ` - Update sum of squares (complex single)
24. `ZLASSQ` - Update sum of squares (complex double)
25. `SLAS2` - Compute singular values of 2x2 matrix (single)
26. `DLAS2` - Compute singular values of 2x2 matrix (double)
27. `SLASV2` - Compute SVD of 2x2 triangular (single)
28. `DLASV2` - Compute SVD of 2x2 triangular (double)
29. `SLAEBZ` - Bisection for eigenvalues (single)
30. `DLAEBZ` - Bisection for eigenvalues (double)
31. `SLADIV` - Complex division (single)
32. `DLADIV` - Complex division (double)

#### Eigenvalue Utilities (52 operations)

1. `SLAED0` - Divide-and-conquer driver (single)
2. `DLAED0` - Divide-and-conquer driver (double)
3. `SLAED1` - D&C update eigenvectors (single)
4. `DLAED1` - D&C update eigenvectors (double)
5. `SLAED2` - D&C merge eigenvalues (single)
6. `DLAED2` - D&C merge eigenvalues (double)
7. `SLAED3` - D&C solve secular equation (single)
8. `DLAED3` - D&C solve secular equation (double)
9. `SLAED4` - D&C compute single eigenvalue (single)
10. `DLAED4` - D&C compute single eigenvalue (double)
11. `SLAED5` - D&C 2x2 problem (single)
12. `DLAED5` - D&C 2x2 problem (double)
13. `SLAED6` - D&C rational interpolation (single)
14. `DLAED6` - D&C rational interpolation (double)
15. `SLAED7` - D&C deflation (single)
16. `DLAED7` - D&C deflation (double)
17. `SLAED8` - D&C merge (single)
18. `DLAED8` - D&C merge (double)
19. `SLAED9` - D&C eigenvectors (single)
20. `DLAED9` - D&C eigenvectors (double)
21. `SLAEDA` - D&C Z vector (single)
22. `DLAEDA` - D&C Z vector (double)
23. `SLAEV2` - Eigenvalues of 2x2 symmetric (single)
24. `DLAEV2` - Eigenvalues of 2x2 symmetric (double)
25. `SLAEXC` - Exchange diagonal blocks (single)
26. `DLAEXC` - Exchange diagonal blocks (double)
27. `CLAEXC` - Exchange diagonal blocks (complex single)
28. `ZLAEXC` - Exchange diagonal blocks (complex double)
29. `SLAEIN` - Inverse iteration for eigenvectors (single)
30. `DLAEIN` - Inverse iteration for eigenvectors (double)
31. `CLAEIN` - Inverse iteration for eigenvectors (complex single)
32. `ZLAEIN` - Inverse iteration for eigenvectors (complex double)
33. `SLAHQR` - Hessenberg QR algorithm (single)
34. `DLAHQR` - Hessenberg QR algorithm (double)
35. `CLAHQR` - Hessenberg QR algorithm (complex single)
36. `ZLAHQR` - Hessenberg QR algorithm (complex double)
37. `SLAHRD` - Hessenberg reduction (single)
38. `DLAHRD` - Hessenberg reduction (double)
39. `CLAHRD` - Hessenberg reduction (complex single)
40. `ZLAHRD` - Hessenberg reduction (complex double)
41. `SLAIC1` - Update eigenvector estimate (single)
42. `DLAIC1` - Update eigenvector estimate (double)
43. `CLAIC1` - Update eigenvector estimate (complex single)
44. `ZLAIC1` - Update eigenvector estimate (complex double)
45. `SLALN2` - Solve 1x1 or 2x2 linear system (single)
46. `DLALN2` - Solve 1x1 or 2x2 linear system (double)
47. `SLAQR0` - QR algorithm with aggressive deflation (single)
48. `DLAQR0` - QR algorithm with aggressive deflation (double)
49. `SLAQR1` - QR single-shift (single)
50. `DLAQR1` - QR single-shift (double)
51. `SLAQR2` - QR deflation (single)
52. `DLAQR2` - QR deflation (double)

#### SVD and Bidiagonal Utilities (40 operations)

1. `SLASQ1` - Positive definite bidiagonal SVD (single)
2. `DLASQ1` - Positive definite bidiagonal SVD (double)
3. `SLASQ2` - Bidiagonal SVD via dqds (single)
4. `DLASQ2` - Bidiagonal SVD via dqds (double)
5. `SLASQ3` - dqds with checks (single)
6. `DLASQ3` - dqds with checks (double)
7. `SLASQ4` - Convergence test (single)
8. `DLASQ4` - Convergence test (double)
9. `SLASQ5` - One dqds sweep (single)
10. `DLASQ5` - One dqds sweep (double)
11. `SLASQ6` - Compute shift (single)
12. `DLASQ6` - Compute shift (double)
13. `SLASD0` - SVD D&C driver (single)
14. `DLASD0` - SVD D&C driver (double)
15. `SLASD1` - SVD D&C update (single)
16. `DLASD1` - SVD D&C update (double)
17. `SLASD2` - SVD D&C merge (single)
18. `DLASD2` - SVD D&C merge (double)
19. `SLASD3` - SVD D&C solve (single)
20. `DLASD3` - SVD D&C solve (double)
21. `SLASD4` - SVD D&C secular equation (single)
22. `DLASD4` - SVD D&C secular equation (double)
23. `SLASD5` - SVD D&C 2x2 problem (single)
24. `DLASD5` - SVD D&C 2x2 problem (double)
25. `SLASD6` - SVD D&C update singular vectors (single)
26. `DLASD6` - SVD D&C update singular vectors (double)
27. `SLASD7` - SVD D&C merge (single)
28. `DLASD7` - SVD D&C merge (double)
29. `SLASD8` - SVD D&C solve and update (single)
30. `DLASD8` - SVD D&C solve and update (double)
31. `SLASDA` - SVD D&C tree (single)
32. `DLASDA` - SVD D&C tree (double)
33. `SLASDQ` - Bidiagonal SVD via QR (single)
34. `DLASDQ` - Bidiagonal SVD via QR (double)
35. `SLASDT` - Create merge tree (single)
36. `DLASDT` - Create merge tree (double)
37. `SLASR` - Apply Givens rotations (single)
38. `DLASR` - Apply Givens rotations (double)
39. `SLASRT` - Sort numbers (single)
40. `DLASRT` - Sort numbers (double)

#### RRR Algorithm Utilities (24 operations)

1. `SLAR1V` - Compute eigenvector via RRR (single)
2. `DLAR1V` - Compute eigenvector via RRR (double)
3. `SLAR2V` - Apply plane rotations (single)
4. `DLAR2V` - Apply plane rotations (double)
5. `SLARRA` - Compute splitting points (single)
6. `DLARRA` - Compute splitting points (double)
7. `SLARRB` - Sturm count with RRR (single)
8. `DLARRB` - Sturm count with RRR (double)
9. `SLARRC` - Count eigenvalues (single)
10. `DLARRC` - Count eigenvalues (double)
11. `SLARRD` - Compute eigenvalues via RRR (single)
12. `DLARRD` - Compute eigenvalues via RRR (double)
13. `SLARRE` - Preprocessing for RRR (single)
14. `DLARRE` - Preprocessing for RRR (double)
15. `SLARRF` - Refine eigenvalue approximation (single)
16. `DLARRF` - Refine eigenvalue approximation (double)
17. `SLARRJ` - Improve accuracy (single)
18. `DLARRJ` - Improve accuracy (double)
19. `SLARRK` - Compute k-th eigenvalue (single)
20. `DLARRK` - Compute k-th eigenvalue (double)
21. `SLARRR` - Test for splitting (single)
22. `DLARRR` - Test for splitting (double)
23. `SLARRV` - Compute eigenvectors via RRR (single)
24. `DLARRV` - Compute eigenvectors via RRR (double)

#### QR with Column Pivoting Utilities (12 operations)

1. `SLAQP2` - QR pivoting unblocked (single)
2. `DLAQP2` - QR pivoting unblocked (double)
3. `CLAQP2` - QR pivoting unblocked (complex single)
4. `ZLAQP2` - QR pivoting unblocked (complex double)
5. `SLAQPS` - QR pivoting blocked (single)
6. `DLAQPS` - QR pivoting blocked (double)
7. `CLAQPS` - QR pivoting blocked (complex single)
8. `ZLAQPS` - QR pivoting blocked (complex double)
9. `SLARNV` - Random vector generation (single)
10. `DLARNV` - Random vector generation (double)
11. `CLARNV` - Random vector generation (complex single)
12. `ZLARNV` - Random vector generation (complex double)

#### Triangular and RZ Factorization Utilities (20 operations)

1. `SLARZ` - Apply RZ reflector (single)
2. `DLARZ` - Apply RZ reflector (double)
3. `CLARZ` - Apply RZ reflector (complex single)
4. `ZLARZ` - Apply RZ reflector (complex double)
5. `SLARZB` - Apply block RZ reflector (single)
6. `DLARZB` - Apply block RZ reflector (double)
7. `CLARZB` - Apply block RZ reflector (complex single)
8. `ZLARZB` - Apply block RZ reflector (complex double)
9. `SLARZT` - Form block RZ reflector (single)
10. `DLARZT` - Form block RZ reflector (double)
11. `CLARZT` - Form block RZ reflector (complex single)
12. `ZLARZT` - Form block RZ reflector (complex double)
13. `SLATRZ` - RZ factorization (single)
14. `DLATRZ` - RZ factorization (double)
15. `CLATRZ` - RZ factorization (complex single)
16. `ZLATRZ` - RZ factorization (complex double)
17. `SLATRD` - Reduce symmetric to tridiagonal (single)
18. `DLATRD` - Reduce symmetric to tridiagonal (double)
19. `CLATRD` - Reduce Hermitian to tridiagonal (complex single)
20. `ZLATRD` - Reduce Hermitian to tridiagonal (complex double)

#### Triangular System Solvers (16 operations)

1. `SLATRS` - Solve triangular with scaling (single)
2. `DLATRS` - Solve triangular with scaling (double)
3. `CLATRS` - Solve triangular with scaling (complex single)
4. `ZLATRS` - Solve triangular with scaling (complex double)
5. `SLATBS` - Solve banded triangular (single)
6. `DLATBS` - Solve banded triangular (double)
7. `CLATBS` - Solve banded triangular (complex single)
8. `ZLATBS` - Solve banded triangular (complex double)
9. `SLATPS` - Solve packed triangular (single)
10. `DLATPS` - Solve packed triangular (double)
11. `CLATPS` - Solve packed triangular (complex single)
12. `ZLATPS` - Solve packed triangular (complex double)
13. `SLATDF` - Solve 2x2 triangular (single)
14. `DLATDF` - Solve 2x2 triangular (double)
15. `CLATDF` - Solve 2x2 triangular (complex single)
16. `ZLATDF` - Solve 2x2 triangular (complex double)

#### Matrix Multiplication and UU^T Operations (12 operations)

1. `SLAUU2` - Compute U*U^T or L^T*L unblocked (single)
2. `DLAUU2` - Compute U*U^T or L^T*L unblocked (double)
3. `CLAUU2` - Compute U*U^H or L^H*L unblocked (complex single)
4. `ZLAUU2` - Compute U*U^H or L^H*L unblocked (complex double)
5. `SLAUUM` - Compute U*U^T or L^T*L blocked (single)
6. `DLAUUM` - Compute U*U^T or L^T*L blocked (double)
7. `CLAUUM` - Compute U*U^H or L^H*L blocked (complex single)
8. `ZLAUUM` - Compute U*U^H or L^H*L blocked (complex double)
9. `SLARGE` - Apply large reflector (single)
10. `DLARGE` - Apply large reflector (double)
11. `CLARGV` - Generate plane rotations (complex single)
12. `ZLARGV` - Generate plane rotations (complex double)

#### Complex-Only Utilities (12 operations)

1. `CLACRM` - Multiply real by complex (complex single)
2. `ZLACRM` - Multiply real by complex (complex double)
3. `CLACRT` - Apply plane rotation (complex single)
4. `ZLACRT` - Apply plane rotation (complex double)
5. `CLAESY` - Solve 2x2 system (complex single)
6. `ZLAESY` - Solve 2x2 system (complex double)
7. `CSPMV` - Packed symmetric matrix-vector (complex single)
8. `ZSPMV` - Packed symmetric matrix-vector (complex double)
9. `CSPR` - Packed symmetric rank-1 update (complex single)
10. `ZSPR` - Packed symmetric rank-1 update (complex double)
11. `CSYMV` - Symmetric matrix-vector (complex single)
12. `ZSYMV` - Symmetric matrix-vector (complex double)

#### Generalized Eigenvalue Utilities (16 operations)

1. `STGEX2` - Swap diagonal blocks 2x2 (single)
2. `DTGEX2` - Swap diagonal blocks 2x2 (double)
3. `CTGEX2` - Swap diagonal blocks 2x2 (complex single)
4. `ZTGEX2` - Swap diagonal blocks 2x2 (complex double)
5. `STGSY2` - Solve generalized Sylvester (single)
6. `DTGSY2` - Solve generalized Sylvester (double)
7. `CTGSY2` - Solve generalized Sylvester (complex single)
8. `ZTGSY2` - Solve generalized Sylvester (complex double)
9. `SLAGS2` - 2x2 SVD (single)
10. `DLAGS2` - 2x2 SVD (double)
11. `CLAGS2` - 2x2 SVD (complex single)
12. `ZLAGS2` - 2x2 SVD (complex double)
13. `SLAGV2` - 2x2 eigenvalue problem (single)
14. `DLAGV2` - 2x2 eigenvalue problem (double)
15. `CLAGV2` - 2x2 eigenvalue problem (complex single)
16. `ZLAGV2` - 2x2 eigenvalue problem (complex double)

#### Miscellaneous Utilities (12 operations)

1. `SRSCL` - Reciprocal scale (single)
2. `DRSCL` - Reciprocal scale (double)
3. `CSRSCL` - Reciprocal scale (complex single)
4. `ZDRSCL` - Reciprocal scale (complex double)
5. `ILAENV` - Environment/tuning parameters (integer)
6. `ILADLC` - Last non-zero column (integer)
7. `ILADLR` - Last non-zero row (integer)
8. `ICMAX1` - Index of max absolute value (integer)
9. `SCSUM1` - Sum of absolute values (single complex)
10. `DZSUM1` - Sum of absolute values (double complex)
11. `LSAME` - Case-insensitive character comparison (logical)
12. `XERBLA` - Error handler (void)

**AUXILIARY ROUTINES TOTAL: 384 operations**

---

### LAPACK SUMMARY

---

## COMBINED SUPERSET (AUTHORITATIVE CANONICAL COUNTS)

**Canonical concrete operation namespace (authoritative):**

| Item                                                   | Count    | Notes                                                                        |
| ------------------------------------------------------ | -------- | ---------------------------------------------------------------------------- |
| Concrete operation names in `src/judge/judge_op_ids.h` | **3054** | Canonical IDs `0..3053`                                                      |
| Concrete operation names in Appendix A                 | **3054** | Exact sorted expansion of the judge namespace                                |
| Previously extracted appendix surface                  | 2254     | Historical undercount from incomplete wildcard/family expansion              |
| Additional names restored in the 2026-04 sync          | 800      | `796` judge-only names + `4` explicit names already present in the spec body |

**What the previous appendix extraction missed:**

| Missing family in prior appendix extraction                                               | Added names |
| ----------------------------------------------------------------------------------------- | ----------- |
| `fb_*` operations                                                                         | 634         |
| ScaLAPACK `p*` operations                                                                 | 154         |
| Batched/strided GEMM concrete names                                                       | 8           |
| Explicit spec names skipped by the old extractor (`isamax`, `idamax`, `icamax`, `izamax`) | 4           |

**Interpretation rules:**

- `src/judge/judge_op_ids.h` and Appendix A define the concrete faster-blaster operation surface.
- The family sections below remain organized by mathematical domain for readability.
- Unified-implementation discussions, alias counts, and planning-oriented family subtotals are design notes, not a second authoritative concrete namespace.

### Design vtable for: **3054 canonical judge-tracked operations**

*Note: Backend implementations may expose fewer operations. Family-level planning tables later in this document may count aliases, optional modules, or conceptual design groupings separately, but Appendix A remains the authoritative concrete list.*

---

# EXTENDED LINEAR ALGEBRA CAPABILITIES

The following sections document **extensions and specialized capabilities** beyond standard BLAS/LAPACK. These are **optional** but provide significant performance benefits for modern workloads.

---

## SPARSE BLAS OPERATIONS (Inspector-Executor Interface)

**Purpose**: Operations on sparse matrices (stored in compressed formats: CSR, CSC, COO, BSR)  
**Backend Support**: Intel MKL, AMD AOCL-Sparse, NVIDIA cuSPARSE, AMD rocSPARSE  
**Use Cases**: ML (sparse neural networks), scientific computing, graph algorithms, iterative solvers

**Modern Approach**: All major backends now use **Inspector-Executor** pattern (two-phase API):
1. **Inspector Phase**: Analyze matrix structure once, optimize kernel selection
2. **Executor Phase**: Fast repeated operations with no analysis overhead

**Performance**: 2-10× faster than legacy single-call API for iterative workloads (solvers, ML training)

**Note**: Legacy Sparse BLAS (format-specific calls like `?csrmv`, `?cscmv`) is deprecated industry-wide. faster-blaster implements only the modern Inspector-Executor interface.

### Sparse Matrix Storage Formats

- **CSR** (Compressed Sparse Row): Row-wise compressed, best for SpMV
- **CSC** (Compressed Sparse Column): Column-wise compressed, best for SpMV with transpose
- **COO** (Coordinate): Triplet format (row, col, value), easy construction, slow computation
- **BSR** (Block Sparse Row): Block-compressed for dense sub-blocks (FEM, graph partitions)
- **ELL** (ELLPACK): Fixed-width format, GPU-friendly
- **Hybrid formats**: CSR5, SELL-C-σ (backend-specific optimizations)

### Inspector Phase (Matrix Lifecycle Management)

**Matrix Creation** (~5 operations × 5 formats × 4 precisions = **20 operations**):
- `fb_sparse_create_csr`: Create from CSR arrays (row_ptr, col_idx, values)
- `fb_sparse_create_csc`: Create from CSC arrays
- `fb_sparse_create_coo`: Create from COO triplets
- `fb_sparse_create_bsr`: Create from BSR arrays with block size
- `fb_sparse_copy`: Deep copy sparse matrix

**Matrix Optimization** (**4 operations**):
- `fb_sparse_optimize`: Analyze structure, select optimal kernel, pre-allocate buffers
- `fb_sparse_set_mv_hint`: Hint for repeated SpMV (expected iteration count)
- `fb_sparse_set_mm_hint`: Hint for repeated SpMM
- `fb_sparse_set_sv_hint`: Hint for repeated sparse triangular solve

**Matrix Conversion** (~10 operations):
- `fb_sparse_convert_csr`: Convert any format → CSR
- `fb_sparse_convert_coo`: Convert any format → COO
- `fb_sparse_convert_bsr`: Convert any format → BSR with specified block size
- `fb_sparse_export`: Export to user-owned arrays (zero-copy where possible)

**Matrix Properties** (**6 operations**):
- `fb_sparse_get_size`: Query rows, cols, nnz (non-zero count)
- `fb_sparse_get_format`: Query current storage format
- `fb_sparse_set_matrix_type`: Set type (general, symmetric, triangular, Hermitian)
- `fb_sparse_set_fill_mode`: Upper/lower triangular for symmetric matrices
- `fb_sparse_set_diag_type`: Unit/non-unit diagonal

**Matrix Destruction** (**1 operation**):
- `fb_sparse_destroy`: Free all resources

**Inspector Phase Total**: ~**40 operations**

### Executor Phase (Computational Operations)

**Sparse Matrix-Vector Operations** (**4 operations** × 4 precisions = **16 operations**):
- `fb_sparse_mv`: Sparse matrix-vector multiply (y := α·op(A)·x + β·y)
  - Supports: transpose, conjugate transpose
  - Works with all matrix types (general, symmetric, triangular, Hermitian)
- `fb_sparse_trsv`: Sparse triangular solve (op(A)·x = α·y)
- `fb_sparse_symv`: Symmetric sparse matrix-vector (optimized, uses half storage)
- `fb_sparse_hermv`: Hermitian sparse matrix-vector (complex only, 2 precisions)

**Sparse Matrix-Dense Matrix Operations** (**6 operations** × 4 precisions = **24 operations**):
- `fb_sparse_mm`: Sparse × dense matrix multiply (C := α·op(A)·B + β·C)
- `fb_sparse_trsm`: Sparse triangular solve with multiple RHS
- `fb_sparse_symm`: Symmetric sparse × dense matrix
- `fb_sparse_hermm`: Hermitian sparse × dense matrix (complex only)
- `fb_sparse_syrk`: Sparse symmetric rank-k update (C := α·A·A^T + β·C)
- `fb_sparse_syr2k`: Sparse symmetric rank-2k update

**Sparse-Sparse Operations** (**3 operations** × 4 precisions = **12 operations**):
- `fb_sparse_spmm`: Sparse × sparse matrix multiply (C := α·op(A)·op(B))
  - Result is sparse, requires analysis phase for optimal output format
- `fb_sparse_add`: Sparse matrix addition (C := α·A + β·B)
- `fb_sparse_spmv`: Sparse matrix × sparse vector (for specialized use cases)

**Executor Phase Total**: ~**52 operations**

### Sparse Iterative Solvers (Built on Inspector-Executor)

**Krylov Subspace Methods** (~8 solver variants × 4 precisions = **32 operations**):
- `fb_sparse_cg`: Conjugate Gradient (for SPD systems)
- `fb_sparse_bicgstab`: BiConjugate Gradient Stabilized (general nonsymmetric)
- `fb_sparse_gmres`: Generalized Minimal Residual (restart parameter)
- `fb_sparse_fgmres`: Flexible GMRES (for varying preconditioners)
- `fb_sparse_minres`: Minimal Residual (symmetric indefinite)
- `fb_sparse_qmr`: Quasi-Minimal Residual
- `fb_sparse_tfqmr`: Transpose-Free QMR
- `fb_sparse_bicg`: BiConjugate Gradient

**Preconditioner Support**:
- User-provided via callback (flexible for custom preconditioners)
- Built-in: ILU(0), Jacobi, block-Jacobi via separate calls

**Solver Interface** (~8 operations):
- `fb_sparse_solver_init`: Initialize solver context with parameters
- `fb_sparse_solver_set_tol`: Set convergence tolerance
- `fb_sparse_solver_set_maxiter`: Set maximum iterations
- `fb_sparse_solver_set_precond`: Attach preconditioner callback
- `fb_sparse_solver_solve`: Execute iterative solve
- `fb_sparse_solver_get_residual`: Query final residual norm
- `fb_sparse_solver_get_iters`: Query iteration count
- `fb_sparse_solver_destroy`: Free solver resources

**Iterative Solvers Total**: ~**40 operations**

### Design Pattern for faster-blaster

**Typical User Code**:
```c
// INSPECTOR: One-time setup
fb_sparse_matrix_t A = fb_sparse_create_csr(m, n, row_ptr, col_idx, values);
fb_sparse_set_mv_hint(A, FB_NO_TRANSPOSE, 1000);  // Will do 1000 SpMVs
fb_sparse_optimize(A);  // Analyze & optimize

// EXECUTOR: Fast repeated operations
for (int iter = 0; iter < 1000; iter++) {
    fb_sparse_mv(FB_NO_TRANSPOSE, 1.0, A, x, 0.0, y);
    // ... other computation ...
}

fb_sparse_destroy(A);
```

**Backend Dispatch** (faster-blaster internals):
```c
// fb_sparse_create_csr() routes to:
//   Intel MKL:       mkl_sparse_s_create_csr()
//   NVIDIA cuSPARSE: cusparseCreateCsr()
//   AMD rocSPARSE:   rocsparse_create_csr_descr()
//   AMD AOCL:        aoclsparse_create_csr()

// fb_sparse_mv() routes to:
//   Intel MKL:       mkl_sparse_s_mv()
//   NVIDIA cuSPARSE: cusparseSpMV()
//   AMD rocSPARSE:   rocsparse_scsrmv()
//   AMD AOCL:        aoclsparse_smv()
```

**Sparse Preconditioner Operations** (~80 operations):
- **Construction/Setup** (32 ops):
  - Jacobi preconditioner: `fb_sparse_precond_jacobi_create/setup/destroy` (8 ops - s/d/c/z)
  - ILU(0) preconditioner: `fb_sparse_precond_ilu0_create/setup/destroy` (8 ops)
  - ILU(p) preconditioner: `fb_sparse_precond_ilup_create/setup/destroy` (8 ops)
  - Incomplete Cholesky IC(0): `fb_sparse_precond_ic0_create/setup/destroy` (8 ops)
- **Application** (28 ops):
  - Apply preconditioner: `fb_sparse_precond_apply` (4 ops - s/d/c/z)
  - Triangular solves: `fb_sparse_precond_trsv_lower/upper` (8 ops)
  - Multi-colored variants: `fb_sparse_precond_gs/sgs/sor/ssor` (16 ops)
- **Approximate Inverses** (20 ops):
  - SPAI (Sparse Approximate Inverse): create/setup/apply/destroy (8 ops)
  - FSAI (Factorized Sparse Approximate Inverse): create/setup/apply/destroy (8 ops)
  - TNS (Truncated Neumann Series): create/setup/apply/destroy (4 ops)

**Structured Sparse Operations** (2:4 sparsity, ~24 operations):
- **Matrix Creation**: `fb_sparse_structured_create_2_4` (4 ops - s/d/c/z)
- **Pruning**: Convert dense → 2:4 structured: `fb_sparse_prune_to_2_4` (4 ops)
- **SpMM (2:4 structured)**: Optimized sparse-dense matmul (8 ops - strided/batched variants)
- **Metadata queries**: `fb_sparse_structured_get_compression_ratio` (4 ops)
- **Backends**: NVIDIA Ampere+ (cuSPARSELt), AMD RDNA3+ (hipSPARSELt)
- **Use case**: ML inference with structured pruning (50% sparsity, hardware-accelerated)
- **Format**: 2-out-of-4 pattern enforced at hardware level for tensor cores
- **Performance**: 2× theoretical speedup vs. dense when properly structured

**SPARSE BLAS COMPLETE TOTAL**: ~**236 operations**
- Inspector Phase: 40 operations (lifecycle, optimization, conversion)
- Executor Phase: 52 operations (SpMV, SpMM, sparse-sparse)
- Iterative Solvers: 40 operations (Krylov methods)
- Preconditioners: 80 operations (rocALUTION-style)
- Structured Sparse: 24 operations (2:4 sparsity for ML)

**Key Advantages Over Legacy Interface**:
1. **Format-agnostic**: Single API works with CSR, COO, BSR, etc. (backend chooses best)
2. **Performance**: Optimization phase eliminates per-call overhead
3. **Modern backends**: All vendors support this pattern (legacy APIs being removed)
4. **GPU-friendly**: Enables asynchronous execution and buffer pre-allocation
5. **ML-optimized**: 2:4 structured sparse for tensor core acceleration
6. **Complete solver stack**: Preconditioners enable production-quality iterative solvers

---

## 6. UNIFIED LINEAR ALGEBRA OPERATIONS

**Purpose**: Single implementation with zero-cost aliases for all GEMM variants (precision, batching, fusion). Eliminates code duplication while preserving familiar APIs.

### 6.1 Unified GEMM Implementation

**Core unified interface** covering 82 GEMM variants across the specification:

```c
// Single implementation replacing all GEMM variants
fb_status_t fb_gemm_unified(
    // Precision selection
    fb_precision_t precision,        // FP64, FP32, C64, C32, BF16, FP16, FP8_E4M3, FP8_E5M2, INT8, INT4
    
    // Batching mode
    fb_batch_mode_t batch_mode,      // SINGLE, ARRAY (array of pointers), STRIDED (fixed stride)
    size_t batch_count,              // Number of matrices (1 for SINGLE)
    
    // Fusion operations
    fb_fusion_t fusion,              // NONE, RELU, GELU, SILU, SWISH, MISH, TANH, SIGMOID
    fb_fusion_t fusion2,             // Secondary fusion: BIAS, RESIDUAL, LAYERNORM
    
    // Matrix dimensions and layout
    fb_layout_t layout,              // ROW_MAJOR, COL_MAJOR
    fb_transpose_t trans_a,          // NO_TRANS, TRANS, CONJ_TRANS
    fb_transpose_t trans_b,
    size_t m, size_t n, size_t k,
    
    // Scaling factors (type determined by precision)
    const void* alpha,
    const void* beta,
    
    // Matrix data (arrays for batched modes)
    const void* a,  size_t lda,  size_t stride_a,
    const void* b,  size_t ldb,  size_t stride_b,
    void* c,        size_t ldc,  size_t stride_c,
    
    // Optional fusion parameters
    const fb_fusion_params_t* fusion_params   // Bias vectors, residual, normalization
);
```

**Enumerations**:

```c
typedef enum {
    FB_PREC_FP64,      // Double precision (BLAS: DGEMM)
    FB_PREC_FP32,      // Single precision (BLAS: SGEMM)
    FB_PREC_C64,       // Complex double (BLAS: ZGEMM)
    FB_PREC_C32,       // Complex single (BLAS: CGEMM)
    FB_PREC_BF16,      // BFloat16 (ML training)
    FB_PREC_FP16,      // IEEE half precision
    FB_PREC_FP8_E4M3,  // FP8 E4M3 (AI inference)
    FB_PREC_FP8_E5M2,  // FP8 E5M2 (AI inference)
    FB_PREC_INT8,      // 8-bit integer quantized
    FB_PREC_INT4       // 4-bit integer (weight-only quantization)
} fb_precision_t;

typedef enum {
    FB_BATCH_SINGLE,   // Single matrix operation
    FB_BATCH_ARRAY,    // Array of matrix pointers (non-contiguous)
    FB_BATCH_STRIDED   // Fixed-stride batching (contiguous memory)
} fb_batch_mode_t;

typedef enum {
    FB_FUSION_NONE,
    FB_FUSION_RELU,      // max(0, x)
    FB_FUSION_GELU,      // x * Φ(x) - Gaussian Error Linear Unit
    FB_FUSION_SILU,      // x * σ(x) - Swish/SiLU activation
    FB_FUSION_SWISH,     // Same as SILU
    FB_FUSION_MISH,      // x * tanh(softplus(x))
    FB_FUSION_TANH,      // tanh(x)
    FB_FUSION_SIGMOID,   // σ(x) = 1/(1 + e^(-x))
    FB_FUSION_BIAS,      // Add broadcasted bias vector
    FB_FUSION_RESIDUAL,  // Add residual connection
    FB_FUSION_LAYERNORM  // Apply layer normalization
} fb_fusion_t;
```

**Backend Dispatch**:
- **AMD EPYC CPU**: AOCL-DLP (low-precision), AOCL-BLIS (standard)
- **Intel CPU**: MKL (with oneMKL matmul_post_ops for fusion)
- **NVIDIA GPU**: cuBLAS (standard), cuBLASTLt (fused ops)
- **AMD GPU**: rocBLAS (standard), hipBLASLt (fused ops)
- **Apple Silicon**: Accelerate framework

**Performance Characteristics**:
- Zero-cost abstraction: Compiler inlines alias wrappers (verified with `-O2` on GCC/Clang/MSVC)
- Single kernel launch for fused operations eliminates 2-5× memory bandwidth overhead
- Mixed-precision achieves 3-10× speedup on Tensor Cores (NVIDIA Ampere+, AMD CDNA2+)
- Batched operations achieve near-linear scaling on GPUs (vs sequential calls)

**UNIFIED GEMM TOTAL**: **1 implementation** (replaces 82 separate implementations)

### 6.2 Unified Normalization Implementation

**Core unified interface** covering ~30 normalization variants across the specification:

```c
// Single implementation replacing all normalization variants
fb_status_t fb_normalize_unified(
    // Normalization mode selection
    fb_norm_mode_t mode,             // Z_SCORE, BATCH_NORM, LAYER_NORM, INSTANCE_NORM, GROUP_NORM,
                                     // L1_NORM, L2_NORM, MAX_NORM, MIN_MAX_SCALE
    
    // Execution phase (for trainable normalizations)
    fb_norm_phase_t phase,           // TRAINING, INFERENCE
    
    // Data dimensions
    const size_t* dims,              // Input tensor dimensions [N, C, H, W] or [batch, seq, features]
    size_t ndims,                    // Number of dimensions
    
    // Normalization axis control
    const int* normalize_axes,       // Which axes to normalize over (NULL = auto-select by mode)
    size_t num_axes,
    
    // Input/output data
    const void* input,
    void* output,
    
    // Trainable parameters (NULL if not applicable)
    const void* scale,               // γ (gamma) - multiplicative parameter
    const void* bias,                // β (beta) - additive parameter
    
    // Running statistics (for batch norm in inference)
    const void* running_mean,
    const void* running_var,
    float momentum,                  // For updating running stats in training
    
    // Normalization parameters
    float epsilon,                   // Numerical stability constant (default: 1e-5)
    int num_groups,                  // For group normalization
    
    // Min-max scaling parameters
    float scale_min,                 // Target minimum value
    float scale_max,                 // Target maximum value
    
    // Backward pass (if phase == TRAINING)
    const void* grad_output,         // Gradient from next layer
    void* grad_input,                // Gradient to propagate
    void* grad_scale,                // Gradient w.r.t. scale
    void* grad_bias                  // Gradient w.r.t. bias
);
```

**Enumerations**:

```c
typedef enum {
    FB_NORM_Z_SCORE,        // Statistical z-score: (x - μ) / σ
    FB_NORM_BATCH,          // Batch normalization (DNN - normalize over batch dimension)
    FB_NORM_LAYER,          // Layer normalization (Transformers - normalize over features)
    FB_NORM_INSTANCE,       // Instance normalization (Style transfer - per-sample per-channel)
    FB_NORM_GROUP,          // Group normalization (num_groups divides channels)
    FB_NORM_L1,             // L1 normalization: x / ||x||₁
    FB_NORM_L2,             // L2 normalization: x / ||x||₂
    FB_NORM_MAX,            // Max normalization: x / max(|x|)
    FB_NORM_MIN_MAX_SCALE   // Min-max scaling: (x - min) / (max - min) * (scale_max - scale_min) + scale_min
} fb_norm_mode_t;

typedef enum {
    FB_NORM_TRAINING,       // Training mode: compute and update running statistics
    FB_NORM_INFERENCE       // Inference mode: use fixed running statistics
} fb_norm_phase_t;
```

**Axis Selection by Mode** (when normalize_axes = NULL):
- **BATCH_NORM**: Normalizes over batch + spatial dims (keep channels independent)
- **LAYER_NORM**: Normalizes over feature dimensions (common in Transformers)
- **INSTANCE_NORM**: Normalizes per sample, per channel (spatial dims only)
- **GROUP_NORM**: Splits channels into groups, normalizes within groups
- **Z_SCORE, L1, L2, MAX**: Normalizes across all specified axes
- **MIN_MAX_SCALE**: Scales based on min/max of input data

**Backend Dispatch**:
- **DNN backends**: oneDNN, cuDNN, MIOpen, ZenDNN (batch/layer/instance/group norm)
- **ML libraries**: oneMKL (VSL statistics), AOCL-DA (z-score, standardization)
- **GPU compute**: cuBLAS (vector norms), rocBLAS (vector norms)
- **Fallback**: faster-blaster-reference (pure C implementation)

**Performance Characteristics**:
- Zero-cost abstraction: Mode-specific implementations selected at compile time when possible
- Single kernel for DNN normalizations (no separate mean/variance/normalize passes)
- Fused with preceding/following operations when backend supports it
- Automatic axis selection reduces parameter count for common cases

**UNIFIED NORMALIZATION TOTAL**: **1 implementation** (replaces ~30 separate implementations)

### 6.3 Unified Reduction Implementation

**Core unified interface** covering ~24 reduction variants across the specification:

```c
// Single implementation replacing all reduction variants
fb_status_t fb_reduce_unified(
    // Reduction operation selection
    fb_reduce_op_t operation,        // SUM, MIN, MAX, PRODUCT, MEAN, VARIANCE, STD_DEV, NORM_L1, NORM_L2
    
    // Reduction scope
    fb_reduce_scope_t scope,         // LOCAL (single device), COLLECTIVE (multi-device)
    
    // Input/output configuration
    const void* input,
    void* output,
    const size_t* input_dims,        // Input tensor dimensions
    size_t ndims,
    
    // Reduction axes (NULL = reduce all)
    const int* reduce_axes,          // Which axes to reduce over
    size_t num_axes,
    bool keepdims,                   // Keep reduced dimensions as size-1
    
    // Scan (prefix operation) configuration
    fb_scan_mode_t scan_mode,        // NONE (pure reduction), INCLUSIVE, EXCLUSIVE
    const void* scan_init,           // Initial value for exclusive scan
    
    // Collective communication (for COLLECTIVE scope)
    fb_comm_t communicator,          // Multi-device communicator (NCCL/RCCL/MPI)
    int root_rank,                   // Root device for Reduce (not AllReduce)
    
    // Output configuration
    void* scratch_buffer,            // Workspace for temporary data
    size_t scratch_size,
    
    // Stream/queue for async execution
    void* stream                     // CUDA stream / HIP stream / SYCL queue
);\n```

**Enumerations**:

```c
typedef enum {
    FB_REDUCE_SUM,          // Sum: Σx_i
    FB_REDUCE_MIN,          // Minimum: min(x_i)
    FB_REDUCE_MAX,          // Maximum: max(x_i)
    FB_REDUCE_PRODUCT,      // Product: \u220fx_i
    FB_REDUCE_MEAN,         // Mean: (Σx_i) / n
    FB_REDUCE_VARIANCE,     // Variance: Σ(x_i - μ)² / n
    FB_REDUCE_STD_DEV,      // Standard deviation: √variance
    FB_REDUCE_NORM_L1,      // L1 norm: Σ|x_i|
    FB_REDUCE_NORM_L2,      // L2 norm: √(Σx_i²)
    FB_REDUCE_NORM_MAX      // Max norm: max(|x_i|)
} fb_reduce_op_t;

typedef enum {
    FB_REDUCE_LOCAL,        // Single device reduction (tensor/parallel primitives)
    FB_REDUCE_COLLECTIVE    // Multi-device reduction (NCCL/RCCL/MPI)
} fb_reduce_scope_t;

typedef enum {
    FB_SCAN_NONE,           // Pure reduction (collapse to smaller tensor)
    FB_SCAN_INCLUSIVE,      // Prefix scan: output[i] = op(input[0..i])
    FB_SCAN_EXCLUSIVE       // Prefix scan: output[i] = op(input[0..i-1])
} fb_scan_mode_t;
```

**Reduction Patterns**:

1. **Tensor Reduction** (Section 10):
   ```c
   // fb_tensor_reduce_sum(input, axes, output)
   fb_reduce_unified(FB_REDUCE_SUM, FB_REDUCE_LOCAL, input, output, 
                    dims, ndims, axes, num_axes, keepdims, FB_SCAN_NONE, ...)
   ```

2. **Parallel Primitive Reduction** (Section 11):
   ```c
   // fb_prim_reduce(input, output, size, op)
   fb_reduce_unified(op, FB_REDUCE_LOCAL, input, output, 
                    &size, 1, NULL, 0, false, FB_SCAN_NONE, ...)
   ```

3. **Parallel Scan/Prefix Sum** (Section 11):
   ```c
   // fb_prim_scan_inclusive(input, output, size, op)
   fb_reduce_unified(op, FB_REDUCE_LOCAL, input, output,
                    &size, 1, NULL, 0, false, FB_SCAN_INCLUSIVE, ...)
   ```

4. **Collective AllReduce** (Section 12):
   ```c
   // fb_nccl_allreduce(sendbuf, recvbuf, count, datatype, op, comm)
   fb_reduce_unified(op, FB_REDUCE_COLLECTIVE, sendbuf, recvbuf,
                    &count, 1, NULL, 0, false, FB_SCAN_NONE, comm, -1, ...)
   ```

5. **Statistics** (Section 14):
   ```c
   // fb_stats_sum(data, n, result), fb_stats_mean(data, n, result)
   fb_reduce_unified(FB_REDUCE_SUM / FB_REDUCE_MEAN, FB_REDUCE_LOCAL,
                    data, result, &n, 1, NULL, 0, false, FB_SCAN_NONE, ...)
   ```

**Backend Dispatch**:
- **Tensor reductions**: cuTENSOR, hipTensor, oneMKL (DPC++)
- **GPU primitives**: CUB, rocPRIM, Thrust/rocThrust
- **Collective**: NCCL (NVIDIA), RCCL (AMD), oneCCL (Intel), MPI (CPU fallback)
- **Statistics**: oneMKL VSL, AOCL-DA, GSL (fallback)
- **CPU fallback**: OpenMP parallel reductions, blas-lapack-reference

**Performance Characteristics**:
- Zero-copy for contiguous reductions (no intermediate buffers)
- Kernel fusion: Multi-axis reductions in single kernel when backend supports
- Hierarchical reduction: Block → Warp → Device for GPU implementations
- Communication overlap: Computation overlaps with AllReduce in multi-GPU
- Scan optimization: Hillis-Steele (work-efficient) vs Blelloch (step-efficient) selected by backend

**UNIFIED REDUCTION TOTAL**: **1 implementation** (replaces ~24 separate implementations)

---

## BLAS-LIKE EXTENSIONS (Aliases to Unified Implementations) (Aliases to Unified Implementations)

⚠️ **Note**: Most GEMM operations in this section are convenience aliases to unified implementations (Section 6). For maximum performance and flexibility, use unified APIs directly.

**Purpose**: Modern operations for batched computation, mixed-precision, in-place transformations  
**Backend Support**: Intel MKL, AMD AOCL (partial), NVIDIA cuBLAS (batched variants), rocBLAS (batched variants)  
**Use Cases**: ML inference (many small matrices), batched solvers, tensor operations

### Batched Operations (Primary Extension)
*Process multiple independent matrices/vectors in single call - critical for GPU efficiency*

**Batched Level 1**: ~6 operations × 4 precisions = **24 operations**
- `cblas_?axpy_batch` / `cblas_?axpy_batch_strided`: Batched AXPY
- `cblas_?copy_batch` / `cblas_?copy_batch_strided`: Batched vector copy

**Batched Level 2**: ~8 operations × 4 precisions = **32 operations**
- `cblas_?gemv_batch` / `cblas_?gemv_batch_strided`: Batched matrix-vector multiply
- `cblas_?dgmm_batch` / `cblas_?dgmm_batch_strided`: Batched diagonal matrix multiply

**Batched Level 3 GEMM** (16 aliases - array mode):
```c
void cblas_sgemm_batch(...)  // Alias → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_ARRAY, batch_count, ...)
void cblas_dgemm_batch(...)  // Alias → fb_gemm_unified(FB_PREC_FP64, FB_BATCH_ARRAY, batch_count, ...)
void cblas_cgemm_batch(...)  // Alias → fb_gemm_unified(FB_PREC_C32, FB_BATCH_ARRAY, batch_count, ...)
void cblas_zgemm_batch(...)  // Alias → fb_gemm_unified(FB_PREC_C64, FB_BATCH_ARRAY, batch_count, ...)
```

**Batched Level 3 GEMM** (16 aliases - strided mode):
```c
void cblas_sgemm_batch_strided(...)  // Alias → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_STRIDED, batch_count, ...)
void cblas_dgemm_batch_strided(...)  // Alias → fb_gemm_unified(FB_PREC_FP64, FB_BATCH_STRIDED, batch_count, ...)
void cblas_cgemm_batch_strided(...)  // Alias → fb_gemm_unified(FB_PREC_C32, FB_BATCH_STRIDED, batch_count, ...)
void cblas_zgemm_batch_strided(...)  // Alias → fb_gemm_unified(FB_PREC_C64, FB_BATCH_STRIDED, batch_count, ...)
```

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

**Other Batched Level 3**: ~16 operations (SYMM, SYRK, SYR2K, TRSM batched variants - not unified)
- `cblas_?gemm3m_batch` / `cblas_?gemm3m_batch_strided`: Optimized complex multiply (3 real GEMMs)
- `cblas_?symm_batch`, `cblas_?syrk_batch`, `cblas_?syr2k_batch`: Batched symmetric operations
- `cblas_?trsm_batch` / `cblas_?trsm_batch_strided`: Batched triangular solve

### Mixed-Precision Operations (Aliases)
*Compute in lower precision, accumulate in higher - faster with maintained accuracy*

**Mixed Precision GEMM** (12 aliases):
```c
void cblas_gemm_bf16bf16f32(...)  // Alias → fb_gemm_unified(FB_PREC_BF16, ..., output_fp32)
void cblas_gemm_f16f16f32(...)    // Alias → fb_gemm_unified(FB_PREC_FP16, ..., output_fp32)
void cblas_gemm_e5m2e5m2f32(...)  // Alias → fb_gemm_unified(FB_PREC_FP8_E5M2, ..., output_fp32)
void cblas_gemm_e4m3e4m3f32(...)  // Alias → fb_gemm_unified(FB_PREC_FP8_E4M3, ..., output_fp32)
void cblas_gemm_s8s8s32(...)      // Alias → fb_gemm_unified(FB_PREC_INT8, ..., output_int32)
void cblas_gemm_s8u8s32(...)      // Alias → fb_gemm_unified(FB_PREC_INT8, ..., unsigned_b, output_int32)
```

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

**Packed Matrix Operations**: **16 operations**
- `cblas_?gemm_pack_get_size`: Query buffer size for packed matrix
- `cblas_?gemm_pack`: Pack matrix into efficient layout (cache-optimized)
- `cblas_?gemm_compute`: GEMM with pre-packed matrices

### In-Place Matrix Transformations
*Avoid memory allocations by transforming matrices in-place*

**In-place Copy/Transpose**: ~4 operations × 4 precisions = **16 operations**
- `mkl_?imatcopy`: In-place matrix transpose/copy with scaling
- `mkl_?imatcopy_batch` / `mkl_?imatcopy_batch_strided`: Batched in-place operations

**Out-of-place Copy/Transpose**: ~4 operations × 4 precisions = **16 operations**
- `mkl_?omatcopy`: Out-of-place matrix transpose/copy with scaling
- `mkl_?omatcopy2`: Two-strided copy (non-contiguous access)
- `mkl_?omatcopy_batch` / `mkl_?omatcopy_batch_strided`: Batched out-of-place
- `mkl_?omatadd`: Matrix addition with transpose/scaling (C := α*op(A) + β*op(B))

### Specialized GEMM Variants
**Triangular result only**: ~1 operation × 4 precisions = **4 operations**
- `cblas_?gemmt`: GEMM updating only triangular part of result

**Extended AXPY**:  **4 operations** (1 per precision)
- `cblas_?axpby`: y := α*x + β*y (two scalars instead of one)

### JIT (Just-In-Time) Compilation
**Code generation at runtime**: **~8 operations**
- `mkl_jit_create_?gemm`: Generate specialized GEMM kernel for specific parameters
- `mkl_jit_get_?gemm_ptr`: Retrieve function pointer to JIT-compiled kernel
- `mkl_jit_destroy`: Free JIT resources

### Tensor Operations with Fused Epilogues (Aliases)

⚠️ **All fused GEMM operations are aliases to `fb_gemm_unified()` with fusion flags.**

**Purpose**: Eliminate intermediate memory writes by fusing GEMM with activations/bias  
**Performance**: 2-5× faster for ML inference workloads (single kernel vs multiple passes)  
**Backends**: Intel oneMKL (matmul_post_ops), NVIDIA cuBLASLt, AMD hipBLASLt

**Fused GEMM Operations** (28 aliases):
```c
// GEMM + Activation (16 aliases - 4 activations × 4 precisions)
void fb_gemm_fused_relu(...)  // Alias → fb_gemm_unified(..., FB_FUSION_RELU, FB_FUSION_NONE, ...)
void fb_gemm_fused_gelu(...)  // Alias → fb_gemm_unified(..., FB_FUSION_GELU, FB_FUSION_NONE, ...)
void fb_gemm_fused_silu(...)  // Alias → fb_gemm_unified(..., FB_FUSION_SILU, FB_FUSION_NONE, ...)
void fb_gemm_fused_swish(...) // Alias → fb_gemm_unified(..., FB_FUSION_SWISH, FB_FUSION_NONE, ...)

// GEMM + Bias (4 aliases)
void fb_gemm_fused_bias_add(...)  // Alias → fb_gemm_unified(..., FB_FUSION_NONE, FB_FUSION_BIAS, ...)

// GEMM + Scale + Activation (8 aliases)
void fb_gemm_fused_scale_act(...)  // Alias → fb_gemm_unified(..., FB_FUSION_<ACT>, FB_FUSION_NONE, ..., scale_params)
```

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

**Tensor Contraction Operations**: ~**16 operations**
- `fb_tensor_contract`: Generalize matmul to multi-dimensional tensors (4 ops - s/d/c/z)
  - Example: `C[i,j,k] = sum_l (A[i,l,k] * B[j,l])`
- `fb_tensor_contract_batched`: Batched tensor contractions (4 ops)
- `fb_einsum`: Einstein summation notation (4 ops)
  - Example: `"ij,jk->ik"` (matmul), `"ii->i"` (diagonal)
- `fb_tensor_transpose_scale`: Fused transpose with scaling (4 ops)

**Extended Mixed-Precision**: ~**8 operations**
- `fb_gemm_fp8_e4m3/e5m2`: FP8 GEMM variants (4 ops)
  - E4M3: 4-bit exponent, 3-bit mantissa (better range)
  - E5M2: 5-bit exponent, 2-bit mantissa (better precision)
- `fb_gemm_int4`: INT4 GEMM for quantized inference (4 ops)

**TENSOR FUSION TOTAL**: ~**52 operations**

### Matrix Core Primitives (WMMA)

**Purpose**: Enable faster-blaster to **dynamically generate fused kernels** at runtime  
**Scope**: Device-side API for JIT kernel compilation  
**Use Cases**: Custom kernel fusion when backend doesn't provide needed operation

**NVIDIA WMMA (Warp Matrix Multiply-Accumulate)**:
- **Fragment Load/Store**: 12 operations
  - `wmma_load_matrix_sync_a/b/c`: Load from memory → registers (4 ops)
  - `wmma_store_matrix_sync_d`: Store result from registers (4 ops)
  - `wmma_fill_fragment`: Initialize accumulator (4 ops)
- **Matrix Multiply-Accumulate**: 4 operations
  - `wmma_mma_sync`: Core D = A*B + C on 16×16 tiles (4 ops - s/d/c/z)
- **Supported Sizes**: 16×16×16, 32×8×16, 8×32×16 (Ampere+)
- **Precisions**: FP16, BF16, TF32, FP64, INT8, INT4

**AMD rocWMMA (Wavefront Matrix Multiply-Accumulate)**:
- **Fragment Operations**: 12 operations
  - `rocwmma_load_matrix_sync` (4 ops)
  - `rocwmma_store_matrix_sync` (4 ops)
  - `rocwmma_fill_fragment` (4 ops)
- **Accumulate Operations**: 4 operations
  - `rocwmma_mma_sync`: Maps to MFMA instructions (4 ops)
- **Supported Sizes**: 16×16×16, 32×32×8 (CDNA2+)
- **Precisions**: FP16, BF16, FP32, FP64, INT8

**Kernel Fusion Examples**:
1. Multiple matmuls in sequence → fuse into single kernel
2. Matmul + activation chain → eliminate intermediate writes
3. Custom attention (Q*K + softmax + *V) → fused kernel
4. Sparse-dense hybrid → custom mixed operations

**WMMA PRIMITIVES TOTAL**: ~**32 operations** (device-side, JIT only)

**BLAS-LIKE EXTENSIONS COMPLETE TOTAL**: ~**264 operations**
- Batched: 104 operations
- Mixed-precision: 28 operations
- Matrix transformations: 32 operations
- Specialized variants: 8 operations
- JIT: 8 operations
- **Tensor fusion: 52 operations** (fused GEMM, contractions, einsum)
- **WMMA primitives: 32 operations** (device-side, for JIT kernel generation)

---

## SCALAPACK (Distributed LAPACK)

**Purpose**: Parallel linear algebra for distributed-memory systems (MPI-based)  
**Backend Support**: Intel MKL ScaLAPACK, AMD AOCL-ScaLAPACK, NVIDIA NVPL ScaLAPACK, NVIDIA cuBLASMp (PBLAS-like)  
**Target Audience**: HPC clusters, multi-node distributed computing  
**Requirements**: MPI library, BLACS (Basic Linear Algebra Communication Subprograms)

**Architecture**:
```
ScaLAPACK (p? prefix) → PBLAS → BLACS → MPI
                    ↓
                 LAPACK (local computation)
                    ↓
                  BLAS
```

**Naming Convention**: `p?` prefix (p = parallel/distributed)
- Example: `pdgemm` (distributed double-precision GEMM), `psgesv` (distributed single-precision linear system solver)

### ScaLAPACK Driver Routines

**Linear System Solvers**: ~18 operations × 4 precisions = **72 operations**
- `p?gesv`: General linear systems (LU factorization)
- `p?gesvx`: Expert driver with equilibration and error bounds
- `p?gbsv`, `p?gbtrf`, `p?gbtrs`: General banded systems
- `p?posv`, `p?posvx`: Symmetric/Hermitian positive definite
- `p?ppsv`: Positive definite packed storage
- `p?pbsv`: Positive definite banded
- `p?ptsv`: Positive definite tridiagonal
- `p?sysv`, `p?sysv`: Symmetric indefinite
- `p?hesv`: Hermitian indefinite (complex only, 2 precisions)

**Least Squares**: ~3 operations × 4 precisions = **12 operations**
- `p?gels`: Linear least squares (QR/LQ factorization)
- `p?gelsy`: Least squares using rank-revealing QR
- `p?gelss`: Least squares using SVD

**Eigenvalue Problems**: ~8 operations × 4 precisions = **32 operations**
- `p?syev` / `p?heev`: Symmetric/Hermitian eigenvalues + eigenvectors
- `p?syevx` / `p?heevx`: Selected eigenvalues
- `p?syevd` / `p?heevd`: Divide-and-conquer algorithm
- `p?sygv` / `p?hegv`: Generalized symmetric/Hermitian eigenproblems
- `p?geevx`: General nonsymmetric eigenvalue (expert driver)

**Singular Value Decomposition**: ~2 operations × 4 precisions = **8 operations**
- `p?gesvd`: Compute singular values and singular vectors
- `p?gesdd`: SVD using divide-and-conquer

**ScaLAPACK Drivers TOTAL**: ~**124 operations**

### ScaLAPACK Computational Routines

**Matrix Factorizations**: ~15 operations × 4 precisions = **60 operations**
- `p?getrf`: LU factorization (general matrices)
- `p?gbtrf`: LU factorization (banded)
- `p?potrf`: Cholesky factorization (positive definite)
- `p?pbtrf`: Cholesky (banded)
- `p?pttrf`: Cholesky (tridiagonal)
- `p?sytrf` / `p?hetrf`: Bunch-Kaufman factorization (symmetric/Hermitian indefinite)
- `p?getrf`: QR factorization
- `p?gerqf`: RQ factorization
- `p?geqlf`: QL factorization
- `p?gelqf`: LQ factorization
- `p?tzrzf`: RZ factorization (trapezoidal)

**Solving Factored Systems**: ~18 operations × 4 precisions = **72 operations**
- `p?getrs`, `p?gbtrs`, `p?potrs`, `p?pbtrs`, `p?pttrs`, `p?sytrs`, `p?hetrs`: Solve using factorizations

**Matrix Inversion**: ~3 operations × 4 precisions = **12 operations**
- `p?getri`: Invert general matrix (after LU)
- `p?potri`: Invert positive definite (after Cholesky)
- `p?trtri`: Invert triangular matrix

**Orthogonal/Unitary Factorizations**: ~8 operations × 4 precisions = **32 operations**
- `p?orgqr` / `p?ungqr`: Generate Q from QR
- `p?ormqr` / `p?unmqr`: Multiply by Q from QR
- Similar for RQ, QL, LQ factorizations

**Reductions**: ~12 operations × 4 precisions = **48 operations**
- `p?gehrd`: Reduce to Hessenberg form
- `p?sytrd` / `p?hetrd`: Reduce symmetric/Hermitian to tridiagonal
- `p?sygst` / `p?hegst`: Reduce generalized eigenproblem to standard form
- `p?gebrd`: Reduce general to bidiagonal
- `p?gghrd`: Reduce generalized eigenproblem pair to Hessenberg-triangular

**ScaLAPACK Computational TOTAL**: ~**224 operations**

### ScaLAPACK Auxiliary Routines

**Matrix Equilibration**: ~6 operations × 4 precisions = **24 operations**
- `p?geequ`, `p?poequ`, `p?syequb`, `p?heequb`: Compute row/column scaling

**Condition Estimation**: ~6 operations × 4 precisions = **24 operations**
- `p?gecon`, `p?pocon`, `p?trcon`: Estimate reciprocal condition number

**Error Bounds**: ~6 operations × 4 precisions = **24 operations**
- `p?gerfs`, `p?porfs`, `p?trrfs`: Refine solution and compute error bounds

**ScaLAPACK Auxiliary TOTAL**: ~**72 operations**

### PBLAS (Parallel BLAS) - Underlying Communication Layer

**PBLAS Level 1**: ~13 operations × 4 precisions = **52 operations**
- `p?swap`, `p?scal`, `p?copy`, `p?axpy`, `p?dot`, `p?nrm2`, `p?asum`, `p?amax`

**PBLAS Level 2**: ~22 operations × 4 precisions = **88 operations**
- `p?gemv`, `p?symv`, `p?trmv`, `p?trsv`, `p?ger`, `p?syr`, `p?syr2`, etc.

**PBLAS Level 3**: ~7 operations × 4 precisions = **28 operations**
- `p?gemm`, `p?symm`, `p?syrk`, `p?syr2k`, `p?trmm`, `p?trsm`

**PBLAS TOTAL**: ~**168 operations**

**SCALAPACK COMPLETE TOTAL**: ~**588 operations**
- Drivers: 124
- Computational: 224
- Auxiliary: 72
- PBLAS: 168

**Important Notes**:
1. **Separate module recommended**: ScaLAPACK requires MPI infrastructure
2. **Different programming model**: Distributed 2D block-cyclic data layout
3. **Process grid setup**: Users must configure `nprows × npcols` process grid via BLACS
4. **Target audience**: HPC users with multi-node clusters, not typical single-workstation users
5. **cuBLASMp alternative**: NVIDIA provides PBLAS-like interface (not full ScaLAPACK) with NCCL backend

---

## EXTENDED MATH FUNCTIONS

**Rationale**: Scientific computing and ML frequently need transcendental, special functions, and vector math alongside linear algebra. Bundling these reduces library management overhead.

### Vector Math Operations (~120 operations)

**Transcendental Functions** (60 ops):
- **Trigonometric**: `fb_vsin`, `fb_vcos`, `fb_vtan`, `fb_vasin`, `fb_vacos`, `fb_vatan` (24 ops - s/d/c/z)
- **Hyperbolic**: `fb_vsinh`, `fb_vcosh`, `fb_vtanh`, `fb_vasinh`, `fb_vacosh`, `fb_vatanh` (24 ops)
- **Exponential/Log**: `fb_vexp`, `fb_vlog`, `fb_vlog10`, `fb_vexp2`, `fb_vlog2` (20 ops)

**Power and Root Functions** (24 ops):
- `fb_vpow`, `fb_vsqrt`, `fb_vinvsqrt`, `fb_vcbrt`, `fb_vinvcbrt`, `fb_vpow2o3`, `fb_vpow3o2` (24 ops)

**Special Functions** (36 ops):
- **Error Function**: `fb_verf`, `fb_verfc`, `fb_verfinv` (12 ops)
- **Gamma Functions**: `fb_vgamma`, `fb_vlgamma` (8 ops)
- **Bessel Functions**: `fb_vj0`, `fb_vj1`, `fb_vy0`, `fb_vy1` (16 ops - cylindrical Bessel)

**Backends**:
- CPU: Intel MKL Vector Math Library (VML), AMD AOCL-LibM, Arm Performance Libraries
- GPU: CUDA Math Library (libcurand includes vector math), AMD rocRAND, SYCL math

**EXTENDED MATH TOTAL**: ~**120 operations**

---

## RANDOM NUMBER GENERATION

**Rationale**: LA algorithms need RNG for initialization (random matrices), stochastic methods (randomized SVD, Monte Carlo), and ML (dropout, data augmentation).

### RNG Distribution Operations (~48 operations)

**Continuous Distributions** (28 ops):
- **Uniform**: `fb_rng_uniform` (4 ops - s/d/c/z)
- **Gaussian/Normal**: `fb_rng_gaussian` (4 ops)
- **Log-normal**: `fb_rng_lognormal` (4 ops)
- **Exponential**: `fb_rng_exponential` (4 ops)
- **Cauchy**: `fb_rng_cauchy` (4 ops)
- **Beta**: `fb_rng_beta` (4 ops)
- **Gamma**: `fb_rng_gamma` (4 ops)

**Discrete Distributions** (12 ops):
- **Poisson**: `fb_rng_poisson` (4 ops)
- **Binomial**: `fb_rng_binomial` (4 ops)
- **Bernoulli**: `fb_rng_bernoulli` (4 ops)

**RNG State Management** (8 ops):
- **Create generator**: `fb_rng_create_philox/mt19937/xorwow` (4 ops)
- **Set seed**: `fb_rng_set_seed` (1 op)
- **Skip ahead**: `fb_rng_skip_ahead` (1 op)
- **Destroy**: `fb_rng_destroy` (1 op)
- **Get state size**: `fb_rng_get_state_size` (1 op)

**Algorithms**:
- **CPU**: Philox 4×32, MT19937, PCG, AES-NI (MKL, AOCL-RNG)
- **GPU**: Philox, XORWOW, MRG32k3a (cuRAND, rocRAND, oneMKL)
- **Quality**: All generators pass TestU01 BigCrush suite

**Backends**:
- Intel: oneMKL Random Number Generators
- AMD CPU: AOCL-RNG
- AMD GPU: rocRAND, hipRAND
- NVIDIA: cuRAND
- Portable: VSL (Vector Statistics Library) interface

**RNG TOTAL**: ~**48 operations**

---

## 9. Fast Fourier Transform (FFT) Operations

**Rationale**: FFT is essential for faster-blaster's target users:
- **ML practitioners**: Audio processing (speech recognition, music classification), time-series analysis (financial, sensor data), signal denoising, frequency-domain feature engineering
- **Scientific researchers**: Spectral methods for PDEs, fast convolution (O(n log n)), signal processing, frequency-domain analysis
- **Industry adoption**: All major backends provide comprehensive FFT APIs (Intel MKL, NVIDIA cuFFT, AMD rocFFT)
- **Dependency reduction**: Bundling FFT alongside BLAS/LAPACK reduces library management overhead (same rationale as RNG/Extended Math)

### 9.1 Transform Types

**Dimensions**: 1D, 2D, 3D (some backends support up to 7D)

**Transform Variants**:
- **Complex-to-Complex (C2C)**: `FFT_C2C`, `FFT_Z2Z` (single/double precision)
  - Forward and inverse transforms
  - Use: General-purpose frequency analysis, spectral filtering
- **Real-to-Complex (R2C)**: `FFT_R2C`, `FFT_D2Z`
  - Exploits Hermitian symmetry → produces N/2+1 coefficients for 1D
  - Use: Real-valued signals (audio, sensor data) - saves 50% memory
- **Complex-to-Real (C2R)**: `FFT_C2R`, `FFT_Z2D`
  - Inverse of R2C, reconstructs real signal from Hermitian spectrum
  - Use: Frequency-domain filtering with real-valued output

### 9.2 Core API Operations

**Lifecycle Operations** (~20 operations):
```c
// Plan/descriptor management (descriptor-based API like Sparse Inspector-Executor)
fb_fft_create_plan_1d(n, type, precision)              // Single dimension
fb_fft_create_plan_2d(nx, ny, type, precision)         // 2D (images, matrices)
fb_fft_create_plan_3d(nx, ny, nz, type, precision)    // 3D (volumes, simulations)
fb_fft_create_plan_many(rank, dims[], type, batch)    // Batched transforms

fb_fft_set_parameter(plan, param, value)               // Configure placement, scaling, stride
fb_fft_commit(plan)                                    // Finalize configuration (optimization)
fb_fft_destroy(plan)                                   // Free resources

// Estimate workspace size before allocation
fb_fft_estimate_workspace_1d/2d/3d()
```

**Execution Operations** (~12 operations):
```c
// Forward transforms
fb_fft_execute_forward_c2c(plan, in_complex, out_complex)
fb_fft_execute_forward_z2z(plan, in_complex_dp, out_complex_dp)
fb_fft_execute_forward_r2c(plan, in_real, out_complex)
fb_fft_execute_forward_d2z(plan, in_real_dp, out_complex_dp)

// Inverse transforms
fb_fft_execute_backward_c2c(plan, in_complex, out_complex)
fb_fft_execute_backward_z2z(plan, in_complex_dp, out_complex_dp)
fb_fft_execute_backward_c2r(plan, in_complex, out_real)
fb_fft_execute_backward_z2d(plan, in_complex_dp, out_real_dp)

// Combined forward/backward (single call for round-trip)
fb_fft_execute_round_trip_r2c2r(plan, in_real, out_real)
fb_fft_execute_round_trip_d2z2d(plan, in_real_dp, out_real_dp)
```

**Configuration Operations** (~8 operations):
```c
fb_fft_set_placement(plan, FB_FFT_INPLACE / FB_FFT_OUT_OF_PLACE)
fb_fft_set_scale(plan, forward_scale, backward_scale)  // Default: 1.0, 1/N
fb_fft_set_stride(plan, input_stride[], output_stride[])
fb_fft_set_normalization(plan, FB_FFT_NORMALIZE_FORWARD / BACKWARD / ORTHOGONAL)
fb_fft_get_optimal_size(n, &optimal_n)                 // Suggest power-of-2 or mixed-radix size
```

### 9.3 Extended Operations (GPU/Multi-Device)

**GPU Stream Integration** (~4 operations):
```c
fb_fft_set_stream(plan, stream)                        // Associate with compute stream (async execution)
fb_fft_wait_stream(plan)                               // Synchronize on FFT completion
```

**Multi-GPU Operations** (~8 operations):
```c
fb_fft_set_gpus(plan, device_ids[], num_devices)
fb_fft_execute_multi_gpu(plan, input_distributed, output_distributed)
fb_fft_malloc_distributed(plan, &device_pointers[], size)
fb_fft_free_distributed(device_pointers[])
```

**JIT Callbacks** (GPU kernels for load/store operations, ~4 operations):
```c
fb_fft_set_load_callback(plan, callback_fn, user_data)   // Custom preprocessing
fb_fft_set_store_callback(plan, callback_fn, user_data)  // Custom postprocessing
fb_fft_clear_callbacks(plan)
```

### 9.4 Batched Operations

**Use Case**: Process multiple independent FFTs simultaneously (e.g., batch of audio files, time-series windows)

```c
// Example: 100 independent 1D FFTs of length 1024
fb_fft_plan plan = fb_fft_create_plan_many(
    1,              // 1D transforms
    {1024},         // Length of each FFT
    FB_FFT_C2C,     // Complex-to-complex
    100,            // Batch count
    FB_FFT_SINGLE   // Single precision
);
fb_fft_execute_forward_c2c(plan, batch_input, batch_output);
```

**Batched Operations**: ~4 additional operations (already covered in `create_plan_many`)

### 9.5 Utility Operations

**Query Functions** (~8 operations):
```c
fb_fft_get_version(major, minor, patch)
fb_fft_get_backend_name()                               // "MKL FFT", "cuFFT", "rocFFT"
fb_fft_get_supported_transforms()                       // Bitmask: C2C | R2C | C2R
fb_fft_get_max_dimensions()                             // 3, 7, etc.
fb_fft_get_workspace_size(plan)                         // Query committed plan
fb_fft_is_size_optimal(n)                               // Check if n is power-of-2 or mixed-radix friendly
```

### 9.6 Backend Mapping

| Operation Category       | Intel MKL                              | NVIDIA cuFFT                      | AMD rocFFT/hipFFT                   |
| ------------------------ | -------------------------------------- | --------------------------------- | ----------------------------------- |
| **Plan Management**      | `DftiCreateDescriptor`                 | `cufftCreate`, `cufftPlan*`       | `hipfftCreate`, `hipfftPlan*`       |
| **Configuration**        | `DftiSetValue`, `DftiCommitDescriptor` | `cufftMakePlan*`, `cufftSet*`     | `hipfftMakePlan*`, `hipfftSet*`     |
| **Execution (C2C)**      | `DftiComputeForward/Backward`          | `cufftExecC2C`, `cufftExecZ2Z`    | `hipfftExecC2C`, `hipfftExecZ2Z`    |
| **Execution (R2C)**      | `DftiComputeForward`                   | `cufftExecR2C`, `cufftExecD2Z`    | `hipfftExecR2C`, `hipfftExecD2Z`    |
| **Execution (C2R)**      | `DftiComputeBackward`                  | `cufftExecC2R`, `cufftExecZ2D`    | `hipfftExecC2R`, `hipfftExecZ2D`    |
| **Multi-GPU**            | N/A (CPU library)                      | `cufftXt*` extended API           | `hipfftXt*` extended API            |
| **Workspace Estimation** | N/A (internal)                         | `cufftEstimate*`, `cufftGetSize*` | `hipfftEstimate*`, `hipfftGetSize*` |
| **Stream Integration**   | N/A                                    | `cufftSetStream`                  | `hipfftSetStream`                   |
| **FFTW Compatibility**   | FFTW3 wrappers included                | CUFFTW library                    | N/A (use hipFFT directly)           |
| **Lifecycle**            | `DftiFreeDescriptor`                   | `cufftDestroy`                    | `hipfftDestroy`                     |

**Notes**:
- **MKL FFT**: Multithreaded CPU execution, DFTI interface (descriptor-based), supports mixed-radix up to order 7, includes FFTW3 compatibility wrappers
- **cuFFT**: GPU execution, supports up to 128M elements (single precision), 64M (double), JIT callbacks for custom load/store
- **rocFFT**: Native HIP implementation for AMD GPUs, hipFFT provides cuFFT-compatible API
- **Precision**: All backends support single (`float`) and double (`double`) precision for all transform types

### 9.7 Example Usage

**1D Real-to-Complex FFT (Audio Processing)**:
```c
#include <faster-blaster/faster_blaster.h>

// Process 1-second audio clip at 44.1 kHz sampling rate
const int n = 44100;
float audio_signal[n];        // Time-domain input
float complex spectrum[n/2 + 1];  // Frequency-domain output (Hermitian symmetry)

// Create plan
fb_fft_plan plan = fb_fft_create_plan_1d(n, FB_FFT_R2C, FB_FFT_SINGLE);
fb_fft_set_normalization(plan, FB_FFT_NORMALIZE_FORWARD);  // Scale by 1/N
fb_fft_commit(plan);

// Execute transform
fb_fft_execute_forward_r2c(plan, audio_signal, spectrum);

// Analyze frequency spectrum: spectrum[i] corresponds to frequency i * (44100/n) Hz
// Example: spectrum[1000] ≈ 1000 Hz tone magnitude

// Cleanup
fb_fft_destroy(plan);
```

**2D FFT (Image Processing)**:
```c
// Process 512x512 grayscale image for frequency-domain filtering
const int width = 512, height = 512;
float complex image[height][width];
float complex filtered_image[height][width];

// Create 2D plan
fb_fft_plan plan = fb_fft_create_plan_2d(width, height, FB_FFT_C2C, FB_FFT_SINGLE);
fb_fft_set_placement(plan, FB_FFT_OUT_OF_PLACE);
fb_fft_commit(plan);

// Forward FFT
fb_fft_execute_forward_c2c(plan, image, filtered_image);

// Apply filter in frequency domain (e.g., low-pass filter)
// filtered_image[i][j] *= filter_kernel[i][j];

// Inverse FFT to get spatial domain result
fb_fft_execute_backward_c2c(plan, filtered_image, image);

fb_fft_destroy(plan);
```

**Batched 1D FFT (Time-Series Analysis)**:
```c
// Process 1000 time-series windows simultaneously
const int window_size = 256;
const int num_windows = 1000;
float complex time_series[num_windows][window_size];
float complex frequency_series[num_windows][window_size];

// Create batched plan
fb_fft_plan plan = fb_fft_create_plan_many(
    1,                // 1D FFTs
    &window_size,     // Size of each FFT
    FB_FFT_C2C,
    num_windows,      // Batch count
    FB_FFT_SINGLE
);
fb_fft_commit(plan);

// Execute all 1000 FFTs in single call (efficient batching)
fb_fft_execute_forward_c2c(plan, time_series, frequency_series);

fb_fft_destroy(plan);
```

### 9.8 Implementation Notes

**Performance Considerations**:
- **Optimal sizes**: Power-of-2 (2, 4, 8, ..., 2^20) fastest, mixed-radix (2^a × 3^b × 5^c) good, prime sizes slowest
- **Memory layout**: Contiguous storage critical for performance (use `fb_fft_set_stride` for non-contiguous)
- **Batching**: Process multiple FFTs together (10-100×) for GPU efficiency
- **In-place vs out-of-place**: In-place saves memory but may be slower; out-of-place safer for beginners
- **Normalization**: Convention varies:
  - MKL default: No scaling (user must divide by N for IFFT)
  - cuFFT default: No scaling
  - FFTW default: No forward scaling, 1/N on inverse
  - faster-blaster: Provide explicit `fb_fft_set_normalization()` to avoid confusion

**Thread Safety**: Plans are thread-safe for execution (multiple threads can call `execute` on same plan), but configuration (`set_*`) is NOT thread-safe.

**Precision Variants**: Single precision (`float`, `float complex`) sufficient for most ML/audio; double precision (`double`, `double complex`) for scientific computing.

**FFT TOTAL**: ~**68 operations**
- Lifecycle: 8 operations (create 1D/2D/3D/many, commit, destroy, estimate)
- Execution: 10 operations (forward/backward for C2C/Z2Z/R2C/D2Z/C2R/Z2D)
- Configuration: 5 operations (placement, scale, stride, normalization, optimal size)
- GPU extensions: 16 operations (stream control, multi-GPU setup/execution/memory, JIT callbacks)
- Batched: Covered by `create_plan_many`
- Utilities: 6 operations (version, backend name, capabilities query, workspace size, size optimization check)

**Backend Coverage**:
- ✅ Intel MKL: DFTI interface (CPU, multithreaded)
- ✅ NVIDIA cuFFT: Full GPU FFT with Xt extensions
- ✅ AMD rocFFT/hipFFT: HIP-based GPU FFT
- ⚠️ OpenBLAS: No FFT (BLAS-only library)
- ⚠️ BLIS: No FFT (BLAS-only library)
- ⚠️ Accelerate: vDSP provides FFT (Apple-specific API, can integrate)
- ⚠️ oneMKL: Includes FFT domain (SYCL-based, can integrate)

**Alternative Backends** (if primary backend lacks FFT):
- **FFTW3**: Cross-platform, well-tested, LGPL/GPL license
- **Pocket FFT**: Header-only C++, BSD license, good for embedded
- **KissFFT**: Simple, BSD license, slower but portable

---

## 10. Tensor Operations (hipTensor / cuTENSOR)

**Rationale**: Essential for faster-blaster's **custom kernel generation** and multi-dimensional array operations in scientific computing. When faster-blaster needs to fuse sequences of operations across devices, tensor contractions and permutations are fundamental building blocks.

**Use Cases**:
- Physics simulations with multi-dimensional arrays
- Quantum chemistry tensor network contractions
- Custom fused operations requiring complex data layouts
- Multi-device tensor distribution and re-arrangement

### 10.1 Core Operations

**Tensor Contractions** (~20 operations):
```c
// Einstein notation: D_{ijkl} = α · A_{ijmn} B_{mnkl} + β · C_{ijkl}
fb_tensor_contract(α, A, mode_A, B, mode_B, β, C, mode_C, descriptor)

// Specific contraction types
fb_tensor_gemm(α, A, B, β, C)                      // Matrix-matrix (2D tensors)
fb_tensor_ttm(α, A, M, mode, β, C)                 // Tensor-times-matrix
fb_tensor_ttv(α, A, v, mode, β, C)                 // Tensor-times-vector
fb_tensor_mttkrp(A, matrices[], modes[], output)   // Matricized tensor times Khatri-Rao product
fb_tensor_hadamard(α, A, B, β, C)                  // Element-wise multiplication
```

**Tensor Permutations** (~8 operations):
```c
// Transpose multi-dimensional arrays
fb_tensor_permute(input, perm[], output)           // General permutation: [0,1,2,3] → [0,2,1,3]
fb_tensor_transpose(input, dim1, dim2, output)     // Swap two dimensions

// Example: Convert NCHW → NHWC (batch, channels, height, width)
fb_tensor_permute(input_nchw, [0,2,3,1], output_nhwc)
```

**Tensor Reduction** (6 aliases):
```c
fb_tensor_reduce_sum(...)     // Alias → fb_reduce_unified(FB_REDUCE_SUM, FB_REDUCE_LOCAL, ..., axes)
fb_tensor_reduce_max(...)     // Alias → fb_reduce_unified(FB_REDUCE_MAX, FB_REDUCE_LOCAL, ..., axes)
fb_tensor_reduce_min(...)     // Alias → fb_reduce_unified(FB_REDUCE_MIN, FB_REDUCE_LOCAL, ..., axes)
fb_tensor_reduce_norm(...)    // Alias → fb_reduce_unified(FB_REDUCE_NORM_L2, FB_REDUCE_LOCAL, ..., axes)
```

→ **Unified Implementation**: See Section 6.3 - Unified Reduction

**Tensor Element-wise Operations** (~4 operations):
```c
fb_tensor_scale(α, A)                              // Scale all elements: A ← α·A
fb_tensor_add(α, A, β, B, C)                       // C ← α·A + β·B
fb_tensor_copy(src, dst)                           // Deep copy
```

**Tensor Reshaping** (~2 operations):
```c
fb_tensor_reshape(input, new_shape[], output)      // Change dimensions (same total elements)
fb_tensor_view(input, new_shape[], output)         // Zero-copy view (if contiguous)
```

### 10.2 Backend Mapping

| Operation Category | hipTensor (AMD)        | cuTENSOR (NVIDIA)           |
| ------------------ | ---------------------- | --------------------------- |
| **Contraction**    | `hiptensorContraction` | `cutensorContraction`       |
| **Permutation**    | `hiptensorPermutation` | `cutensorPermute`           |
| **Reduction**      | `hiptensorReduction`   | `cutensorReduce`            |
| **Element-wise**   | `hiptensorElementwise` | `cutensorElementwiseBinary` |

**TENSOR OPERATIONS TOTAL**: ~**40 operations**

---

## 11. Parallel Primitives (rocPRIM / rocThrust / hipCUB / CUB / Thrust)

**Rationale**: Essential building blocks for **custom GPU algorithm development**. When faster-blaster generates custom kernels to execute sequences of operations, these primitives provide the fundamental parallel patterns needed for data movement, aggregation, and reordering.

**Use Cases**:
- Custom sorting/filtering algorithms for data preprocessing
- Building complex reduction operations
- Histogram computation for data analysis
- Parallel scan for prefix sums (cumulative operations)

### 11.1 Aggregation Operations (15 aliases)

⚠️ **All reduce and scan operations are aliases to `fb_reduce_unified()` with scope/mode flags.**

**Reduce** (6 aliases - collapse sequence to single value):
```c
fb_prim_reduce(...)              // Alias → fb_reduce_unified(op, FB_REDUCE_LOCAL, ..., FB_SCAN_NONE)
fb_prim_reduce_by_key(...)       // Alias with segmented reduction

// Variants: device-wide, block-level, warp-level
fb_prim_reduce_device(...)       // Alias → fb_reduce_unified(..., device scope)
fb_prim_reduce_block(...)        // Alias → fb_reduce_unified(..., block scope)
fb_prim_reduce_warp(...)         // Alias → fb_reduce_unified(..., warp scope)
```

**Scan** (9 aliases - prefix sum / cumulative operations):
```c
fb_prim_scan_inclusive(...)      // Alias → fb_reduce_unified(op, ..., FB_SCAN_INCLUSIVE, NULL)
fb_prim_scan_exclusive(...)      // Alias → fb_reduce_unified(op, ..., FB_SCAN_EXCLUSIVE, init_value)
fb_prim_scan_by_key(...)         // Alias with segmented scan (by keys)
```

→ **Unified Implementation**: See Section 6.3 - Unified Reduction
fb_prim_scan_exclusive(input, output, size, op, init)  // output[i] = op(input[0..i-1])
fb_prim_scan_by_key(keys, values, output, size, op)    // Segmented scan

// Use case: Parallel stream compaction, radix sort
```

### 11.2 Basic Operations (~8 operations)

**Transform** (map operation):
```c
fb_prim_transform(input, output, size, unary_op)
fb_prim_transform_if(input, output, size, unary_op, predicate)
```

**Select** (filter elements):
```c
fb_prim_select_flagged(input, flags, output, size)  // Select where flag == 1
fb_prim_select_if(input, output, size, predicate)   // Select where predicate(x) == true
```

**Unique** (remove duplicates):
```c
fb_prim_unique(input, output, size)                 // Keep first occurrence
fb_prim_unique_by_key(keys, values, keys_out, values_out, size)
```

**Histogram** (statistical distribution):
```c
fb_prim_histogram_even(samples, histogram, size, levels, lower, upper)
fb_prim_histogram_range(samples, histogram, size, level_boundaries)
```

### 11.3 Reordering Operations (~20 operations)

**Sort**:
```c
fb_prim_sort_keys(input, output, size)              // Sort array
fb_prim_sort_pairs(keys, values, keys_out, values_out, size)  // Sort key-value pairs
fb_prim_radix_sort(input, output, size, begin_bit, end_bit)   // Fast integer sort
fb_prim_merge_sort(input, output, size)             // Stable sort

// Partial sort
fb_prim_partial_sort(input, output, size, k)        // Sort first k elements
fb_prim_nth_element(input, size, nth)               // Find nth smallest element
```

**Partition** (reorder based on predicate):
```c
fb_prim_partition(input, output, size, predicate)   // Move elements satisfying predicate to front
fb_prim_partition_three_way(input, output, size, pivot)  // < pivot | == pivot | > pivot
```

**Merge**:
```c
fb_prim_merge(input1, input2, output, size1, size2, comparator)
fb_prim_merge_by_key(keys1, vals1, keys2, vals2, keys_out, vals_out, sizes, comp)
```

### 11.4 Difference Operations (~4 operations)

```c
fb_prim_adjacent_difference(input, output, size, op)  // output[i] = input[i] - input[i-1]
fb_prim_discontinuity(input, flags, size, op)         // Find discontinuities in sequence
```

### 11.5 Data Movement Operations (~8 operations)

**Exchange** (block-level transpose):
```c
fb_prim_exchange_block_to_striped(input, output, block_size)
fb_prim_exchange_striped_to_block(input, output, block_size)
```

**Shuffle** (warp-level data movement):
```c
fb_prim_shuffle_up/down/xor(value, offset, width)  // Warp shuffle operations
fb_prim_shuffle_rotate(value, offset, width)
```

**Copy**:
```c
fb_prim_copy_if(input, output, size, predicate)     // Conditional copy (stream compaction)
fb_prim_gather(input, indices, output, size)        // output[i] = input[indices[i]]
fb_prim_scatter(input, indices, output, size)       // output[indices[i]] = input[i]
```

### 11.6 High-Level Thrust/rocThrust Operations (~15 operations)

```c
thrust::fill(begin, end, value)                     // Fill range with value
thrust::sequence(begin, end, init, step)            // Generate arithmetic sequence
thrust::generate(begin, end, generator)             // Generate values using functor

thrust::count(begin, end, value)                    // Count occurrences
thrust::count_if(begin, end, predicate)

thrust::find(begin, end, value)                     // Find first occurrence
thrust::binary_search(begin, end, value)            // Check if value exists (sorted data)
thrust::lower_bound(begin, end, value)              // Find insertion point
thrust::upper_bound(begin, end, value)

thrust::set_union(set1_begin, set1_end, set2_begin, set2_end, output)
thrust::set_intersection(...)
thrust::set_difference(...)

thrust::mismatch(begin1, end1, begin2)              // Find first difference
thrust::equal(begin1, end1, begin2)                 // Check equality
```

### 11.7 Backend Mapping

| Operation Category | rocPRIM (AMD)              | CUB (NVIDIA)                          | Thrust (Portable)        |
| ------------------ | -------------------------- | ------------------------------------- | ------------------------ |
| **Reduce**         | `rocprim::reduce`          | `cub::DeviceReduce::Reduce`           | `thrust::reduce`         |
| **Scan**           | `rocprim::inclusive_scan`  | `cub::DeviceScan::InclusiveScan`      | `thrust::inclusive_scan` |
| **Sort**           | `rocprim::radix_sort_keys` | `cub::DeviceRadixSort::SortKeys`      | `thrust::sort`           |
| **Histogram**      | `rocprim::histogram_even`  | `cub::DeviceHistogram::HistogramEven` | N/A (use reduce_by_key)  |
| **Select**         | `rocprim::select_if`       | `cub::DeviceSelect::If`               | `thrust::copy_if`        |
| **Partition**      | `rocprim::partition`       | `cub::DevicePartition::If`            | `thrust::partition`      |

### 11.8 Extended Primitives (New Operations)

These operations are implemented in faster-blaster-reference but were absent from the original spec. All typed variants follow the standard suffix convention: `_s` = float, `_d` = double, `_i32` = int32, `_u32` = uint32.

**Argument Reduction** (typed variants: s/d):
```c
fb_reduce_argmin_s(input, size, result_index)       // Index of minimum element (float)
fb_reduce_argmin_d(input, size, result_index)       // Index of minimum element (double)
fb_reduce_argmax_s(input, size, result_index)       // Index of maximum element (float)
fb_reduce_argmax_d(input, size, result_index)       // Index of maximum element (double)
```

**Norm Reductions** (typed variants: s/d):
```c
fb_reduce_norm1_s(input, size, result)              // L1 norm: Σ|x_i| (float)
fb_reduce_norm1_d(input, size, result)              // L1 norm: Σ|x_i| (double)
fb_reduce_norm2_s(input, size, result)              // L2 norm: √(Σx_i²) (float)
fb_reduce_norm2_d(input, size, result)              // L2 norm: √(Σx_i²) (double)
fb_reduce_norminf_s(input, size, result)            // Infinity norm: max|x_i| (float)
fb_reduce_norminf_d(input, size, result)            // Infinity norm: max|x_i| (double)
```

**Top-K and Index Sort** (typed variants: s/d):
```c
fb_topk_s(input, output, indices, size, k)          // Top-k largest elements + indices (float)
fb_topk_d(input, output, indices, size, k)          // Top-k largest elements + indices (double)
fb_argsort_s(input, indices_out, size)              // Sorted index permutation (float)
fb_argsort_d(input, indices_out, size)              // Sorted index permutation (double)
```

**Axis Reductions** (2-D tensor-like; typed variants: s/d):
```c
fb_reduce_axis_sum_s(input, output, rows, cols, axis)   // Sum along axis 0 or 1 (float)
fb_reduce_axis_sum_d(input, output, rows, cols, axis)   // Sum along axis 0 or 1 (double)
fb_reduce_axis_max_s(input, output, rows, cols, axis)   // Max along axis 0 or 1 (float)
fb_reduce_axis_max_d(input, output, rows, cols, axis)   // Max along axis 0 or 1 (double)
fb_reduce_axis_min_s(input, output, rows, cols, axis)   // Min along axis 0 or 1 (float)
fb_reduce_axis_min_d(input, output, rows, cols, axis)   // Min along axis 0 or 1 (double)
```

**Prefix Scan Variants** (beyond inclusive/exclusive sum; typed: s/d):
```c
fb_scan_min_s(input, output, size)                  // Prefix minimum (running min, float)
fb_scan_min_d(input, output, size)                  // Prefix minimum (running min, double)
fb_scan_max_s(input, output, size)                  // Prefix maximum (running max, float)
fb_scan_max_d(input, output, size)                  // Prefix maximum (running max, double)
fb_scan_prod_s(input, output, size)                 // Prefix product (running product, float)
fb_scan_prod_d(input, output, size)                 // Prefix product (running product, double)
```

**Scatter / Stream Compaction / Clamp** (typed: s/d):
```c
fb_scatter_add_s(values, indices, output, size, output_size)  // output[indices[i]] += values[i] (float)
fb_scatter_add_d(values, indices, output, size, output_size)  // output[indices[i]] += values[i] (double)
fb_compress_s(input, mask, output, out_size, size)            // Stream compaction; keep where mask[i]!=0 (float)
fb_compress_d(input, mask, output, out_size, size)            // Stream compaction (double)
fb_clamp_s(input, output, size, min_val, max_val)             // Clamp to [min, max] (float)
fb_clamp_d(input, output, size, min_val, max_val)             // Clamp to [min, max] (double)
```

**PARALLEL PRIMITIVES TOTAL**: ~**70 operations** (original) + **32 new operations (Section 11.8)** = ~**102 operations**

---

## 12. Collective Communications (NCCL / RCCL)

**Rationale**: **NCCL/RCCL vs ScaLAPACK serve different purposes**:
- **NCCL/RCCL**: GPU-to-GPU collective communications for ML training (345% faster than MPI for GPU operations)
- **ScaLAPACK**: CPU-based distributed linear algebra for HPC (thousands of nodes)
- **faster-blaster needs BOTH**: GPU collectives for multi-GPU ML, ScaLAPACK for traditional HPC

**Use Cases**:
- Multi-GPU training: Gradient averaging during backpropagation (AllReduce)
- Data parallelism: Distribute data across GPUs (Scatter), gather results (Gather)
- Model parallelism: Synchronize layer outputs across devices (Broadcast)
- Distributed inference: Combine predictions from multiple GPUs (ReduceScatter)

### 12.1 Core Collective Operations (3 reduction aliases + 7 other operations)

⚠️ **Reduction collectives are aliases to `fb_reduce_unified()` with FB_REDUCE_COLLECTIVE scope.**

**AllReduce** (1 alias - most common in ML training):
```c
// Sum/min/max across all GPUs → result on all GPUs
fb_nccl_allreduce(...)  // Alias → fb_reduce_unified(op, FB_REDUCE_COLLECTIVE, ..., comm, root=-1)

// Use case: Average gradients across 8 GPUs during distributed training
// Each GPU has local gradients → AllReduce with sum → each GPU has averaged gradients
```

**Reduce** (1 alias):
```c
// Reduce to root GPU only
fb_nccl_reduce(...)  // Alias → fb_reduce_unified(op, FB_REDUCE_COLLECTIVE, ..., comm, root)
```

**ReduceScatter** (1 alias):
```c
// Reduce + distribute chunks to all GPUs
fb_nccl_reducescatter(...)  // Alias → fb_reduce_unified(op, FB_REDUCE_COLLECTIVE, ..., scatter=true)

// Use case: Reduce-scatter followed by compute, then allgather (common in ring-allreduce)
```

→ **Unified Implementation**: See Section 6.3 - Unified Reduction

**Non-Reduction Collectives** (7 independent operations):

**Broadcast**:
```c
// Root GPU → all other GPUs
fb_nccl_broadcast(sendbuf, recvbuf, count, datatype, root, comm, stream)

// Use case: Distribute model weights from GPU 0 to all other GPUs
```

**AllGather**:
```c
// Gather from all GPUs → concatenate on all GPUs
fb_nccl_allgather(sendbuf, recvbuf, sendcount, datatype, comm, stream)

// Use case: Each GPU has different data batch → allgather to get full dataset on all GPUs
```

**Gather**:
```c
// All GPUs → root GPU (concatenate)
fb_nccl_gather(sendbuf, recvbuf, sendcount, datatype, root, comm, stream)
```

**Scatter**:
```c
// Root GPU → distribute different data to each GPU
fb_nccl_scatter(sendbuf, recvbuf, recvcount, datatype, root, comm, stream)
```

**AllToAll**:
```c
// Each GPU sends different data to each other GPU
fb_nccl_alltoall(sendbuf, recvbuf, count, datatype, comm, stream)

// Use case: Tensor model parallelism (redistribute sliced tensors)
```

### 12.2 Point-to-Point Operations (~2 operations)

```c
fb_nccl_send(buf, count, datatype, peer, comm, stream)
fb_nccl_recv(buf, count, datatype, peer, comm, stream)
```

### 12.3 Group Operations (~3 operations)

```c
fb_nccl_group_start()                               // Begin grouped operations
fb_nccl_group_end()                                 // Execute all grouped ops together
fb_nccl_comm_split(parent_comm, color, key, new_comm)  // Split communicator
```

### 12.4 Performance Characteristics

| Operation         | Use Case              | Typical Latency (8 GPUs)          |
| ----------------- | --------------------- | --------------------------------- |
| **AllReduce**     | Gradient averaging    | ~1-5 ms (NVLink), ~5-20 ms (PCIe) |
| **Broadcast**     | Model distribution    | ~0.5-2 ms                         |
| **AllGather**     | Data collection       | ~2-8 ms                           |
| **ReduceScatter** | Distributed optimizer | ~1-4 ms                           |

**Why NCCL/RCCL is faster than MPI for GPUs**:
- Single kernel handles communication + computation (no CPU involvement)
- Topology-aware routing (NVLink, NVSwitch, InfiniBand)
- GPU Direct RDMA (bypass host memory)
- Optimized ring/tree algorithms for GPU interconnects

**COLLECTIVE COMMUNICATIONS TOTAL**: ~**15 operations**

---

## 13. Deep Learning Primitives (cuDNN / MIOpen / oneDNN)

**Rationale**: Essential for **ML framework developers** building training/inference engines. These are the fundamental operations in neural networks — faster-blaster can optimize sequences of these operations or fuse them for better performance.

**Use Cases**:
- Custom neural network layer implementations
- Fused operation kernels (Conv+BatchNorm+ReLU)
- Inference optimization pipelines
- Training loop acceleration

### 13.1 Convolution Operations (~20 operations)

**Forward Convolution**:
```c
fb_dnn_conv2d_forward(input, weights, output, descriptor)
fb_dnn_conv3d_forward(input, weights, output, descriptor)  // 3D convolution (video, volumetric)

// Variants
fb_dnn_conv_grouped(...)                            // Grouped convolution (e.g., MobileNet)
fb_dnn_conv_depthwise(...)                          // Depthwise separable (1 filter per channel)
fb_dnn_conv_dilated(...)                            // Dilated/atrous convolution
fb_dnn_conv_transposed(...)                         // Deconvolution (upsampling)
```

**Backward Convolution**:
```c
fb_dnn_conv_backward_data(grad_output, weights, grad_input, descriptor)
fb_dnn_conv_backward_filter(input, grad_output, grad_weights, descriptor)
fb_dnn_conv_backward_bias(grad_output, grad_bias, descriptor)
```

**Algorithm Selection** (cuDNN specific):
```c
fb_dnn_conv_find_algorithm(...)                     // Auto-tune for best algorithm
// Algorithms: GEMM, Winograd, FFT, direct, implicit GEMM
```

### 13.2 Pooling Operations (~10 operations)

```c
fb_dnn_maxpool_forward(input, output, descriptor)   // Max pooling
fb_dnn_avgpool_forward(input, output, descriptor)   // Average pooling
fb_dnn_globalpool(input, output, mode)              // Global pooling (H×W → 1×1)
fb_dnn_adaptivepool(input, output, target_size)     // Adaptive pooling (arbitrary output size)

// Backward
fb_dnn_maxpool_backward(input, grad_output, indices, grad_input, descriptor)
fb_dnn_avgpool_backward(grad_output, grad_input, descriptor)
```

### 13.3 Activation Functions (~18 operations)

**Forward**:
```c
fb_dnn_relu(input, output)                          // ReLU: max(0, x)
fb_dnn_leaky_relu(input, output, negative_slope)    // LeakyReLU: max(αx, x)
fb_dnn_elu(input, output, alpha)                    // ELU: x if x>0, α(e^x - 1) if x≤0
fb_dnn_gelu(input, output)                          // GELU: x·Φ(x) (transformers)
fb_dnn_swish(input, output, beta)                   // Swish: x·sigmoid(βx)
fb_dnn_mish(input, output)                          // Mish: x·tanh(softplus(x))
fb_dnn_sigmoid(input, output)                       // Sigmoid: 1/(1+e^-x)
fb_dnn_tanh(input, output)                          // Tanh: (e^x - e^-x)/(e^x + e^-x)
fb_dnn_softplus(input, output)                      // Softplus: log(1 + e^x)
```

**Backward** (gradients):
```c
fb_dnn_relu_backward(input, grad_output, grad_input)
fb_dnn_leaky_relu_backward(input, grad_output, grad_input, negative_slope)
fb_dnn_elu_backward(input, grad_output, grad_input, alpha)
fb_dnn_gelu_backward(input, grad_output, grad_input)
fb_dnn_swish_backward(input, grad_output, grad_input, beta)
fb_dnn_mish_backward(input, grad_output, grad_input)
fb_dnn_sigmoid_backward(output, grad_output, grad_input)   // Uses output (not input) from forward pass
fb_dnn_tanh_backward(output, grad_output, grad_input)      // Uses output from forward pass
fb_dnn_softplus_backward(input, grad_output, grad_input)
```

### 13.4 Normalization Operations (12 aliases)

⚠️ **All normalization operations are aliases to `fb_normalize_unified()` with mode flags.**

**Batch Normalization** (3 aliases):
```c
fb_dnn_batchnorm_forward_training(...)   // Alias → fb_normalize_unified(FB_NORM_BATCH, FB_NORM_TRAINING, ...)
fb_dnn_batchnorm_forward_inference(...)  // Alias → fb_normalize_unified(FB_NORM_BATCH, FB_NORM_INFERENCE, ...)
fb_dnn_batchnorm_backward(...)           // Alias → fb_normalize_unified(FB_NORM_BATCH, FB_NORM_TRAINING, ..., grad_output)
```

**Layer Normalization** (2 aliases):
```c
fb_dnn_layernorm_forward(...)   // Alias → fb_normalize_unified(FB_NORM_LAYER, FB_NORM_INFERENCE, ...)
fb_dnn_layernorm_backward(...)  // Alias → fb_normalize_unified(FB_NORM_LAYER, FB_NORM_TRAINING, ..., grad_output)
```

**Instance Normalization** (1 alias):
```c
fb_dnn_instancenorm_forward(...)  // Alias → fb_normalize_unified(FB_NORM_INSTANCE, FB_NORM_INFERENCE, ...)
```

**Group Normalization** (1 alias):
```c
fb_dnn_groupnorm_forward(...)  // Alias → fb_normalize_unified(FB_NORM_GROUP, FB_NORM_INFERENCE, ..., num_groups)
```

→ **Unified Implementation**: See Section 6.2 - Unified Normalization

### 13.5 Dropout (~2 operations)

```c
fb_dnn_dropout_forward(input, output, dropout_rate, seed)  // Training mode
fb_dnn_dropout_backward(grad_output, dropout_mask, grad_input)
```

### 13.6 Softmax (~4 operations)

```c
fb_dnn_softmax_forward(input, output, axis)         // Softmax: exp(x_i) / Σexp(x_j)
fb_dnn_logsoftmax_forward(input, output, axis)      // LogSoftmax: log(softmax(x))
fb_dnn_softmax_backward(output, grad_output, grad_input, axis)
fb_dnn_logsoftmax_backward(...)
```

### 13.7 Recurrent Operations (~15 operations)

**RNN/LSTM/GRU**:
```c
fb_dnn_rnn_forward(input, hidden, weights, output, new_hidden, descriptor)
fb_dnn_lstm_forward(input, hidden, cell, weights, output, new_hidden, new_cell, descriptor)
fb_dnn_gru_forward(input, hidden, weights, output, new_hidden, descriptor)

// Backward passes
fb_dnn_rnn_backward_data(...)
fb_dnn_rnn_backward_weights(...)
// ... LSTM/GRU backward variants
```

### 13.8 Attention Operations (~10 operations)

**Multi-Head Attention** (modern transformers):
```c
fb_dnn_scaled_dot_product_attention(Q, K, V, output, attention_mask, dropout)
// output = softmax(Q·K^T / √d_k) · V

fb_dnn_multi_head_attention(input, Q_weights, K_weights, V_weights, output, num_heads, ...)

// Flash Attention (memory-efficient)
fb_dnn_flash_attention_forward(Q, K, V, output)
fb_dnn_flash_attention_backward(...)
```

### 13.9 Fused Operations (~10 operations)

**Common Fusions** (single kernel for multiple ops):
```c
fb_dnn_conv_bias_relu(input, weights, bias, output)               // Conv → Bias → ReLU
fb_dnn_conv_batchnorm_relu(input, weights, bn_params, output)     // Conv → BN → ReLU
fb_dnn_linear_relu(input, weights, bias, output)                  // Linear → ReLU
fb_dnn_linear_gelu(input, weights, bias, output)                  // Linear → GELU (transformers)
fb_dnn_residual_block(input, weights, output)                     // Conv → BN → ReLU → Conv → BN → Add → ReLU
```

### 13.10 Backend Mapping

| Operation Category | cuDNN (NVIDIA)                           | MIOpen (AMD GPU)                          | oneDNN (Intel/Portable)            | ZenDNN (AMD EPYC CPU)                               |
| ------------------ | ---------------------------------------- | ----------------------------------------- | ---------------------------------- | --------------------------------------------------- |
| **Convolution**    | `cudnnConvolutionForward`                | `miopenConvolutionForward`                | `dnnl_convolution_forward`         | `dnnl_convolution_forward` (EPYC-optimized)         |
| **Pooling**        | `cudnnPoolingForward`                    | `miopenPoolingForward`                    | `dnnl_pooling_forward`             | `dnnl_pooling_forward` (EPYC-optimized)             |
| **Activation**     | `cudnnActivationForward`                 | `miopenActivationForward`                 | `dnnl_eltwise_forward`             | `dnnl_eltwise_forward` (EPYC-optimized)             |
| **BatchNorm**      | `cudnnBatchNormalizationForwardTraining` | `miopenBatchNormalizationForwardTraining` | `dnnl_batch_normalization_forward` | `dnnl_batch_normalization_forward` (EPYC-optimized) |
| **RNN/LSTM**       | `cudnnRNNForward`                        | `miopenRNNForward`                        | `dnnl_rnn_forward`                 | `dnnl_rnn_forward` (EPYC-optimized)                 |
| **Attention**      | `cudnnMultiHeadAttnForward`              | Via MIGraphX fusion                       | `dnnl_matmul + softmax fusion`     | SDPA with KV cache (EPYC cache hierarchy optimized) |
| **Fused Ops**      | cuDNN Graph API                          | MIOpen fusion                             | oneDNN graph API                   | oneDNN graph + zentorch.llm.optimize()              |

**Backend Notes**:
- **ZenDNN**: Compatible with oneDNN API but adds AMD EPYC-specific optimizations:
  - 400% performance uplift on EPYC 5th Gen (Turin, 192 cores) vs generic oneDNN
  - NUMA-aware multi-instance inference (physical core binding with `numactl`)
  - AOCL BLIS 5.0 integration for architecture-specific matmul kernels (Zen4/Zen5 AVX-512)
  - LLM optimizations: BF16:0 auto-tuning, INT4 WOQ (AWQ quantization), KV cache tuning
  - Large L3 cache optimization (32MB-384MB per socket)
  - Installation: `pip install zendnn` (PyTorch/TensorFlow plugins)

**DEEP LEARNING PRIMITIVES TOTAL**: ~**100 operations**

---

## 14. Data Fitting & Statistics (oneMKL / AOCL-DA)

**Rationale**: Essential for **data analysis, preprocessing, and interpolation** — operations ML practitioners need before/after linear algebra computations. Splines are ubiquitous in scientific computing for function approximation and data smoothing.

**Use Cases**:
- Interpolating sensor data with irregular sampling
- Curve fitting for experimental results
- Smoothing noisy measurements
- Computing derivatives/integrals of empirical functions

### 14.1 Spline Construction (~15 operations)

**1D Splines**:
```c
fb_spline_create_1d_linear(x[], y[], n, spline_handle)
fb_spline_create_1d_quadratic(x[], y[], n, boundary_type, spline_handle)
fb_spline_create_1d_cubic(x[], y[], n, boundary_type, spline_handle)  // Natural/clamped/periodic
fb_spline_create_1d_hermite(x[], y[], dy[], n, spline_handle)         // Hermite interpolation
fb_spline_create_1d_bessel(x[], y[], n, spline_handle)                // Bessel boundary conditions
fb_spline_create_1d_akima(x[], y[], n, spline_handle)                 // Akima spline (C1 continuous)
```

**2D Splines** (bivariate interpolation):
```c
fb_spline_create_2d_linear(x[], y[], z[][], nx, ny, spline_handle)
fb_spline_create_2d_cubic(x[], y[], z[][], nx, ny, boundary_type, spline_handle)
```

### 14.2 Spline Evaluation (~8 operations)

```c
fb_spline_eval(spline_handle, query_points[], output[], n_queries)
fb_spline_eval_derivative(spline_handle, query_points[], output[], order, n_queries)  // 1st, 2nd derivatives
fb_spline_eval_integral(spline_handle, a, b, result)                 // Definite integral ∫[a,b] S(x)dx

// Batch evaluation
fb_spline_eval_batch(spline_handle, query_points[], output[], n_queries, n_sites)
```

### 14.3 Spline Utilities (~5 operations)

```c
fb_spline_get_coefficients(spline_handle, coefficients[], n)
fb_spline_get_breakpoints(spline_handle, breakpoints[], n)
fb_spline_search_cell(spline_handle, x, cell_index)                  // Find which spline segment contains x
fb_spline_destroy(spline_handle)
```

### 14.4 Summary Statistics (~20 operations)

**Central Moments**:
```c
fb_stats_mean(data[], n, result)
fb_stats_variance(data[], n, mean, result)                           // Var = E[(X - μ)²]
fb_stats_std(data[], n, mean, result)                                // StdDev = √Var
fb_stats_skewness(data[], n, mean, variance, result)                 // Measure of asymmetry
fb_stats_kurtosis(data[], n, mean, variance, result)                 // Measure of tail heaviness

// Batch computation
fb_stats_raw_moments(data[], n, max_order, moments[])                // Compute E[X^k] for k=1..max_order
fb_stats_central_moments(data[], n, mean, max_order, moments[])
```

**Order Statistics**:
```c
fb_stats_min_max(data[], n, min, max)
fb_stats_median(data[], n, result)
fb_stats_quantile(data[], n, q, result)                              // q-th quantile (q ∈ [0,1])
fb_stats_quartiles(data[], n, Q1, Q2, Q3)
fb_stats_percentile(data[], n, p, result)                            // p-th percentile (p ∈ [0,100])
```

**Correlation & Covariance**:
```c
fb_stats_covariance(x[], y[], n, result)
fb_stats_covariance_matrix(data[][], n_samples, n_features, cov_matrix[][])
fb_stats_correlation(x[], y[], n, result)                            // Pearson correlation
fb_stats_correlation_matrix(data[][], n_samples, n_features, corr_matrix[][])
```

**Distribution Fitting**:
```c
fb_stats_histogram(data[], n, bins[], n_bins, histogram[])
fb_stats_kde(data[], n, bandwidth, kernel_type, kde_function)        // Kernel density estimation
```

**Outlier Detection** (2 aliases + 1 independent):
```c
fb_stats_zscore(...)          // Alias → fb_normalize_unified(FB_NORM_Z_SCORE, ..., mean, std)
fb_stats_modified_zscore(...) // Alias → fb_normalize_unified(FB_NORM_Z_SCORE, ..., median, MAD)
fb_stats_iqr_outliers(...)    // Independent operation (not normalization)
```

→ **Unified Implementation**: See Section 6.2 - Unified Normalization

### 14.5 Backend Mapping

| Operation Category | oneMKL                           | AOCL-DA               | GSL (Fallback)         |
| ------------------ | -------------------------------- | --------------------- | ---------------------- |
| **Spline 1D**      | `dfdNewTask1D`, `dfdConstruct1D` | AOCL-DA interpolation | `gsl_spline_alloc`     |
| **Spline Eval**    | `dfdInterpolate1D`               | AOCL-DA eval          | `gsl_spline_eval`      |
| **Statistics**     | `vslSSCompute` (VSL)             | AOCL-DA stats         | `gsl_stats_*`          |
| **Covariance**     | `vslSSComputeCov`                | AOCL-DA cov           | `gsl_stats_covariance` |

**DATA FITTING & STATISTICS TOTAL**: ~**48 operations**

---

## 15. Data Analytics & ML Algorithms (AOCL-DA)

**Rationale**: High-level ML/data science operations that BLAS/LAPACK users need in their workflows. Instead of implementing PCA or k-means from scratch using GEMM calls, faster-blaster provides optimized implementations.

**Use Cases**:
- Feature engineering for ML models
- Dimensionality reduction before neural network training
- Clustering for data exploration
- Regression analysis for scientific data

### 15.1 Dimensionality Reduction (~8 operations)

**Principal Component Analysis (PCA)**:
```c
fb_pca_fit(data[][], n_samples, n_features, n_components, pca_model)
fb_pca_transform(data[][], pca_model, transformed[][])               // Project to principal components
fb_pca_fit_transform(data[][], n_samples, n_features, n_components, transformed[][])
fb_pca_inverse_transform(transformed[][], pca_model, reconstructed[][])  // Reconstruct original space
fb_pca_explained_variance(pca_model, variances[], n_components)      // Variance explained by each PC
```

**Singular Value Decomposition** (for PCA):
```c
fb_svd_truncated(A[][], m, n, k, U[][], S[], Vt[][])                // Top-k singular values
```

### 15.2 Linear Models (~12 operations)

**Ordinary Least Squares (OLS)**:
```c
fb_linear_regression_fit(X[][], y[], n_samples, n_features, coef[], intercept)
fb_linear_regression_predict(X[][], coef[], intercept, n_samples, n_features, predictions[])
fb_linear_regression_score(X[][], y[], coef[], intercept, n_samples, n_features, r_squared)
```

**Regularized Regression**:
```c
fb_ridge_regression_fit(X[][], y[], n_samples, n_features, alpha, coef[], intercept)  // L2 regularization
fb_lasso_regression_fit(X[][], y[], n_samples, n_features, alpha, coef[], intercept)  // L1 regularization (sparse)
fb_elastic_net_fit(X[][], y[], n_samples, n_features, alpha, l1_ratio, coef[], intercept)  // L1 + L2
```

**Logistic Regression** (classification):
```c
fb_logistic_regression_fit(X[][], y[], n_samples, n_features, max_iter, coef[], intercept)
fb_logistic_regression_predict(X[][], coef[], intercept, n_samples, n_features, predictions[])
fb_logistic_regression_predict_proba(X[][], coef[], intercept, n_samples, n_features, probabilities[])
```

### 15.3 Clustering (~10 operations)

**K-Means**:
```c
fb_kmeans_fit(data[][], n_samples, n_features, k, max_iter, centroids[][], labels[])
fb_kmeans_predict(data[][], centroids[][], k, n_features, n_samples, labels[])
fb_kmeans_inertia(data[][], centroids[][], labels[], n_samples, n_features, k, inertia)  // Sum of squared distances
```

**DBSCAN** (density-based clustering):
```c
fb_dbscan_fit(data[][], n_samples, n_features, eps, min_samples, labels[], n_clusters)
```

**Hierarchical Clustering**:
```c
fb_hierarchical_fit(data[][], n_samples, n_features, linkage_type, distance_matrix[][])  // Linkage: single, complete, average
fb_hierarchical_cut(distance_matrix[][], n_samples, n_clusters, labels[])
```

**Agglomerative Clustering**:
```c
fb_agglomerative_fit(data[][], n_samples, n_features, n_clusters, linkage, labels[])
```

### 15.4 Decision Trees (~6 operations)

```c
fb_decision_tree_fit(X[][], y[], n_samples, n_features, max_depth, min_samples_split, tree_model)
fb_decision_tree_predict(X[][], tree_model, n_samples, n_features, predictions[])
fb_decision_tree_feature_importance(tree_model, importances[], n_features)
```

### 15.5 Data Preprocessing (3 aliases + 11 independent operations)

**Scaling/Normalization** (3 aliases):
```c
fb_standardize(...)    // Alias → fb_normalize_unified(FB_NORM_Z_SCORE, ..., mean, std)
fb_normalize(...)      // Alias → fb_normalize_unified(norm_type, ...) where norm_type ∈ {L1, L2, MAX}
fb_min_max_scale(...)  // Alias → fb_normalize_unified(FB_NORM_MIN_MAX_SCALE, ..., min, max)
```

→ **Unified Implementation**: See Section 6.2 - Unified Normalization

**Missing Value Imputation**:
```c
fb_impute_mean(data[][], mask[][], n_samples, n_features, imputed[][])
fb_impute_median(data[][], mask[][], n_samples, n_features, imputed[][])
fb_impute_knn(data[][], mask[][], n_samples, n_features, k, imputed[][])         // K-nearest neighbors imputation
```

**Feature Selection**:
```c
fb_feature_variance_threshold(data[][], n_samples, n_features, threshold, selected_indices[], n_selected)
fb_feature_univariate_selection(X[][], y[], n_samples, n_features, k_best, selected_indices[], scores[])
fb_feature_recursive_elimination(X[][], y[], n_samples, n_features, n_features_to_select, selected_indices[])
```

**Data Splitting**:
```c
fb_train_test_split(data[][], labels[], n_samples, test_size, shuffle, train_data[][], test_data[][], train_labels[], test_labels[])
fb_kfold_split(data[][], n_samples, n_folds, fold_indices[][])      // K-fold cross-validation
```

### 15.6 Backend Mapping

| Operation Category | AOCL-DA               | Alternatives                   |
| ------------------ | --------------------- | ------------------------------ |
| **PCA**            | AOCL-DA PCA           | oneMKL VSL, GSL                |
| **Linear Models**  | AOCL-DA regression    | Custom (BLAS + LAPACK)         |
| **K-Means**        | AOCL-DA clustering    | Custom (BLAS-based)            |
| **DBSCAN**         | AOCL-DA clustering    | Scikit-learn (CPU), cuML (GPU) |
| **Decision Trees** | AOCL-DA trees         | XGBoost, LightGBM              |
| **Preprocessing**  | AOCL-DA preprocessing | Custom (elementwise ops)       |

**DATA ANALYTICS TOTAL**: ~**50 operations**

---

## 16. Low-Precision GEMM & Quantization (Aliases to Unified Implementations)

⚠️ **Note**: All GEMM operations in this section are convenience aliases to `fb_gemm_unified()` with precision flags.

**Purpose**: Hardware-optimized low-precision matrix multiplication for inference acceleration. AOCL-DLP provides AMD EPYC-optimized implementations of quantized GEMM operations with fused element-wise operations, essential for deploying ML models in production.

### 16.1 Low-Precision GEMM (12 aliases)

```c
// LPGEMM: Low-Precision GEMM with mixed precision - aliases to fb_gemm_unified()
fb_lpgemm_fp32(...)           // Alias → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_SINGLE, 1, ...)
fb_lpgemm_bf16(...)           // Alias → fb_gemm_unified(FB_PREC_BF16, FB_BATCH_SINGLE, 1, ..., output_fp32)
fb_lpgemm_bf16_bf16(...)      // Alias → fb_gemm_unified(FB_PREC_BF16, FB_BATCH_SINGLE, 1, ..., output_bf16)
fb_lpgemm_int8(...)           // Alias → fb_gemm_unified(FB_PREC_INT8, FB_BATCH_SINGLE, 1, ..., output_int32)
fb_lpgemm_int8_fp32(...)      // Alias → fb_gemm_unified(FB_PREC_INT8, FB_BATCH_SINGLE, 1, ..., output_fp32, scaling)

// Batch GEMM - aliases to fb_gemm_unified() with FB_BATCH_ARRAY
fb_batch_gemm_fp32(...)       // Alias → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_ARRAY, batch_size, ...)
fb_batch_gemm_bf16(...)       // Alias → fb_gemm_unified(FB_PREC_BF16, FB_BATCH_ARRAY, batch_size, ...)
fb_batch_gemm_int8(...)       // Alias → fb_gemm_unified(FB_PREC_INT8, FB_BATCH_ARRAY, batch_size, ...)

// Strided batch GEMM - aliases to fb_gemm_unified() with FB_BATCH_STRIDED
fb_batch_gemm_strided_fp32(...)  // Alias → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_STRIDED, batch_size, ...)
fb_batch_gemm_strided_bf16(...)  // Alias → fb_gemm_unified(FB_PREC_BF16, FB_BATCH_STRIDED, batch_size, ...)
fb_batch_gemm_strided_int8(...)  // Alias → fb_gemm_unified(FB_PREC_INT8, FB_BATCH_STRIDED, batch_size, ...)

// Quantized GEMM with per-tensor or per-channel scaling
fb_qgemm_int8_asymmetric(...)    // Alias → fb_gemm_unified(FB_PREC_INT8, ..., quantization_params)
```

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

### 16.2 Fused GEMM Operations (18 aliases)

```c
// GEMM + Bias - aliases to fb_gemm_unified() with FB_FUSION_BIAS
fb_gemm_bias_fp32(...)        // Alias → fb_gemm_unified(FB_PREC_FP32, ..., FB_FUSION_NONE, FB_FUSION_BIAS, ...)
fb_gemm_bias_bf16(...)        // Alias → fb_gemm_unified(FB_PREC_BF16, ..., FB_FUSION_NONE, FB_FUSION_BIAS, ...)
fb_gemm_bias_int8(...)        // Alias → fb_gemm_unified(FB_PREC_INT8, ..., FB_FUSION_NONE, FB_FUSION_BIAS, ...)

// GEMM + Bias + Activation - aliases to fb_gemm_unified() with dual fusion
fb_gemm_bias_relu_fp32(...)   // Alias → fb_gemm_unified(FB_PREC_FP32, ..., FB_FUSION_RELU, FB_FUSION_BIAS, ...)
fb_gemm_bias_gelu_fp32(...)   // Alias → fb_gemm_unified(FB_PREC_FP32, ..., FB_FUSION_GELU, FB_FUSION_BIAS, ...)
fb_gemm_bias_sigmoid_fp32(...) // Alias → fb_gemm_unified(FB_PREC_FP32, ..., FB_FUSION_SIGMOID, FB_FUSION_BIAS, ...)
fb_gemm_bias_tanh_fp32(...)   // Alias → fb_gemm_unified(FB_PREC_FP32, ..., FB_FUSION_TANH, FB_FUSION_BIAS, ...)

// BF16 variants
fb_gemm_bias_relu_bf16(...)   // Alias → fb_gemm_unified(FB_PREC_BF16, ..., FB_FUSION_RELU, FB_FUSION_BIAS, ...)
fb_gemm_bias_gelu_bf16(...)   // Alias → fb_gemm_unified(FB_PREC_BF16, ..., FB_FUSION_GELU, FB_FUSION_BIAS, ...)
fb_gemm_bias_sigmoid_bf16(...) // Alias → fb_gemm_unified(FB_PREC_BF16, ..., FB_FUSION_SIGMOID, FB_FUSION_BIAS, ...)
fb_gemm_bias_tanh_bf16(...)   // Alias → fb_gemm_unified(FB_PREC_BF16, ..., FB_FUSION_TANH, FB_FUSION_BIAS, ...)

// INT8 quantized variants
fb_qgemm_bias_relu_int8(...)  // Alias → fb_gemm_unified(FB_PREC_INT8, ..., FB_FUSION_RELU, FB_FUSION_BIAS, ..., quantization)
fb_qgemm_bias_gelu_int8(...)  // Alias → fb_gemm_unified(FB_PREC_INT8, ..., FB_FUSION_GELU, FB_FUSION_BIAS, ..., quantization)

// Residual connection fusion
fb_gemm_bias_residual_relu_fp32(...)  // Alias → fb_gemm_unified(FB_PREC_FP32, ..., FB_FUSION_RELU, FB_FUSION_BIAS|FB_FUSION_RESIDUAL, ...)
fb_gemm_bias_residual_relu_bf16(...)  // Alias → fb_gemm_unified(FB_PREC_BF16, ..., FB_FUSION_RELU, FB_FUSION_BIAS|FB_FUSION_RESIDUAL, ...)

// Layer normalization fusion
fb_gemm_bias_layernorm_fp32(...)  // Alias → fb_gemm_unified(FB_PREC_FP32, ..., FB_FUSION_NONE, FB_FUSION_BIAS|FB_FUSION_LAYERNORM, ...)
fb_gemm_bias_layernorm_bf16(...)  // Alias → fb_gemm_unified(FB_PREC_BF16, ..., FB_FUSION_NONE, FB_FUSION_BIAS|FB_FUSION_LAYERNORM, ...)
```

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

**Backend**: AOCL-DLP (AMD EPYC), with future support for NVIDIA's INT8 Tensor Core operations, Intel AMX, ARM SVE

### 16.3 Extended Precision & Fusion GEMM (7 new operations)

These are **genuine new operations** (not aliases to `fb_gemm_unified()`) — they implement precision formats and fusion patterns not expressible via existing unified flags.

```c
// FP8 GEMM (H100 / MI300X tensor core format; stored as uint8)
fb_gemm_fp8_e4m3(A, B, C, m, n, k, alpha, beta)            // FP8 E4M3 inputs, FP8 output
fb_gemm_fp8_e4m3_fp32_out(A, B, C_fp32, m, n, k)           // FP8 compute, FP32 accumulation + output

// INT4 GEMM (GPTQ/AWQ quantized LLM inference)
fb_gemm_int4(A_int4, B_fp32, C, m, n, k, scales, zeros)    // 4-bit packed weights, FP32 activations
fb_gemm_int4_group_quant(A, B, C, m, n, k, scales, zeros, group_size)  // Per-group quantization

// SiLU gate fusion (LLaMA/Mistral FFN: output = A_gate * SiLU(A_up))
fb_gemm_fused_silu(A_gate, A_up, C, m, n, k)               // Element-wise SiLU gate fusion (float)
fb_gemm_fused_silu_bf16(A_gate, A_up, C, m, n, k)          // BF16 variant

// Generic GEMM + per-channel scale + activation dispatcher
fb_gemm_fused_scale_act(A, B, C, m, n, k, scale, activation_type)
// activation_type: FB_FUSION_RELU | FB_FUSION_GELU | FB_FUSION_SILU | FB_FUSION_SIGMOID | FB_FUSION_NONE
```

**LOW-PRECISION GEMM TOTAL**: **30 aliases + 7 new implementations = 37 total** (1 unified implementation in Section 6)

---

## 17. Geometric Deep Learning (cuEquivariance)

**Purpose**: Equivariant neural network operations that respect geometric symmetries (rotation, translation, reflection). Essential for molecular dynamics, protein structure prediction, materials science, and any domain where physical laws are orientation-independent.

### 17.1 Core Principles

**Equivariance**: If input transforms by group element g, output transforms predictably:
- **Translation equivariance**: f(x + t) = f(x) + t
- **Rotation equivariance**: f(Rx) = Rf(x)
- **Permutation equivariance**: f(π(x)) = π(f(x))

### 17.2 Spherical Harmonics & Tensor Products (8 operations)

```c
// Spherical harmonic basis functions (SO(3) irreducible representations)
fb_spherical_harmonics(l_max, positions[], n_points, Y_lm[])           // Compute Y_l^m(θ,φ)
fb_clebsch_gordan_coefficients(l1, l2, l3, cg_coeffs[])               // Coupling coefficients for tensor products

// Wigner D-matrices (rotation matrices in spherical harmonic basis)
fb_wigner_d_matrix(l, alpha, beta, gamma, D[])                         // D^l(α,β,γ) for SO(3) rotations

// Tensor product of irreducible representations
fb_irrep_tensor_product(x1[], l1, x2[], l2, cg_coeffs[], result[])    // x1 ⊗ x2

// Point cloud convolution (SE(3)-equivariant)
fb_se3_convolution(features_in[], positions[], neighbors[], edge_attr[], weights[], features_out[])

// Message passing on graphs with geometric features
fb_geometric_message_passing(node_features[], edge_index[], edge_vectors[], edge_features[], output[])

// Equivariant multi-layer perceptron
fb_equivariant_mlp(input[], irreps_in, weights[], biases[], irreps_out, output[])

// Spherical tensor to Cartesian conversion
fb_spherical_to_cartesian(Y_lm[], l_max, cartesian_tensor[])
```

### 17.3 Symmetry Operations (12 operations)

```c
// Rotation equivariant operations
fb_rotate_irreps(features[], l_max, rotation_matrix[][], rotated_features[])     // Apply SO(3) rotation
fb_rotate_spherical_harmonics(Y_lm[], l, alpha, beta, gamma, rotated_Y_lm[])    // Rotate Y_l^m basis

// Translation equivariant operations (already centered)
fb_center_of_mass(positions[], masses[], n_atoms, com[])                        // Compute COM
fb_translate_positions(positions[], n_atoms, translation[], translated[])

// Reflection symmetry
fb_reflect_positions(positions[], n_atoms, plane_normal[], reflected[])
fb_parity_transform(features[], parity[], reflected_features[])                 // Even/odd parity

// Permutation equivariance (atom ordering invariance)
fb_permutation_invariant_aggregation(features[], n_atoms, aggregated[])         // Sum/mean over atoms
fb_permutation_equivariant_attention(query[], key[], value[], n_atoms, output[]) // SE(3)-equivariant attention

// Group action on features
fb_group_action(features[], group_element[], irreps, transformed_features[])

// Equivariant pooling
fb_equivariant_pooling(features[], positions[], l_max, pooled_features[])

// Covariance matrix (O(3)-equivariant)
fb_covariance_matrix_equivariant(positions[], n_atoms, cov_matrix[][])

// Radial basis functions for distance encoding
fb_radial_basis_functions(distances[], n_distances, n_basis, cutoff, rbf[])
```

**Backend**: cuEquivariance (NVIDIA), e3nn (PyTorch/JAX, portable), potential future AMD/Intel implementations

**Use case**: Molecular ML potentials (force fields), protein folding, crystal structure prediction, quantum chemistry, climate modeling (spherical Earth geometry)

**GEOMETRIC DEEP LEARNING TOTAL**: ~**20 operations**

---

## 18. Computational Chemistry & Materials Science (ALCHEMI Toolkit-Ops)

**Purpose**: GPU-accelerated batch operations for atomistic simulations. NVIDIA ALCHEMI Toolkit-Ops provides high-throughput kernels for processing thousands of molecular systems simultaneously, essential for ML-driven molecular dynamics and materials discovery.

### 18.1 Neighbor Lists (3 operations)

```c
// Build neighbor list for atomic interactions (within cutoff radius)
fb_neighbor_list_allpairs(positions[], n_atoms, cutoff, cell[][], pbc[], batch_idx[], neighbors[][], num_neighbors[])
// O(N²) all-pairs distance check

fb_neighbor_list_celllist(positions[], n_atoms, cutoff, cell[][], pbc[], batch_idx[], neighbors[][], num_neighbors[])
// ~O(N) cell-list algorithm (spatial hashing)

fb_neighbor_list_verlet(positions[], n_atoms, cutoff, skin, verlet_list[][], num_neighbors[])
// Verlet list with skin distance for reuse across timesteps
```

### 18.2 Dispersion Corrections (4 operations)

```c
// DFT-D3: Grimme's D3 dispersion correction for DFT calculations
fb_dftd3_dispersion_energy(positions[], atomic_numbers[], n_atoms, cell[][], pbc[], 
                            c6_params[][], energy[])
fb_dftd3_dispersion_forces(positions[], atomic_numbers[], n_atoms, cell[][], pbc[], 
                            c6_params[][], forces[][])

// DFT-D4: Improved D4 dispersion model
fb_dftd4_dispersion_energy(positions[], atomic_numbers[], n_atoms, cell[][], pbc[], 
                            c6_params[][], polarizabilities[], energy[])
fb_dftd4_dispersion_forces(positions[], atomic_numbers[], n_atoms, cell[][], pbc[], 
                            c6_params[][], polarizabilities[], forces[][])
```

### 18.3 Electrostatics (8 operations)

```c
// Ewald summation for periodic systems
fb_ewald_energy(positions[], charges[], n_atoms, cell[][], pbc[], alpha, kmax, energy[])
fb_ewald_forces(positions[], charges[], n_atoms, cell[][], pbc[], alpha, kmax, forces[][])

// Particle-Mesh Ewald (PME) - more efficient for large systems
fb_pme_energy(positions[], charges[], n_atoms, cell[][], pbc[], grid_size[], 
              spline_order, alpha, energy[])
fb_pme_forces(positions[], charges[], n_atoms, cell[][], pbc[], grid_size[], 
              spline_order, alpha, forces[][])

// PME with automatic parameter tuning (accuracy-driven)
fb_pme_auto_energy(positions[], charges[], n_atoms, cell[][], pbc[], accuracy, energy[])
fb_pme_auto_forces(positions[], charges[], n_atoms, cell[][], pbc[], accuracy, forces[][])

// Direct Coulomb (for small systems or non-periodic)
fb_coulomb_direct_energy(positions[], charges[], n_atoms, energy[])
fb_coulomb_direct_forces(positions[], charges[], n_atoms, forces[][])
```

**Backend**: NVIDIA ALCHEMI Toolkit-Ops (nvalchemi-toolkit-ops), with PyTorch integration

**Use case**: ML interatomic potentials (MLIP), ab initio molecular dynamics, crystal structure relaxation, drug discovery, catalyst design

**COMPUTATIONAL CHEMISTRY TOTAL**: ~**15 operations**

---

## COMPREHENSIVE OPERATION COUNT

⚠️ **Implementation vs API Surface**: Following the "Unified Implementation + Domain-Specific Aliases" pattern (Section 6), counts distinguish between unique implementations and zero-cost convenience aliases.

| Category                              | API Surface | Implementations | Aliases | Status in faster-blaster                                                                               |
| ------------------------------------- | ----------- | --------------- | ------- | ------------------------------------------------------------------------------------------------------ |
| **Standard BLAS**                     | 174         | 170             | 4       | ✅ Core specification (4 GEMM ops aliased to unified)                                                   |
| **Standard LAPACK**                   | 1488        | 1488            | 0       | ✅ Core specification                                                                                   |
| **Sparse BLAS (Inspector-Executor)**  | 132         | 132             | 0       | ⚠️ **Recommended extension** (modern interface, all backends)                                           |
| **Sparse Preconditioners**            | 80          | 80              | 0       | ⚠️ **Recommended extension** (rocALUTION-style, production solvers)                                     |
| **Structured Sparse (2:4)**           | 24          | 24              | 0       | ⚠️ **Recommended extension** (ML inference, tensor core acceleration)                                   |
| **Unified Linear Algebra (NEW)**      | 3           | 3               | 0       | ⚠️ **Unified implementations** (GEMM + Normalization + Reduction replace 122 duplicate implementations) |
| **BLAS-Like Extensions**              | 180         | 132             | 48      | ⚠️ **Recommended extension** (48 GEMM aliases, 132 other ops)                                           |
| **Tensor Fusion Operations**          | 52          | 24              | 28      | ⚠️ **Recommended extension** (28 fused GEMM aliases, 24 other ops)                                      |
| **WMMA Primitives**                   | 32          | 32              | 0       | ⚠️ **Recommended extension** (JIT kernel generation)                                                    |
| **Extended Math Functions**           | 120         | 120             | 0       | ⚠️ **Recommended extension** (scientific computing)                                                     |
| **Random Number Generation**          | 48          | 48              | 0       | ⚠️ **Recommended extension** (initialization, stochastic methods)                                       |
| **Fast Fourier Transform (FFT)**      | 68          | 68              | 0       | ⚠️ **Recommended extension** (audio, time-series, spectral methods)                                     |
| **Tensor Operations**                 | 40          | 34              | 6       | ⚠️ **Recommended extension** (6 reduction aliases, 34 other ops)                                        |
| **Parallel Primitives**               | 102         | 87              | 15      | ⚠️ **Recommended extension** (15 reduction/scan aliases, 55 original + 32 new Section 11.8 ops)         |
| **Collective Communications**         | 15          | 12              | 3       | ⚠️ **Recommended extension** (3 reduction aliases, 12 data movement ops)                                |
| **Deep Learning Primitives**          | 108         | 96              | 12      | ⚠️ **Recommended extension** (12 normalization aliases, 88 original + 8 new backward passes)            |
| **Data Fitting & Statistics**         | 48          | 46              | 2       | ⚠️ **Recommended extension** (2 z-score aliases, 46 other ops)                                          |
| **Data Analytics & ML Algorithms**    | 50          | 47              | 3       | ⚠️ **Recommended extension** (3 normalization aliases, 47 other ops)                                    |
| **Low-Precision GEMM & Quantization** | 37          | 7               | 30      | ⚠️ **30 GEMM aliases + 7 new FP8/INT4/SiLU implementations** (Section 16.3)                            |
| **Geometric Deep Learning**           | 20          | 20              | 0       | ⚠️ **Recommended extension** (equivariant networks, molecular ML)                                       |
| **Computational Chemistry**           | 15          | 15              | 0       | ⚠️ **Recommended extension** (atomistic simulations, materials science)                                 |
| **ScaLAPACK**                         | 588         | 588             | 0       | ⏸️ **Optional module** (distributed computing, requires MPI)                                            |
| **GRAND TOTAL (family-planning surface)** | **3489** | **3326** | **163** | **Family-planning view: 3489 operations** (3326 implementations + 163 zero-cost aliases)               |

**Key Metrics**:
- **Authoritative Concrete Surface**: 3054 judge-tracked names (Appendix A / `src/judge/judge_op_ids.h`)
- **Family-Planning Surface**: 3489 operations (conceptual design view combining grouped families, aliases, and optional modules) — *+47 added in Phase 4 spec expansion*
- **Unique Implementations**: 3326 operations (actual code to maintain)
- **Zero-Cost Aliases**: 163 operations (inline wrappers, zero overhead)
- **Duplication Eliminated**: 163 redundant implementations removed via unification pattern
  - GEMM unification: 81 aliases → 1 implementation (Section 6.1)
  - Normalization unification: 17 aliases → 1 implementation (Section 6.2)
  - Reduction unification: 24 aliases → 1 implementation (Section 6.3)
  - Other GEMM-related aliases: 40 aliases (batched/mixed-precision/fused in other sections)

**Scope Decision**:
- **Core faster-blaster (Phase 1)**: Standard BLAS/LAPACK (1662 operations)
- **Extensions (Phase 2 - State-of-the-art ML/Scientific Computing)**: **+1175 unique implementations**
  - Sparse BLAS Inspector-Executor: 132 operations
  - Sparse Preconditioners: 80 operations  
  - Structured Sparse (2:4): 24 operations
  - Unified Linear Algebra: 3 implementations (GEMM + Normalization + Reduction)
  - BLAS-Like Extensions (non-GEMM): 132 operations
  - Tensor Fusion Operations (non-GEMM): 24 operations
  - WMMA Primitives (JIT): 32 operations
  - Extended Math Functions: 120 operations
  - Random Number Generation: 48 operations
  - Fast Fourier Transform (FFT): 68 operations
  - Tensor Operations (non-reduction): 34 operations
  - Parallel Primitives (non-reduction): 55 operations
  - Collective Communications (non-reduction): 12 operations
  - Deep Learning Primitives (non-normalization): 88 operations
  - Data Fitting & Statistics (non-normalization): 46 operations
  - Data Analytics & ML Algorithms (non-normalization): 47 operations
  - Low-Precision GEMM & Quantization (AOCL-DLP): 30 operations
  - Geometric Deep Learning (cuEquivariance): 20 operations
  - Computational Chemistry (ALCHEMI): 15 operations
- **Distributed Module (Phase 3)**: ScaLAPACK (588 operations) - separate build target, requires MPI

**Why These Extensions Matter**:
1. **Sparse operations**: Production-quality iterative solvers need preconditioners (not just raw Krylov methods)
2. **Tensor fusion**: 2-5× speedup for ML inference by eliminating intermediate memory traffic
3. **WMMA primitives**: Enable runtime kernel generation for custom fused operations
4. **Structured sparse**: Hardware-accelerated 2:4 sparsity on modern GPUs (Ampere+, RDNA3+)
5. **Extended math/RNG**: Reduce library dependency hell - users need these alongside LA
6. **FFT**: Essential for ML (audio, time-series, signal processing) and scientific computing (spectral methods, convolution) - both are primary BLAS/LAPACK user communities
7. **Tensor operations**: Multi-dimensional contractions/permutations for custom kernel sequences across devices
8. **Parallel primitives**: GPU algorithm building blocks (scan, sort, reduce) for custom operations
9. **Collective communications**: Multi-GPU training (NCCL/RCCL faster than ScaLAPACK for dense GPU systems)
10. **DNN primitives**: Neural network layers (Conv, Pool, BatchNorm, ReLU) - fundamental for ML framework developers
11. **Data fitting/statistics**: Spline interpolation, summary statistics, correlation - data analysis before/after LA computations
12. **ML algorithms**: PCA, k-means, regression - high-level operations users need instead of raw GEMM calls
13. **Low-precision GEMM**: INT8/BF16 inference acceleration (3-4× faster than FP32) with fused activations for production ML deployment
14. **Geometric deep learning**: Equivariant networks for molecular dynamics, protein folding, materials science where physical symmetries are fundamental
15. **Computational chemistry**: GPU-accelerated neighbor lists, DFT-D3/D4 dispersion, PME electrostatics for ML-driven atomistic simulations

**Note on DNN backend diversity**: Section 13 provides unified API across 4 backends (cuDNN, MIOpen, oneDNN, ZenDNN) - faster-blaster automatically selects optimal backend based on hardware (NVIDIA GPU → cuDNN, AMD GPU → MIOpen, Intel CPU → oneDNN, AMD EPYC CPU → ZenDNN with 400% uplift)

**Why Sparse BLAS count changed from 600 → 236**:
- **Old approach**: Legacy format-specific API (15 ops × 5 formats × 4 precisions = 300+ deprecated operations)
- **Modern approach**: Inspector-Executor interface (format-agnostic, 132 operations) + preconditioners (80) + structured sparse (24)
- **Industry direction**: All backends (MKL, cuSPARSE, rocSPARSE, AOCL) have deprecated legacy Sparse BLAS
- **faster-blaster decision**: Modern APIs only - Inspector-Executor + production solver stack

---

## Recommended vtable Structure

```c
// In fb_backend.h or backend header

typedef struct {
    // BLAS Level 1 (54 pointers)
    void (*blas_l1[54])(void);
    
    // BLAS Level 2 (90 pointers)
    void (*blas_l2[90])(void);
    
    // BLAS Level 3 (30 pointers)
    void (*blas_l3[30])(void);
    
    // LAPACK Drivers (264 pointers)
    void (*lapack_drivers[264])(void);
    
    // LAPACK Computational (840 pointers)
    void (*lapack_computational[840])(void);
    
    // LAPACK Auxiliary (384 pointers)
    void (*lapack_auxiliary[384])(void);
    
    // TOTAL: 1662 standard BLAS/LAPACK core pointers in the legacy core layout
} fb_backend_vtable;

// Optional extensions (Phase 2)
typedef struct {
    // Sparse BLAS operations (Inspector-Executor, ~132 pointers)
    void (*sparse_inspector[40])(void);   // Matrix lifecycle & optimization
    void (*sparse_executor[52])(void);    // SpMV, SpMM, sparse-sparse ops
    void (*sparse_solvers[40])(void);     // Krylov iterative solvers
    
    // BLAS-like extensions (~180 pointers)
    void (*blas_batched[104])(void);      // Batched Level 1/2/3 operations
    void (*blas_mixed_precision[28])(void); // BF16, FP16, FP8, INT8 GEMM
    void (*blas_transforms[32])(void);    // imatcopy, omatcopy, omatadd
    void (*blas_specialized[8])(void);    // gemmt, axpby, gemm3m
    void (*blas_jit[8])(void);            // JIT compilation routines
    
    // FFT operations (~68 pointers)
    void (*fft_lifecycle[8])(void);       // Plan creation, commit, destroy, workspace estimation
    void (*fft_execute[10])(void);        // Forward/backward transforms (C2C, R2C, C2R for single/double)
    void (*fft_config[5])(void);          // Placement, scaling, stride, normalization
    void (*fft_gpu_ext[16])(void);        // Stream control, multi-GPU, JIT callbacks
    void (*fft_utility[6])(void);         // Version, backend query, capabilities, workspace size
} fb_backend_ext_vtable;

// Distributed computing (Phase 3 - separate module)
typedef struct {
    // ScaLAPACK operations (~588 pointers)
    void (*scalapack_drivers[124])(void);
    void (*scalapack_computational[224])(void);
    void (*scalapack_auxiliary[72])(void);
    void (*pblas[168])(void);              // Parallel BLAS
} fb_distributed_vtable;
```

### Phase 1 Implementation Targets

Based on practical impact and dependency chains, **design the vtable for all 1662 pointers** but implement Phase 1 strategically:

**BLAS Phase 1** (~50 operations):

- Level 1: All 54 (they're foundation for Level 2/3)
- Level 2: 14 GE (GEMV, GER, GERU, GERC) + 12 TR (TRMV, TRSV) + 4 SY (SYMV, SYR, SYR2 only) = 30 operations
- Level 3: 4 core (GEMM, TRMM, TRSM, SYMM)

**LAPACK Phase 1** (~80 operations):

- Core drivers: GESV, POSV, GELS, GEEV, SYEV, GESVD (6 × 4 = 24)
- Core computational: GETRF, GETRS, POTRF, POTRS, SYTRF, SYTRS, GEQRF, ORGQR, GEBRD, BDSDC (10 × 4 = 40)
- Specialized helpers: Essential band/packed variants (10)

**Phase 1 Total**: ~130 operations (with complete interface defined for all 1759)

**Remaining**: ~1629 operations start as NULL pointers, enable future expansion without breaking changes

---

## Implementation Strategy

1. **Define Complete vtable** with all 1759 pointers (initialize to NULL)
2. **Implement Graceful Degradation**: Operation not supported → clear error message
3. **Phase 1**: Fill ~130 pointers (BLAS L1, selected L2/L3, core LAPACK)
4. **Phases 2+**: Add operations incrementally without vtable restructuring

This allows:

- User requests operation X → either works or returns "not supported in backend Y"
- Adding operation X later → just fill the pointer
- No breaking API changes when expanding coverage

---

## Appendix A: Comprehensive List of Operations (No Wildcards)

To make implementation targets unambiguous, the complete wildcard-expanded operation list is provided in:

- [FASTER-BLASTER-OPERATIONS-LIST-APPENDIX.md](FASTER-BLASTER-OPERATIONS-LIST-APPENDIX.md)

This appendix enumerates concrete operation names with no wildcard tokens.

Normalization and expansion policy used:

- Wildcard expansion: `? -> {s, d, c, z}`
- Operation names are normalized to lowercase C symbol style
- Names are sorted lexicographically for deterministic lookup and comparison

### LAPACK Upstream Parity Additions (Auto-Generated 2026-06-09)

This section is generated from upstream Reference-LAPACK SRC/INSTALL routine filenames to maintain superset parity.

1. `CBBCSD` - Upstream LAPACK routine
2. `CGBBRD` - Upstream LAPACK routine
3. `CGBEQUB` - Upstream LAPACK routine
4. `CGBRFSX` - Upstream LAPACK routine
5. `CGBSVXX` - Upstream LAPACK routine
6. `CGBTF2` - Upstream LAPACK routine
7. `CGEDMD` - Upstream LAPACK routine
8. `CGEDMDQ` - Upstream LAPACK routine
9. `CGEEQUB` - Upstream LAPACK routine
10. `CGEJSV` - Upstream LAPACK routine
11. `CGELQ` - Upstream LAPACK routine
12. `CGELQT` - Upstream LAPACK routine
13. `CGELQT3` - Upstream LAPACK routine
14. `CGELST` - Upstream LAPACK routine
15. `CGEMLQ` - Upstream LAPACK routine
16. `CGEMLQT` - Upstream LAPACK routine
17. `CGEMQR` - Upstream LAPACK routine
18. `CGEMQRT` - Upstream LAPACK routine
19. `CGEQP3RK` - Upstream LAPACK routine
20. `CGEQR` - Upstream LAPACK routine
21. `CGEQR2` - Upstream LAPACK routine
22. `CGEQR2P` - Upstream LAPACK routine
23. `CGEQRFP` - Upstream LAPACK routine
24. `CGEQRT` - Upstream LAPACK routine
25. `CGEQRT2` - Upstream LAPACK routine
26. `CGEQRT3` - Upstream LAPACK routine
27. `CGERFSX` - Upstream LAPACK routine
28. `CGESC2` - Upstream LAPACK routine
29. `CGESVDQ` - Upstream LAPACK routine
30. `CGESVDX` - Upstream LAPACK routine
31. `CGESVJ` - Upstream LAPACK routine
32. `CGETC2` - Upstream LAPACK routine
33. `CGETF2` - Upstream LAPACK routine
34. `CGETRF2` - Upstream LAPACK routine
35. `CGETSLS` - Upstream LAPACK routine
36. `CGETSQRHRT` - Upstream LAPACK routine
37. `CGGBAK` - Upstream LAPACK routine
38. `CGGBAL` - Upstream LAPACK routine
39. `CGGES3` - Upstream LAPACK routine
40. `CGGEV3` - Upstream LAPACK routine
41. `CGGHD3` - Upstream LAPACK routine
42. `CGGQRF` - Upstream LAPACK routine
43. `CGGRQF` - Upstream LAPACK routine
44. `CGGSVD3` - Upstream LAPACK routine
45. `CGGSVP3` - Upstream LAPACK routine
46. `CGSVJ0` - Upstream LAPACK routine
47. `CGSVJ1` - Upstream LAPACK routine
48. `CGTTS2` - Upstream LAPACK routine
49. `CHB2ST_KERNELS` - Upstream LAPACK routine
50. `CHBEV_2STAGE` - Upstream LAPACK routine
51. `CHBEVD_2STAGE` - Upstream LAPACK routine
52. `CHBEVX_2STAGE` - Upstream LAPACK routine
53. `CHBGST` - Upstream LAPACK routine
54. `CHBTRD` - Upstream LAPACK routine
55. `CHECON_3` - Upstream LAPACK routine
56. `CHECON_ROOK` - Upstream LAPACK routine
57. `CHEEQUB` - Upstream LAPACK routine
58. `CHEEV_2STAGE` - Upstream LAPACK routine
59. `CHEEVD_2STAGE` - Upstream LAPACK routine
60. `CHEEVR_2STAGE` - Upstream LAPACK routine
61. `CHEEVX_2STAGE` - Upstream LAPACK routine
62. `CHEGS2` - Upstream LAPACK routine
63. `CHEGV_2STAGE` - Upstream LAPACK routine
64. `CHERFSX` - Upstream LAPACK routine
65. `CHESV_AA` - Upstream LAPACK routine
66. `CHESV_AA_2STAGE` - Upstream LAPACK routine
67. `CHESV_RK` - Upstream LAPACK routine
68. `CHESV_ROOK` - Upstream LAPACK routine
69. `CHESWAPR` - Upstream LAPACK routine
70. `CHETD2` - Upstream LAPACK routine
71. `CHETF2` - Upstream LAPACK routine
72. `CHETF2_RK` - Upstream LAPACK routine
73. `CHETF2_ROOK` - Upstream LAPACK routine
74. `CHETRD_2STAGE` - Upstream LAPACK routine
75. `CHETRD_HB2ST` - Upstream LAPACK routine
76. `CHETRD_HE2HB` - Upstream LAPACK routine
77. `CHETRF_AA` - Upstream LAPACK routine
78. `CHETRF_AA_2STAGE` - Upstream LAPACK routine
79. `CHETRF_RK` - Upstream LAPACK routine
80. `CHETRI2` - Upstream LAPACK routine
81. `CHETRI2X` - Upstream LAPACK routine
82. `CHETRI_3` - Upstream LAPACK routine
83. `CHETRI_3X` - Upstream LAPACK routine
84. `CHETRI_ROOK` - Upstream LAPACK routine
85. `CHETRS2` - Upstream LAPACK routine
86. `CHETRS_3` - Upstream LAPACK routine
87. `CHETRS_AA` - Upstream LAPACK routine
88. `CHETRS_AA_2STAGE` - Upstream LAPACK routine
89. `CHETRS_ROOK` - Upstream LAPACK routine
90. `CHFRK` - Upstream LAPACK routine
91. `CHLA_TRANSTYPE` - Upstream LAPACK routine
92. `CHPGST` - Upstream LAPACK routine
93. `CHPSVX` - Upstream LAPACK routine
94. `CHPTRD` - Upstream LAPACK routine
95. `CHSEIN` - Upstream LAPACK routine
96. `CLA_GBRCOND_C` - Upstream LAPACK routine
97. `CLA_GBRCOND_X` - Upstream LAPACK routine
98. `CLA_GBRFSX_EXTENDED` - Upstream LAPACK routine
99. `CLA_GBRPVGRW` - Upstream LAPACK routine
100. `CLA_GEAMV` - Upstream LAPACK routine
101. `CLA_GERCOND_C` - Upstream LAPACK routine
102. `CLA_GERCOND_X` - Upstream LAPACK routine
103. `CLA_GERFSX_EXTENDED` - Upstream LAPACK routine
104. `CLA_GERPVGRW` - Upstream LAPACK routine
105. `CLA_HEAMV` - Upstream LAPACK routine
106. `CLA_HERCOND_C` - Upstream LAPACK routine
107. `CLA_HERCOND_X` - Upstream LAPACK routine
108. `CLA_HERFSX_EXTENDED` - Upstream LAPACK routine
109. `CLA_HERPVGRW` - Upstream LAPACK routine
110. `CLA_LIN_BERR` - Upstream LAPACK routine
111. `CLA_PORCOND_C` - Upstream LAPACK routine
112. `CLA_PORCOND_X` - Upstream LAPACK routine
113. `CLA_PORFSX_EXTENDED` - Upstream LAPACK routine
114. `CLA_PORPVGRW` - Upstream LAPACK routine
115. `CLA_SYAMV` - Upstream LAPACK routine
116. `CLA_SYRCOND_C` - Upstream LAPACK routine
117. `CLA_SYRCOND_X` - Upstream LAPACK routine
118. `CLA_SYRFSX_EXTENDED` - Upstream LAPACK routine
119. `CLA_SYRPVGRW` - Upstream LAPACK routine
120. `CLA_WWADDW` - Upstream LAPACK routine
121. `CLACN2` - Upstream LAPACK routine
122. `CLACP2` - Upstream LAPACK routine
123. `CLADIV` - Upstream LAPACK routine
124. `CLAED0` - Upstream LAPACK routine
125. `CLAED7` - Upstream LAPACK routine
126. `CLAED8` - Upstream LAPACK routine
127. `CLAEV2` - Upstream LAPACK routine
128. `CLAG2Z` - Upstream LAPACK routine
129. `CLAHEF` - Upstream LAPACK routine
130. `CLAHEF_AA` - Upstream LAPACK routine
131. `CLAHEF_RK` - Upstream LAPACK routine
132. `CLAHEF_ROOK` - Upstream LAPACK routine
133. `CLAHR2` - Upstream LAPACK routine
134. `CLALS0` - Upstream LAPACK routine
135. `CLALSA` - Upstream LAPACK routine
136. `CLALSD` - Upstream LAPACK routine
137. `CLAMSWLQ` - Upstream LAPACK routine
138. `CLAMTSQR` - Upstream LAPACK routine
139. `CLANHF` - Upstream LAPACK routine
140. `CLANHT` - Upstream LAPACK routine
141. `CLAPLL` - Upstream LAPACK routine
142. `CLAPMR` - Upstream LAPACK routine
143. `CLAQHB` - Upstream LAPACK routine
144. `CLAQP2RK` - Upstream LAPACK routine
145. `CLAQP3RK` - Upstream LAPACK routine
146. `CLAQR0` - Upstream LAPACK routine
147. `CLAQR1` - Upstream LAPACK routine
148. `CLAQR2` - Upstream LAPACK routine
149. `CLAQR3` - Upstream LAPACK routine
150. `CLAQR4` - Upstream LAPACK routine
151. `CLAQR5` - Upstream LAPACK routine
152. `CLAQZ0` - Upstream LAPACK routine
153. `CLAQZ1` - Upstream LAPACK routine
154. `CLAQZ2` - Upstream LAPACK routine
155. `CLAQZ3` - Upstream LAPACK routine
156. `CLAR1V` - Upstream LAPACK routine
157. `CLAR2V` - Upstream LAPACK routine
158. `CLARCM` - Upstream LAPACK routine
159. `CLARF1F` - Upstream LAPACK routine
160. `CLARF1L` - Upstream LAPACK routine
161. `CLARFB_GETT` - Upstream LAPACK routine
162. `CLARFGP` - Upstream LAPACK routine
163. `CLARFT_LVL2` - Upstream LAPACK routine
164. `CLARFY` - Upstream LAPACK routine
165. `CLARRV` - Upstream LAPACK routine
166. `CLARSCL2` - Upstream LAPACK routine
167. `CLASCL2` - Upstream LAPACK routine
168. `CLASR` - Upstream LAPACK routine
169. `CLASWLQ` - Upstream LAPACK routine
170. `CLASYF` - Upstream LAPACK routine
171. `CLASYF_AA` - Upstream LAPACK routine
172. `CLASYF_RK` - Upstream LAPACK routine
173. `CLASYF_ROOK` - Upstream LAPACK routine
174. `CLATRS3` - Upstream LAPACK routine
175. `CLATSQR` - Upstream LAPACK routine
176. `CLAUNHR_COL_GETRFNP` - Upstream LAPACK routine
177. `CLAUNHR_COL_GETRFNP2` - Upstream LAPACK routine
178. `CPBSTF` - Upstream LAPACK routine
179. `CPBTF2` - Upstream LAPACK routine
180. `CPFTRF` - Upstream LAPACK routine
181. `CPFTRI` - Upstream LAPACK routine
182. `CPFTRS` - Upstream LAPACK routine
183. `CPOEQUB` - Upstream LAPACK routine
184. `CPORFSX` - Upstream LAPACK routine
185. `CPOTF2` - Upstream LAPACK routine
186. `CPOTRF2` - Upstream LAPACK routine
187. `CPSTF2` - Upstream LAPACK routine
188. `CPSTRF` - Upstream LAPACK routine
189. `CPTTS2` - Upstream LAPACK routine
190. `CRSCL` - Upstream LAPACK routine
191. `CSPSVX` - Upstream LAPACK routine
192. `CSTEMR` - Upstream LAPACK routine
193. `CSYCON_3` - Upstream LAPACK routine
194. `CSYCON_ROOK` - Upstream LAPACK routine
195. `CSYCONV` - Upstream LAPACK routine
196. `CSYCONVF` - Upstream LAPACK routine
197. `CSYCONVF_ROOK` - Upstream LAPACK routine
198. `CSYEQUB` - Upstream LAPACK routine
199. `CSYRFSX` - Upstream LAPACK routine
200. `CSYSV_AA` - Upstream LAPACK routine
201. `CSYSV_AA_2STAGE` - Upstream LAPACK routine
202. `CSYSV_RK` - Upstream LAPACK routine
203. `CSYSV_ROOK` - Upstream LAPACK routine
204. `CSYSWAPR` - Upstream LAPACK routine
205. `CSYTF2` - Upstream LAPACK routine
206. `CSYTF2_RK` - Upstream LAPACK routine
207. `CSYTF2_ROOK` - Upstream LAPACK routine
208. `CSYTRF_AA` - Upstream LAPACK routine
209. `CSYTRF_AA_2STAGE` - Upstream LAPACK routine
210. `CSYTRF_RK` - Upstream LAPACK routine
211. `CSYTRI2X` - Upstream LAPACK routine
212. `CSYTRI_3` - Upstream LAPACK routine
213. `CSYTRI_3X` - Upstream LAPACK routine
214. `CSYTRI_ROOK` - Upstream LAPACK routine
215. `CSYTRS2` - Upstream LAPACK routine
216. `CSYTRS_3` - Upstream LAPACK routine
217. `CSYTRS_AA` - Upstream LAPACK routine
218. `CSYTRS_AA_2STAGE` - Upstream LAPACK routine
219. `CSYTRS_ROOK` - Upstream LAPACK routine
220. `CTFSM` - Upstream LAPACK routine
221. `CTFTRI` - Upstream LAPACK routine
222. `CTFTTP` - Upstream LAPACK routine
223. `CTFTTR` - Upstream LAPACK routine
224. `CTPLQT` - Upstream LAPACK routine
225. `CTPLQT2` - Upstream LAPACK routine
226. `CTPMLQT` - Upstream LAPACK routine
227. `CTPMQRT` - Upstream LAPACK routine
228. `CTPQRT` - Upstream LAPACK routine
229. `CTPQRT2` - Upstream LAPACK routine
230. `CTPRFB` - Upstream LAPACK routine
231. `CTPTTF` - Upstream LAPACK routine
232. `CTPTTR` - Upstream LAPACK routine
233. `CTREVC3` - Upstream LAPACK routine
234. `CTREXC` - Upstream LAPACK routine
235. `CTRSEN` - Upstream LAPACK routine
236. `CTRSNA` - Upstream LAPACK routine
237. `CTRSYL` - Upstream LAPACK routine
238. `CTRSYL3` - Upstream LAPACK routine
239. `CTRTI2` - Upstream LAPACK routine
240. `CTRTTF` - Upstream LAPACK routine
241. `CTRTTP` - Upstream LAPACK routine
242. `CUNBDB` - Upstream LAPACK routine
243. `CUNBDB1` - Upstream LAPACK routine
244. `CUNBDB2` - Upstream LAPACK routine
245. `CUNBDB3` - Upstream LAPACK routine
246. `CUNBDB4` - Upstream LAPACK routine
247. `CUNBDB5` - Upstream LAPACK routine
248. `CUNBDB6` - Upstream LAPACK routine
249. `CUNCSD` - Upstream LAPACK routine
250. `CUNCSD2BY1` - Upstream LAPACK routine
251. `CUNG2R` - Upstream LAPACK routine
252. `CUNGTSQR` - Upstream LAPACK routine
253. `CUNGTSQR_ROW` - Upstream LAPACK routine
254. `CUNHR_COL` - Upstream LAPACK routine
255. `CUNM22` - Upstream LAPACK routine
256. `CUNM2L` - Upstream LAPACK routine
257. `CUNM2R` - Upstream LAPACK routine
258. `CUNML2` - Upstream LAPACK routine
259. `CUNMR2` - Upstream LAPACK routine
260. `CUNMTR` - Upstream LAPACK routine
261. `CUPGTR` - Upstream LAPACK routine
262. `CUPMTR` - Upstream LAPACK routine
263. `DBBCSD` - Upstream LAPACK routine
264. `DBDSVDX` - Upstream LAPACK routine
265. `DDISNA` - Upstream LAPACK routine
266. `DGBBRD` - Upstream LAPACK routine
267. `DGBEQUB` - Upstream LAPACK routine
268. `DGBRFSX` - Upstream LAPACK routine
269. `DGBSVXX` - Upstream LAPACK routine
270. `DGBTF2` - Upstream LAPACK routine
271. `DGEDMD` - Upstream LAPACK routine
272. `DGEDMDQ` - Upstream LAPACK routine
273. `DGEEQUB` - Upstream LAPACK routine
274. `DGEJSV` - Upstream LAPACK routine
275. `DGELQ` - Upstream LAPACK routine
276. `DGELQT` - Upstream LAPACK routine
277. `DGELQT3` - Upstream LAPACK routine
278. `DGELST` - Upstream LAPACK routine
279. `DGEMLQ` - Upstream LAPACK routine
280. `DGEMLQT` - Upstream LAPACK routine
281. `DGEMQR` - Upstream LAPACK routine
282. `DGEMQRT` - Upstream LAPACK routine
283. `DGEQP3RK` - Upstream LAPACK routine
284. `DGEQR` - Upstream LAPACK routine
285. `DGEQR2` - Upstream LAPACK routine
286. `DGEQR2P` - Upstream LAPACK routine
287. `DGEQRFP` - Upstream LAPACK routine
288. `DGEQRT` - Upstream LAPACK routine
289. `DGEQRT2` - Upstream LAPACK routine
290. `DGEQRT3` - Upstream LAPACK routine
291. `DGERFSX` - Upstream LAPACK routine
292. `DGESC2` - Upstream LAPACK routine
293. `DGESVDQ` - Upstream LAPACK routine
294. `DGESVDX` - Upstream LAPACK routine
295. `DGESVJ` - Upstream LAPACK routine
296. `DGETC2` - Upstream LAPACK routine
297. `DGETF2` - Upstream LAPACK routine
298. `DGETRF2` - Upstream LAPACK routine
299. `DGETSLS` - Upstream LAPACK routine
300. `DGETSQRHRT` - Upstream LAPACK routine
301. `DGGBAK` - Upstream LAPACK routine
302. `DGGBAL` - Upstream LAPACK routine
303. `DGGES3` - Upstream LAPACK routine
304. `DGGEV3` - Upstream LAPACK routine
305. `DGGHD3` - Upstream LAPACK routine
306. `DGGQRF` - Upstream LAPACK routine
307. `DGGRQF` - Upstream LAPACK routine
308. `DGGSVD3` - Upstream LAPACK routine
309. `DGGSVP3` - Upstream LAPACK routine
310. `DGSVJ0` - Upstream LAPACK routine
311. `DGSVJ1` - Upstream LAPACK routine
312. `DGTTS2` - Upstream LAPACK routine
313. `DHSEIN` - Upstream LAPACK routine
314. `DISNAN` - Upstream LAPACK routine
315. `DLA_GBRCOND` - Upstream LAPACK routine
316. `DLA_GBRFSX_EXTENDED` - Upstream LAPACK routine
317. `DLA_GBRPVGRW` - Upstream LAPACK routine
318. `DLA_GEAMV` - Upstream LAPACK routine
319. `DLA_GERCOND` - Upstream LAPACK routine
320. `DLA_GERFSX_EXTENDED` - Upstream LAPACK routine
321. `DLA_GERPVGRW` - Upstream LAPACK routine
322. `DLA_LIN_BERR` - Upstream LAPACK routine
323. `DLA_PORCOND` - Upstream LAPACK routine
324. `DLA_PORFSX_EXTENDED` - Upstream LAPACK routine
325. `DLA_PORPVGRW` - Upstream LAPACK routine
326. `DLA_SYAMV` - Upstream LAPACK routine
327. `DLA_SYRCOND` - Upstream LAPACK routine
328. `DLA_SYRFSX_EXTENDED` - Upstream LAPACK routine
329. `DLA_SYRPVGRW` - Upstream LAPACK routine
330. `DLA_WWADDW` - Upstream LAPACK routine
331. `DLACN2` - Upstream LAPACK routine
332. `DLAE2` - Upstream LAPACK routine
333. `DLAG2` - Upstream LAPACK routine
334. `DLAG2S` - Upstream LAPACK routine
335. `DLAHR2` - Upstream LAPACK routine
336. `DLAISNAN` - Upstream LAPACK routine
337. `DLALS0` - Upstream LAPACK routine
338. `DLALSA` - Upstream LAPACK routine
339. `DLALSD` - Upstream LAPACK routine
340. `DLAMCHF77` - Upstream LAPACK routine
341. `DLAMCHTST` - Upstream LAPACK routine
342. `DLAMRG` - Upstream LAPACK routine
343. `DLAMSWLQ` - Upstream LAPACK routine
344. `DLAMTSQR` - Upstream LAPACK routine
345. `DLANEG` - Upstream LAPACK routine
346. `DLANSF` - Upstream LAPACK routine
347. `DLANV2` - Upstream LAPACK routine
348. `DLAORHR_COL_GETRFNP` - Upstream LAPACK routine
349. `DLAORHR_COL_GETRFNP2` - Upstream LAPACK routine
350. `DLAPLL` - Upstream LAPACK routine
351. `DLAPMR` - Upstream LAPACK routine
352. `DLAQP2RK` - Upstream LAPACK routine
353. `DLAQP3RK` - Upstream LAPACK routine
354. `DLAQR3` - Upstream LAPACK routine
355. `DLAQR4` - Upstream LAPACK routine
356. `DLAQR5` - Upstream LAPACK routine
357. `DLAQTR` - Upstream LAPACK routine
358. `DLAQZ0` - Upstream LAPACK routine
359. `DLAQZ1` - Upstream LAPACK routine
360. `DLAQZ2` - Upstream LAPACK routine
361. `DLAQZ3` - Upstream LAPACK routine
362. `DLAQZ4` - Upstream LAPACK routine
363. `DLARF1F` - Upstream LAPACK routine
364. `DLARF1L` - Upstream LAPACK routine
365. `DLARFB_GETT` - Upstream LAPACK routine
366. `DLARFGP` - Upstream LAPACK routine
367. `DLARFT_LVL2` - Upstream LAPACK routine
368. `DLARFY` - Upstream LAPACK routine
369. `DLARGV` - Upstream LAPACK routine
370. `DLARMM` - Upstream LAPACK routine
371. `DLARSCL2` - Upstream LAPACK routine
372. `DLARTGP` - Upstream LAPACK routine
373. `DLARTGS` - Upstream LAPACK routine
374. `DLARUV` - Upstream LAPACK routine
375. `DLASCL2` - Upstream LAPACK routine
376. `DLASWLQ` - Upstream LAPACK routine
377. `DLASY2` - Upstream LAPACK routine
378. `DLASYF` - Upstream LAPACK routine
379. `DLASYF_AA` - Upstream LAPACK routine
380. `DLASYF_RK` - Upstream LAPACK routine
381. `DLASYF_ROOK` - Upstream LAPACK routine
382. `DLAT2S` - Upstream LAPACK routine
383. `DLATRS3` - Upstream LAPACK routine
384. `DLATSQR` - Upstream LAPACK routine
385. `DOPGTR` - Upstream LAPACK routine
386. `DOPMTR` - Upstream LAPACK routine
387. `DORBDB` - Upstream LAPACK routine
388. `DORBDB1` - Upstream LAPACK routine
389. `DORBDB2` - Upstream LAPACK routine
390. `DORBDB3` - Upstream LAPACK routine
391. `DORBDB4` - Upstream LAPACK routine
392. `DORBDB5` - Upstream LAPACK routine
393. `DORBDB6` - Upstream LAPACK routine
394. `DORCSD` - Upstream LAPACK routine
395. `DORCSD2BY1` - Upstream LAPACK routine
396. `DORG2R` - Upstream LAPACK routine
397. `DORGTSQR` - Upstream LAPACK routine
398. `DORGTSQR_ROW` - Upstream LAPACK routine
399. `DORHR_COL` - Upstream LAPACK routine
400. `DORM22` - Upstream LAPACK routine
401. `DORM2L` - Upstream LAPACK routine
402. `DORM2R` - Upstream LAPACK routine
403. `DORML2` - Upstream LAPACK routine
404. `DORMR2` - Upstream LAPACK routine
405. `DORMTR` - Upstream LAPACK routine
406. `DPBSTF` - Upstream LAPACK routine
407. `DPBTF2` - Upstream LAPACK routine
408. `DPFTRF` - Upstream LAPACK routine
409. `DPFTRI` - Upstream LAPACK routine
410. `DPFTRS` - Upstream LAPACK routine
411. `DPOEQUB` - Upstream LAPACK routine
412. `DPORFSX` - Upstream LAPACK routine
413. `DPOTF2` - Upstream LAPACK routine
414. `DPOTRF2` - Upstream LAPACK routine
415. `DPSTF2` - Upstream LAPACK routine
416. `DPSTRF` - Upstream LAPACK routine
417. `DPTTS2` - Upstream LAPACK routine
418. `DROUNDUP_LWORK` - Upstream LAPACK routine
419. `DSB2ST_KERNELS` - Upstream LAPACK routine
420. `DSBEV_2STAGE` - Upstream LAPACK routine
421. `DSBEVD_2STAGE` - Upstream LAPACK routine
422. `DSBEVX_2STAGE` - Upstream LAPACK routine
423. `DSBGST` - Upstream LAPACK routine
424. `DSBTRD` - Upstream LAPACK routine
425. `DSECND_EXT_ETIME` - Upstream LAPACK routine
426. `DSECND_EXT_ETIME_` - Upstream LAPACK routine
427. `DSECND_INT_CPU_TIME` - Upstream LAPACK routine
428. `DSECND_INT_ETIME` - Upstream LAPACK routine
429. `DSECND_NONE` - Upstream LAPACK routine
430. `DSECNDTST` - Upstream LAPACK routine
431. `DSFRK` - Upstream LAPACK routine
432. `DSGESV` - Upstream LAPACK routine
433. `DSPGST` - Upstream LAPACK routine
434. `DSPOSV` - Upstream LAPACK routine
435. `DSPTRD` - Upstream LAPACK routine
436. `DSTEMR` - Upstream LAPACK routine
437. `DSYCON_3` - Upstream LAPACK routine
438. `DSYCON_ROOK` - Upstream LAPACK routine
439. `DSYCONV` - Upstream LAPACK routine
440. `DSYCONVF` - Upstream LAPACK routine
441. `DSYCONVF_ROOK` - Upstream LAPACK routine
442. `DSYEQUB` - Upstream LAPACK routine
443. `DSYEV_2STAGE` - Upstream LAPACK routine
444. `DSYEVD_2STAGE` - Upstream LAPACK routine
445. `DSYEVR_2STAGE` - Upstream LAPACK routine
446. `DSYEVX_2STAGE` - Upstream LAPACK routine
447. `DSYGS2` - Upstream LAPACK routine
448. `DSYGV_2STAGE` - Upstream LAPACK routine
449. `DSYRFSX` - Upstream LAPACK routine
450. `DSYSV_AA` - Upstream LAPACK routine
451. `DSYSV_AA_2STAGE` - Upstream LAPACK routine
452. `DSYSV_RK` - Upstream LAPACK routine
453. `DSYSV_ROOK` - Upstream LAPACK routine
454. `DSYSWAPR` - Upstream LAPACK routine
455. `DSYTD2` - Upstream LAPACK routine
456. `DSYTF2` - Upstream LAPACK routine
457. `DSYTF2_RK` - Upstream LAPACK routine
458. `DSYTF2_ROOK` - Upstream LAPACK routine
459. `DSYTRD_2STAGE` - Upstream LAPACK routine
460. `DSYTRD_SB2ST` - Upstream LAPACK routine
461. `DSYTRD_SY2SB` - Upstream LAPACK routine
462. `DSYTRF_AA` - Upstream LAPACK routine
463. `DSYTRF_AA_2STAGE` - Upstream LAPACK routine
464. `DSYTRF_RK` - Upstream LAPACK routine
465. `DSYTRI2X` - Upstream LAPACK routine
466. `DSYTRI_3` - Upstream LAPACK routine
467. `DSYTRI_3X` - Upstream LAPACK routine
468. `DSYTRI_ROOK` - Upstream LAPACK routine
469. `DSYTRS2` - Upstream LAPACK routine
470. `DSYTRS_3` - Upstream LAPACK routine
471. `DSYTRS_AA` - Upstream LAPACK routine
472. `DSYTRS_AA_2STAGE` - Upstream LAPACK routine
473. `DSYTRS_ROOK` - Upstream LAPACK routine
474. `DTFSM` - Upstream LAPACK routine
475. `DTFTRI` - Upstream LAPACK routine
476. `DTFTTP` - Upstream LAPACK routine
477. `DTFTTR` - Upstream LAPACK routine
478. `DTPLQT` - Upstream LAPACK routine
479. `DTPLQT2` - Upstream LAPACK routine
480. `DTPMLQT` - Upstream LAPACK routine
481. `DTPMQRT` - Upstream LAPACK routine
482. `DTPQRT` - Upstream LAPACK routine
483. `DTPQRT2` - Upstream LAPACK routine
484. `DTPRFB` - Upstream LAPACK routine
485. `DTPTTF` - Upstream LAPACK routine
486. `DTPTTR` - Upstream LAPACK routine
487. `DTREVC3` - Upstream LAPACK routine
488. `DTREXC` - Upstream LAPACK routine
489. `DTRSEN` - Upstream LAPACK routine
490. `DTRSNA` - Upstream LAPACK routine
491. `DTRSYL` - Upstream LAPACK routine
492. `DTRSYL3` - Upstream LAPACK routine
493. `DTRTI2` - Upstream LAPACK routine
494. `DTRTTF` - Upstream LAPACK routine
495. `DTRTTP` - Upstream LAPACK routine
496. `IEEECK` - Upstream LAPACK routine
497. `ILACLC` - Upstream LAPACK routine
498. `ILACLR` - Upstream LAPACK routine
499. `ILADIAG` - Upstream LAPACK routine
500. `ILAENV2STAGE` - Upstream LAPACK routine
501. `ILAPREC` - Upstream LAPACK routine
502. `ILASLC` - Upstream LAPACK routine
503. `ILASLR` - Upstream LAPACK routine
504. `ILATRANS` - Upstream LAPACK routine
505. `ILAUPLO` - Upstream LAPACK routine
506. `ILAVER` - Upstream LAPACK routine
507. `ILAZLC` - Upstream LAPACK routine
508. `ILAZLR` - Upstream LAPACK routine
509. `IPARAM2STAGE` - Upstream LAPACK routine
510. `IPARMQ` - Upstream LAPACK routine
511. `IZMAX1` - Upstream LAPACK routine
512. `LA_CONSTANTS` - Upstream LAPACK routine
513. `LA_XISNAN` - Upstream LAPACK routine
514. `LAPACK_VERSION` - Upstream LAPACK routine
515. `LSAMEN` - Upstream LAPACK routine
516. `LSAMETST` - Upstream LAPACK routine
517. `SBBCSD` - Upstream LAPACK routine
518. `SBDSVDX` - Upstream LAPACK routine
519. `SDISNA` - Upstream LAPACK routine
520. `SECOND_EXT_ETIME` - Upstream LAPACK routine
521. `SECOND_EXT_ETIME_` - Upstream LAPACK routine
522. `SECOND_INT_CPU_TIME` - Upstream LAPACK routine
523. `SECOND_INT_ETIME` - Upstream LAPACK routine
524. `SECOND_NONE` - Upstream LAPACK routine
525. `SECONDTST` - Upstream LAPACK routine
526. `SGBBRD` - Upstream LAPACK routine
527. `SGBEQUB` - Upstream LAPACK routine
528. `SGBRFSX` - Upstream LAPACK routine
529. `SGBSVXX` - Upstream LAPACK routine
530. `SGBTF2` - Upstream LAPACK routine
531. `SGEDMD` - Upstream LAPACK routine
532. `SGEDMDQ` - Upstream LAPACK routine
533. `SGEEQUB` - Upstream LAPACK routine
534. `SGEJSV` - Upstream LAPACK routine
535. `SGELQ` - Upstream LAPACK routine
536. `SGELQT` - Upstream LAPACK routine
537. `SGELQT3` - Upstream LAPACK routine
538. `SGELST` - Upstream LAPACK routine
539. `SGEMLQ` - Upstream LAPACK routine
540. `SGEMLQT` - Upstream LAPACK routine
541. `SGEMQR` - Upstream LAPACK routine
542. `SGEMQRT` - Upstream LAPACK routine
543. `SGEQP3RK` - Upstream LAPACK routine
544. `SGEQR` - Upstream LAPACK routine
545. `SGEQR2` - Upstream LAPACK routine
546. `SGEQR2P` - Upstream LAPACK routine
547. `SGEQRFP` - Upstream LAPACK routine
548. `SGEQRT` - Upstream LAPACK routine
549. `SGEQRT2` - Upstream LAPACK routine
550. `SGEQRT3` - Upstream LAPACK routine
551. `SGERFSX` - Upstream LAPACK routine
552. `SGESC2` - Upstream LAPACK routine
553. `SGESVDQ` - Upstream LAPACK routine
554. `SGESVDX` - Upstream LAPACK routine
555. `SGESVJ` - Upstream LAPACK routine
556. `SGETC2` - Upstream LAPACK routine
557. `SGETF2` - Upstream LAPACK routine
558. `SGETRF2` - Upstream LAPACK routine
559. `SGETSLS` - Upstream LAPACK routine
560. `SGETSQRHRT` - Upstream LAPACK routine
561. `SGGBAK` - Upstream LAPACK routine
562. `SGGBAL` - Upstream LAPACK routine
563. `SGGES3` - Upstream LAPACK routine
564. `SGGEV3` - Upstream LAPACK routine
565. `SGGHD3` - Upstream LAPACK routine
566. `SGGQRF` - Upstream LAPACK routine
567. `SGGRQF` - Upstream LAPACK routine
568. `SGGSVD3` - Upstream LAPACK routine
569. `SGGSVP3` - Upstream LAPACK routine
570. `SGSVJ0` - Upstream LAPACK routine
571. `SGSVJ1` - Upstream LAPACK routine
572. `SGTTS2` - Upstream LAPACK routine
573. `SHSEIN` - Upstream LAPACK routine
574. `SISNAN` - Upstream LAPACK routine
575. `SLA_GBRCOND` - Upstream LAPACK routine
576. `SLA_GBRFSX_EXTENDED` - Upstream LAPACK routine
577. `SLA_GBRPVGRW` - Upstream LAPACK routine
578. `SLA_GEAMV` - Upstream LAPACK routine
579. `SLA_GERCOND` - Upstream LAPACK routine
580. `SLA_GERFSX_EXTENDED` - Upstream LAPACK routine
581. `SLA_GERPVGRW` - Upstream LAPACK routine
582. `SLA_LIN_BERR` - Upstream LAPACK routine
583. `SLA_PORCOND` - Upstream LAPACK routine
584. `SLA_PORFSX_EXTENDED` - Upstream LAPACK routine
585. `SLA_PORPVGRW` - Upstream LAPACK routine
586. `SLA_SYAMV` - Upstream LAPACK routine
587. `SLA_SYRCOND` - Upstream LAPACK routine
588. `SLA_SYRFSX_EXTENDED` - Upstream LAPACK routine
589. `SLA_SYRPVGRW` - Upstream LAPACK routine
590. `SLA_WWADDW` - Upstream LAPACK routine
591. `SLACN2` - Upstream LAPACK routine
592. `SLAE2` - Upstream LAPACK routine
593. `SLAG2` - Upstream LAPACK routine
594. `SLAG2D` - Upstream LAPACK routine
595. `SLAHR2` - Upstream LAPACK routine
596. `SLAISNAN` - Upstream LAPACK routine
597. `SLALS0` - Upstream LAPACK routine
598. `SLALSA` - Upstream LAPACK routine
599. `SLALSD` - Upstream LAPACK routine
600. `SLAMCHF77` - Upstream LAPACK routine
601. `SLAMCHTST` - Upstream LAPACK routine
602. `SLAMRG` - Upstream LAPACK routine
603. `SLAMSWLQ` - Upstream LAPACK routine
604. `SLAMTSQR` - Upstream LAPACK routine
605. `SLANEG` - Upstream LAPACK routine
606. `SLANSF` - Upstream LAPACK routine
607. `SLANV2` - Upstream LAPACK routine
608. `SLAORHR_COL_GETRFNP` - Upstream LAPACK routine
609. `SLAORHR_COL_GETRFNP2` - Upstream LAPACK routine
610. `SLAPLL` - Upstream LAPACK routine
611. `SLAPMR` - Upstream LAPACK routine
612. `SLAQP2RK` - Upstream LAPACK routine
613. `SLAQP3RK` - Upstream LAPACK routine
614. `SLAQR3` - Upstream LAPACK routine
615. `SLAQR4` - Upstream LAPACK routine
616. `SLAQR5` - Upstream LAPACK routine
617. `SLAQTR` - Upstream LAPACK routine
618. `SLAQZ0` - Upstream LAPACK routine
619. `SLAQZ1` - Upstream LAPACK routine
620. `SLAQZ2` - Upstream LAPACK routine
621. `SLAQZ3` - Upstream LAPACK routine
622. `SLAQZ4` - Upstream LAPACK routine
623. `SLARF1F` - Upstream LAPACK routine
624. `SLARF1L` - Upstream LAPACK routine
625. `SLARFB_GETT` - Upstream LAPACK routine
626. `SLARFGP` - Upstream LAPACK routine
627. `SLARFT_LVL2` - Upstream LAPACK routine
628. `SLARFY` - Upstream LAPACK routine
629. `SLARGV` - Upstream LAPACK routine
630. `SLARMM` - Upstream LAPACK routine
631. `SLARSCL2` - Upstream LAPACK routine
632. `SLARTGP` - Upstream LAPACK routine
633. `SLARTGS` - Upstream LAPACK routine
634. `SLARUV` - Upstream LAPACK routine
635. `SLASCL2` - Upstream LAPACK routine
636. `SLASWLQ` - Upstream LAPACK routine
637. `SLASY2` - Upstream LAPACK routine
638. `SLASYF` - Upstream LAPACK routine
639. `SLASYF_AA` - Upstream LAPACK routine
640. `SLASYF_RK` - Upstream LAPACK routine
641. `SLASYF_ROOK` - Upstream LAPACK routine
642. `SLATRS3` - Upstream LAPACK routine
643. `SLATSQR` - Upstream LAPACK routine
644. `SOPGTR` - Upstream LAPACK routine
645. `SOPMTR` - Upstream LAPACK routine
646. `SORBDB` - Upstream LAPACK routine
647. `SORBDB1` - Upstream LAPACK routine
648. `SORBDB2` - Upstream LAPACK routine
649. `SORBDB3` - Upstream LAPACK routine
650. `SORBDB4` - Upstream LAPACK routine
651. `SORBDB5` - Upstream LAPACK routine
652. `SORBDB6` - Upstream LAPACK routine
653. `SORCSD` - Upstream LAPACK routine
654. `SORCSD2BY1` - Upstream LAPACK routine
655. `SORG2R` - Upstream LAPACK routine
656. `SORGTSQR` - Upstream LAPACK routine
657. `SORGTSQR_ROW` - Upstream LAPACK routine
658. `SORHR_COL` - Upstream LAPACK routine
659. `SORM22` - Upstream LAPACK routine
660. `SORM2L` - Upstream LAPACK routine
661. `SORM2R` - Upstream LAPACK routine
662. `SORML2` - Upstream LAPACK routine
663. `SORMR2` - Upstream LAPACK routine
664. `SORMTR` - Upstream LAPACK routine
665. `SPBSTF` - Upstream LAPACK routine
666. `SPBTF2` - Upstream LAPACK routine
667. `SPFTRF` - Upstream LAPACK routine
668. `SPFTRI` - Upstream LAPACK routine
669. `SPFTRS` - Upstream LAPACK routine
670. `SPOEQUB` - Upstream LAPACK routine
671. `SPORFSX` - Upstream LAPACK routine
672. `SPOTF2` - Upstream LAPACK routine
673. `SPOTRF2` - Upstream LAPACK routine
674. `SPSTF2` - Upstream LAPACK routine
675. `SPSTRF` - Upstream LAPACK routine
676. `SPTTS2` - Upstream LAPACK routine
677. `SROUNDUP_LWORK` - Upstream LAPACK routine
678. `SSB2ST_KERNELS` - Upstream LAPACK routine
679. `SSBEV_2STAGE` - Upstream LAPACK routine
680. `SSBEVD_2STAGE` - Upstream LAPACK routine
681. `SSBEVX_2STAGE` - Upstream LAPACK routine
682. `SSBGST` - Upstream LAPACK routine
683. `SSBTRD` - Upstream LAPACK routine
684. `SSFRK` - Upstream LAPACK routine
685. `SSPGST` - Upstream LAPACK routine
686. `SSPTRD` - Upstream LAPACK routine
687. `SSTEMR` - Upstream LAPACK routine
688. `SSYCON_3` - Upstream LAPACK routine
689. `SSYCON_ROOK` - Upstream LAPACK routine
690. `SSYCONV` - Upstream LAPACK routine
691. `SSYCONVF` - Upstream LAPACK routine
692. `SSYCONVF_ROOK` - Upstream LAPACK routine
693. `SSYEQUB` - Upstream LAPACK routine
694. `SSYEV_2STAGE` - Upstream LAPACK routine
695. `SSYEVD_2STAGE` - Upstream LAPACK routine
696. `SSYEVR_2STAGE` - Upstream LAPACK routine
697. `SSYEVX_2STAGE` - Upstream LAPACK routine
698. `SSYGS2` - Upstream LAPACK routine
699. `SSYGV_2STAGE` - Upstream LAPACK routine
700. `SSYRFSX` - Upstream LAPACK routine
701. `SSYSV_AA` - Upstream LAPACK routine
702. `SSYSV_AA_2STAGE` - Upstream LAPACK routine
703. `SSYSV_RK` - Upstream LAPACK routine
704. `SSYSV_ROOK` - Upstream LAPACK routine
705. `SSYSWAPR` - Upstream LAPACK routine
706. `SSYTD2` - Upstream LAPACK routine
707. `SSYTF2` - Upstream LAPACK routine
708. `SSYTF2_RK` - Upstream LAPACK routine
709. `SSYTF2_ROOK` - Upstream LAPACK routine
710. `SSYTRD_2STAGE` - Upstream LAPACK routine
711. `SSYTRD_SB2ST` - Upstream LAPACK routine
712. `SSYTRD_SY2SB` - Upstream LAPACK routine
713. `SSYTRF_AA_2STAGE` - Upstream LAPACK routine
714. `SSYTRF_RK` - Upstream LAPACK routine
715. `SSYTRI2X` - Upstream LAPACK routine
716. `SSYTRI_3` - Upstream LAPACK routine
717. `SSYTRI_3X` - Upstream LAPACK routine
718. `SSYTRI_ROOK` - Upstream LAPACK routine
719. `SSYTRS2` - Upstream LAPACK routine
720. `SSYTRS_3` - Upstream LAPACK routine
721. `SSYTRS_AA` - Upstream LAPACK routine
722. `SSYTRS_AA_2STAGE` - Upstream LAPACK routine
723. `SSYTRS_ROOK` - Upstream LAPACK routine
724. `STFSM` - Upstream LAPACK routine
725. `STFTRI` - Upstream LAPACK routine
726. `STFTTP` - Upstream LAPACK routine
727. `STFTTR` - Upstream LAPACK routine
728. `STPLQT` - Upstream LAPACK routine
729. `STPLQT2` - Upstream LAPACK routine
730. `STPMLQT` - Upstream LAPACK routine
731. `STPMQRT` - Upstream LAPACK routine
732. `STPQRT` - Upstream LAPACK routine
733. `STPQRT2` - Upstream LAPACK routine
734. `STPRFB` - Upstream LAPACK routine
735. `STPTTF` - Upstream LAPACK routine
736. `STPTTR` - Upstream LAPACK routine
737. `STREVC3` - Upstream LAPACK routine
738. `STREXC` - Upstream LAPACK routine
739. `STRSEN` - Upstream LAPACK routine
740. `STRSNA` - Upstream LAPACK routine
741. `STRSYL` - Upstream LAPACK routine
742. `STRSYL3` - Upstream LAPACK routine
743. `STRTI2` - Upstream LAPACK routine
744. `STRTTF` - Upstream LAPACK routine
745. `STRTTP` - Upstream LAPACK routine
746. `TEST_ZCOMPLEXABS` - Upstream LAPACK routine
747. `TEST_ZCOMPLEXDIV` - Upstream LAPACK routine
748. `TEST_ZCOMPLEXMULT` - Upstream LAPACK routine
749. `TEST_ZMINMAX` - Upstream LAPACK routine
750. `TSTIEE` - Upstream LAPACK routine
751. `XERBLA_ARRAY` - Upstream LAPACK routine
752. `ZBBCSD` - Upstream LAPACK routine
753. `ZCGESV` - Upstream LAPACK routine
754. `ZCPOSV` - Upstream LAPACK routine
755. `ZGBBRD` - Upstream LAPACK routine
756. `ZGBEQUB` - Upstream LAPACK routine
757. `ZGBRFSX` - Upstream LAPACK routine
758. `ZGBSVXX` - Upstream LAPACK routine
759. `ZGBTF2` - Upstream LAPACK routine
760. `ZGEDMD` - Upstream LAPACK routine
761. `ZGEDMDQ` - Upstream LAPACK routine
762. `ZGEEQUB` - Upstream LAPACK routine
763. `ZGEJSV` - Upstream LAPACK routine
764. `ZGELQ` - Upstream LAPACK routine
765. `ZGELQT` - Upstream LAPACK routine
766. `ZGELQT3` - Upstream LAPACK routine
767. `ZGELST` - Upstream LAPACK routine
768. `ZGEMLQ` - Upstream LAPACK routine
769. `ZGEMLQT` - Upstream LAPACK routine
770. `ZGEMQR` - Upstream LAPACK routine
771. `ZGEMQRT` - Upstream LAPACK routine
772. `ZGEQP3RK` - Upstream LAPACK routine
773. `ZGEQR` - Upstream LAPACK routine
774. `ZGEQR2` - Upstream LAPACK routine
775. `ZGEQR2P` - Upstream LAPACK routine
776. `ZGEQRFP` - Upstream LAPACK routine
777. `ZGEQRT` - Upstream LAPACK routine
778. `ZGEQRT2` - Upstream LAPACK routine
779. `ZGEQRT3` - Upstream LAPACK routine
780. `ZGERFSX` - Upstream LAPACK routine
781. `ZGESC2` - Upstream LAPACK routine
782. `ZGESVDQ` - Upstream LAPACK routine
783. `ZGESVDX` - Upstream LAPACK routine
784. `ZGESVJ` - Upstream LAPACK routine
785. `ZGETC2` - Upstream LAPACK routine
786. `ZGETF2` - Upstream LAPACK routine
787. `ZGETRF2` - Upstream LAPACK routine
788. `ZGETSLS` - Upstream LAPACK routine
789. `ZGETSQRHRT` - Upstream LAPACK routine
790. `ZGGBAK` - Upstream LAPACK routine
791. `ZGGBAL` - Upstream LAPACK routine
792. `ZGGES3` - Upstream LAPACK routine
793. `ZGGEV3` - Upstream LAPACK routine
794. `ZGGHD3` - Upstream LAPACK routine
795. `ZGGQRF` - Upstream LAPACK routine
796. `ZGGRQF` - Upstream LAPACK routine
797. `ZGGSVD3` - Upstream LAPACK routine
798. `ZGGSVP3` - Upstream LAPACK routine
799. `ZGSVJ0` - Upstream LAPACK routine
800. `ZGSVJ1` - Upstream LAPACK routine
801. `ZGTTS2` - Upstream LAPACK routine
802. `ZHB2ST_KERNELS` - Upstream LAPACK routine
803. `ZHBEV_2STAGE` - Upstream LAPACK routine
804. `ZHBEVD_2STAGE` - Upstream LAPACK routine
805. `ZHBEVX_2STAGE` - Upstream LAPACK routine
806. `ZHBGST` - Upstream LAPACK routine
807. `ZHBTRD` - Upstream LAPACK routine
808. `ZHECON_3` - Upstream LAPACK routine
809. `ZHECON_ROOK` - Upstream LAPACK routine
810. `ZHEEQUB` - Upstream LAPACK routine
811. `ZHEEV_2STAGE` - Upstream LAPACK routine
812. `ZHEEVD_2STAGE` - Upstream LAPACK routine
813. `ZHEEVR_2STAGE` - Upstream LAPACK routine
814. `ZHEEVX_2STAGE` - Upstream LAPACK routine
815. `ZHEGS2` - Upstream LAPACK routine
816. `ZHEGV_2STAGE` - Upstream LAPACK routine
817. `ZHERFSX` - Upstream LAPACK routine
818. `ZHESV_AA` - Upstream LAPACK routine
819. `ZHESV_AA_2STAGE` - Upstream LAPACK routine
820. `ZHESV_RK` - Upstream LAPACK routine
821. `ZHESV_ROOK` - Upstream LAPACK routine
822. `ZHESWAPR` - Upstream LAPACK routine
823. `ZHETD2` - Upstream LAPACK routine
824. `ZHETF2` - Upstream LAPACK routine
825. `ZHETF2_RK` - Upstream LAPACK routine
826. `ZHETF2_ROOK` - Upstream LAPACK routine
827. `ZHETRD_2STAGE` - Upstream LAPACK routine
828. `ZHETRD_HB2ST` - Upstream LAPACK routine
829. `ZHETRD_HE2HB` - Upstream LAPACK routine
830. `ZHETRF_AA` - Upstream LAPACK routine
831. `ZHETRF_AA_2STAGE` - Upstream LAPACK routine
832. `ZHETRF_RK` - Upstream LAPACK routine
833. `ZHETRI2` - Upstream LAPACK routine
834. `ZHETRI2X` - Upstream LAPACK routine
835. `ZHETRI_3` - Upstream LAPACK routine
836. `ZHETRI_3X` - Upstream LAPACK routine
837. `ZHETRI_ROOK` - Upstream LAPACK routine
838. `ZHETRS2` - Upstream LAPACK routine
839. `ZHETRS_3` - Upstream LAPACK routine
840. `ZHETRS_AA` - Upstream LAPACK routine
841. `ZHETRS_AA_2STAGE` - Upstream LAPACK routine
842. `ZHETRS_ROOK` - Upstream LAPACK routine
843. `ZHFRK` - Upstream LAPACK routine
844. `ZHPGST` - Upstream LAPACK routine
845. `ZHPSVX` - Upstream LAPACK routine
846. `ZHPTRD` - Upstream LAPACK routine
847. `ZHSEIN` - Upstream LAPACK routine
848. `ZLA_GBRCOND_C` - Upstream LAPACK routine
849. `ZLA_GBRCOND_X` - Upstream LAPACK routine
850. `ZLA_GBRFSX_EXTENDED` - Upstream LAPACK routine
851. `ZLA_GBRPVGRW` - Upstream LAPACK routine
852. `ZLA_GEAMV` - Upstream LAPACK routine
853. `ZLA_GERCOND_C` - Upstream LAPACK routine
854. `ZLA_GERCOND_X` - Upstream LAPACK routine
855. `ZLA_GERFSX_EXTENDED` - Upstream LAPACK routine
856. `ZLA_GERPVGRW` - Upstream LAPACK routine
857. `ZLA_HEAMV` - Upstream LAPACK routine
858. `ZLA_HERCOND_C` - Upstream LAPACK routine
859. `ZLA_HERCOND_X` - Upstream LAPACK routine
860. `ZLA_HERFSX_EXTENDED` - Upstream LAPACK routine
861. `ZLA_HERPVGRW` - Upstream LAPACK routine
862. `ZLA_LIN_BERR` - Upstream LAPACK routine
863. `ZLA_PORCOND_C` - Upstream LAPACK routine
864. `ZLA_PORCOND_X` - Upstream LAPACK routine
865. `ZLA_PORFSX_EXTENDED` - Upstream LAPACK routine
866. `ZLA_PORPVGRW` - Upstream LAPACK routine
867. `ZLA_SYAMV` - Upstream LAPACK routine
868. `ZLA_SYRCOND_C` - Upstream LAPACK routine
869. `ZLA_SYRCOND_X` - Upstream LAPACK routine
870. `ZLA_SYRFSX_EXTENDED` - Upstream LAPACK routine
871. `ZLA_SYRPVGRW` - Upstream LAPACK routine
872. `ZLA_WWADDW` - Upstream LAPACK routine
873. `ZLACN2` - Upstream LAPACK routine
874. `ZLACP2` - Upstream LAPACK routine
875. `ZLADIV` - Upstream LAPACK routine
876. `ZLAED0` - Upstream LAPACK routine
877. `ZLAED7` - Upstream LAPACK routine
878. `ZLAED8` - Upstream LAPACK routine
879. `ZLAEV2` - Upstream LAPACK routine
880. `ZLAG2C` - Upstream LAPACK routine
881. `ZLAHEF` - Upstream LAPACK routine
882. `ZLAHEF_AA` - Upstream LAPACK routine
883. `ZLAHEF_RK` - Upstream LAPACK routine
884. `ZLAHEF_ROOK` - Upstream LAPACK routine
885. `ZLAHR2` - Upstream LAPACK routine
886. `ZLALS0` - Upstream LAPACK routine
887. `ZLALSA` - Upstream LAPACK routine
888. `ZLALSD` - Upstream LAPACK routine
889. `ZLAMSWLQ` - Upstream LAPACK routine
890. `ZLAMTSQR` - Upstream LAPACK routine
891. `ZLANHF` - Upstream LAPACK routine
892. `ZLANHT` - Upstream LAPACK routine
893. `ZLAPLL` - Upstream LAPACK routine
894. `ZLAPMR` - Upstream LAPACK routine
895. `ZLAQHB` - Upstream LAPACK routine
896. `ZLAQP2RK` - Upstream LAPACK routine
897. `ZLAQP3RK` - Upstream LAPACK routine
898. `ZLAQR0` - Upstream LAPACK routine
899. `ZLAQR1` - Upstream LAPACK routine
900. `ZLAQR2` - Upstream LAPACK routine
901. `ZLAQR3` - Upstream LAPACK routine
902. `ZLAQR4` - Upstream LAPACK routine
903. `ZLAQR5` - Upstream LAPACK routine
904. `ZLAQZ0` - Upstream LAPACK routine
905. `ZLAQZ1` - Upstream LAPACK routine
906. `ZLAQZ2` - Upstream LAPACK routine
907. `ZLAQZ3` - Upstream LAPACK routine
908. `ZLAR1V` - Upstream LAPACK routine
909. `ZLAR2V` - Upstream LAPACK routine
910. `ZLARCM` - Upstream LAPACK routine
911. `ZLARF1F` - Upstream LAPACK routine
912. `ZLARF1L` - Upstream LAPACK routine
913. `ZLARFB_GETT` - Upstream LAPACK routine
914. `ZLARFGP` - Upstream LAPACK routine
915. `ZLARFT_LVL2` - Upstream LAPACK routine
916. `ZLARFY` - Upstream LAPACK routine
917. `ZLARRV` - Upstream LAPACK routine
918. `ZLARSCL2` - Upstream LAPACK routine
919. `ZLASCL2` - Upstream LAPACK routine
920. `ZLASR` - Upstream LAPACK routine
921. `ZLASWLQ` - Upstream LAPACK routine
922. `ZLASYF` - Upstream LAPACK routine
923. `ZLASYF_AA` - Upstream LAPACK routine
924. `ZLASYF_RK` - Upstream LAPACK routine
925. `ZLASYF_ROOK` - Upstream LAPACK routine
926. `ZLAT2C` - Upstream LAPACK routine
927. `ZLATRS3` - Upstream LAPACK routine
928. `ZLATSQR` - Upstream LAPACK routine
929. `ZLAUNHR_COL_GETRFNP` - Upstream LAPACK routine
930. `ZLAUNHR_COL_GETRFNP2` - Upstream LAPACK routine
931. `ZPBSTF` - Upstream LAPACK routine
932. `ZPBTF2` - Upstream LAPACK routine
933. `ZPFTRF` - Upstream LAPACK routine
934. `ZPFTRI` - Upstream LAPACK routine
935. `ZPFTRS` - Upstream LAPACK routine
936. `ZPOEQUB` - Upstream LAPACK routine
937. `ZPORFSX` - Upstream LAPACK routine
938. `ZPOTF2` - Upstream LAPACK routine
939. `ZPOTRF2` - Upstream LAPACK routine
940. `ZPSTF2` - Upstream LAPACK routine
941. `ZPSTRF` - Upstream LAPACK routine
942. `ZPTTS2` - Upstream LAPACK routine
943. `ZRSCL` - Upstream LAPACK routine
944. `ZSPSVX` - Upstream LAPACK routine
945. `ZSTEMR` - Upstream LAPACK routine
946. `ZSYCON_3` - Upstream LAPACK routine
947. `ZSYCON_ROOK` - Upstream LAPACK routine
948. `ZSYCONV` - Upstream LAPACK routine
949. `ZSYCONVF` - Upstream LAPACK routine
950. `ZSYCONVF_ROOK` - Upstream LAPACK routine
951. `ZSYEQUB` - Upstream LAPACK routine
952. `ZSYRFSX` - Upstream LAPACK routine
953. `ZSYSV_AA` - Upstream LAPACK routine
954. `ZSYSV_AA_2STAGE` - Upstream LAPACK routine
955. `ZSYSV_RK` - Upstream LAPACK routine
956. `ZSYSV_ROOK` - Upstream LAPACK routine
957. `ZSYSWAPR` - Upstream LAPACK routine
958. `ZSYTF2` - Upstream LAPACK routine
959. `ZSYTF2_RK` - Upstream LAPACK routine
960. `ZSYTF2_ROOK` - Upstream LAPACK routine
961. `ZSYTRF_AA` - Upstream LAPACK routine
962. `ZSYTRF_AA_2STAGE` - Upstream LAPACK routine
963. `ZSYTRF_RK` - Upstream LAPACK routine
964. `ZSYTRI2` - Upstream LAPACK routine
965. `ZSYTRI2X` - Upstream LAPACK routine
966. `ZSYTRI_3` - Upstream LAPACK routine
967. `ZSYTRI_3X` - Upstream LAPACK routine
968. `ZSYTRI_ROOK` - Upstream LAPACK routine
969. `ZSYTRS2` - Upstream LAPACK routine
970. `ZSYTRS_3` - Upstream LAPACK routine
971. `ZSYTRS_AA` - Upstream LAPACK routine
972. `ZSYTRS_AA_2STAGE` - Upstream LAPACK routine
973. `ZSYTRS_ROOK` - Upstream LAPACK routine
974. `ZTFSM` - Upstream LAPACK routine
975. `ZTFTRI` - Upstream LAPACK routine
976. `ZTFTTP` - Upstream LAPACK routine
977. `ZTFTTR` - Upstream LAPACK routine
978. `ZTPLQT` - Upstream LAPACK routine
979. `ZTPLQT2` - Upstream LAPACK routine
980. `ZTPMLQT` - Upstream LAPACK routine
981. `ZTPMQRT` - Upstream LAPACK routine
982. `ZTPQRT` - Upstream LAPACK routine
983. `ZTPQRT2` - Upstream LAPACK routine
984. `ZTPRFB` - Upstream LAPACK routine
985. `ZTPTTF` - Upstream LAPACK routine
986. `ZTPTTR` - Upstream LAPACK routine
987. `ZTREVC3` - Upstream LAPACK routine
988. `ZTREXC` - Upstream LAPACK routine
989. `ZTRSEN` - Upstream LAPACK routine
990. `ZTRSNA` - Upstream LAPACK routine
991. `ZTRSYL` - Upstream LAPACK routine
992. `ZTRSYL3` - Upstream LAPACK routine
993. `ZTRTI2` - Upstream LAPACK routine
994. `ZTRTTF` - Upstream LAPACK routine
995. `ZTRTTP` - Upstream LAPACK routine
996. `ZUNBDB` - Upstream LAPACK routine
997. `ZUNBDB1` - Upstream LAPACK routine
998. `ZUNBDB2` - Upstream LAPACK routine
999. `ZUNBDB3` - Upstream LAPACK routine
1000. `ZUNBDB4` - Upstream LAPACK routine
1001. `ZUNBDB5` - Upstream LAPACK routine
1002. `ZUNBDB6` - Upstream LAPACK routine
1003. `ZUNCSD` - Upstream LAPACK routine
1004. `ZUNCSD2BY1` - Upstream LAPACK routine
1005. `ZUNG2R` - Upstream LAPACK routine
1006. `ZUNGTSQR` - Upstream LAPACK routine
1007. `ZUNGTSQR_ROW` - Upstream LAPACK routine
1008. `ZUNHR_COL` - Upstream LAPACK routine
1009. `ZUNM22` - Upstream LAPACK routine
1010. `ZUNM2L` - Upstream LAPACK routine
1011. `ZUNM2R` - Upstream LAPACK routine
1012. `ZUNML2` - Upstream LAPACK routine
1013. `ZUNMR2` - Upstream LAPACK routine
1014. `ZUNMTR` - Upstream LAPACK routine
1015. `ZUPGTR` - Upstream LAPACK routine
1016. `ZUPMTR` - Upstream LAPACK routine
