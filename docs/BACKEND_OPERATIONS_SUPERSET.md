# Backend Operations Superset Analysis

**Generated**: December 12, 2025  
**Purpose**: Catalog ALL operations available across all backends to define complete coverage target

## Critical Scope Definition: Where Do We Draw The Line?

### The Scope Question

As we catalog BLAS/LAPACK libraries, a critical architectural question emerges: **Where do we stop?** The landscape includes:

- **Direct BLAS/LAPACK implementations**: cuBLAS, OpenBLAS, MKL (clear inclusions)
- **Math library ecosystems**: MKL includes VML/FFT/RNG, oneMath includes Sparse/Statistics
- **Higher-level wrappers**: Armadillo, Eigen (use BLAS/LAPACK backends)
- **Domain-specific libraries**: GLM (OpenGL math), Boost.Math (general math)
- **Specification standards**: oneMath (UXL Foundation standard), BLAS Technical Forum

### faster-blaster's Core Identity: BLAS/LAPACK Abstraction Layer

**PRIMARY SCOPE** (What we ARE):

- Standard BLAS (Level 1/2/3) operations
- Standard LAPACK dense linear algebra operations  
- Common vendor extensions that maintain BLAS/LAPACK semantics:
  - Batched operations (gemmBatched, getrfBatched)
  - Strided batched (gemmStridedBatched)
  - Mixed precision (gemmEx with FP16/BF16/TF32)
  - Sparse BLAS (when available from BLAS library)
- Backend-specific optimizations (tile algorithms, GPU kernels) **transparently**

**SECONDARY SCOPE** (Include if multiple backends provide it):

- Sparse Linear Algebra (rocSPARSE, cuSPARSE, MKL Sparse BLAS, oneMath Sparse)
- FFT operations (when part of math library: MKL FFT, NVPL FFT, oneMath FFT)
- RNG operations (MKL RNG, NVPL RAND, oneMath RNG) - useful for testing/initialization
- Vector Math (MKL VML, oneMath VM) - element-wise transcendental functions
- ScaLAPACK (distributed parallel LAPACK) - for multi-node systems

**OUT OF SCOPE** (What we are NOT):

- High-level C++ template libraries that wrap BLAS (Eigen, Armadillo) - these are **users** of faster-blaster, not backends
- Domain-specific math (GLM for graphics transformations, Boost.Math for special functions)
- Different API paradigms (GSL uses different conventions than BLAS/LAPACK)
- Machine learning libraries (oneDNN, cuDNN) - different problem domain
- Generic STL extensions (oneDPL) - not numerical linear algebra

### The Key Distinction: Backend vs. Consumer

| Category                   | Relationship to faster-blaster            | Examples                                               |
| -------------------------- | ----------------------------------------- | ------------------------------------------------------ |
| **BLAS/LAPACK Backends**   | Implementations we abstract over          | cuBLAS, OpenBLAS, MKL, rocBLAS, oneMath                |
| **Wrappers/Dispatchers**   | Tools that serve similar role             | FlexiBLAS (runtime switching), oneMath (standard spec) |
| **Higher-Level Libraries** | **Consumers that can USE faster-blaster** | Eigen, Armadillo, Armadillo++                          |
| **Domain-Specific Math**   | Different problem domain                  | GLM, Boost.Math, GSL                                   |

**oneMath Special Case**: oneMath is both a **specification standard** (like BLAS/LAPACK themselves) AND an **implementation** (oneMKL). We support oneMKL as a backend implementing the oneMath spec.

**Armadillo Special Case**: Armadillo is a C++ wrapper that **uses** BLAS/LAPACK backends (OpenBLAS, MKL, etc.). It's a **consumer** of faster-blaster, not a backend. Users who want Armadillo's C++ API can use Armadillo with faster-blaster as the backend.

**clBLAS Status Correction**: Research shows clBLAS is **still maintained** (v2.12, 2018) but development is slow. It provides standard BLAS Level 1/2/3 via OpenCL. Include as optional OpenCL GPU backend.

### Recommended Scope Boundaries

**Phase 1: Core BLAS/LAPACK (Essential)** - 341 operations:

- Standard BLAS Level 1/2/3 (152 ops)
- Standard LAPACK dense operations (28 current + 88 extended = 116 ops)
- GPU batched/strided extensions (44 ops)
- Fused multi-operation functions (29 ops)

**Phase 2: Extended Math Library Features (High Value)** - Select ~100 operations:

- Sparse BLAS/LAPACK (when backend provides it): ~40 ops
- FFT (1D/2D/3D, real/complex): ~20 ops
- RNG (uniform, normal, basic distributions): ~15 ops
- Vector Math (sin, cos, exp, log, etc.): ~25 ops

**Phase 3: Specialized/Optional (Nice to Have)** - Select ~50 operations:

- ScaLAPACK (distributed computing): ~30 ops
- Advanced sparse solvers (PARDISO-like): ~10 ops
- Summary statistics (oneMath): ~10 ops

**Total Target: ~470 operations** (341 core + ~130 extended features)

This keeps faster-blaster focused on its core mission: **providing zero-overhead compile-time abstraction over BLAS/LAPACK backends**, while supporting common extensions and high-value fused operations that maintain the same paradigm.

### Fused and Multi-Operation Functions

Backends provide **fused operations** that combine multiple steps for better performance. These are HIGH VALUE for consumers:

**Already Included - LAPACK Combined Solvers** (solves linear systems in one call):
- `?gesv` = LU factorization + forward/backward solve (fuses `getrf` + `getrs`)
- `?posv` = Cholesky factorization + triangular solve (fuses `potrf` + `potrs`)  
- `?gels` = QR/LQ factorization + least squares solve (fuses `geqrf`/`gelqf` + `ormqr`/`ormlq` + `trsm`)

**Already Included - Batched Operations** (fuse many operations into one kernel):
- `gemmBatched`, `trsmBatched`, `getrfBatched`, `getriBatched` - array-of-pointers batching
- `gemmStridedBatched` - strided batching (faster, assumes regular stride pattern)

**Already Included - Mixed Precision** (fuse type conversion + compute):
- `gemmEx` - Mixed precision GEMM (FP16 input → FP32/TF32 compute → FP16/FP32 output)
- `gemmBatchedEx` - Batched + mixed precision combined

**Should Add - Vendor Extensions** (~25 operations):

**MKL-Specific Fused Operations**:
- `gemm_bf16bf16f32` / `gemm_f16f16f32` - BF16/FP16 → FP32 fused conversion
- `?gemm3m` - Complex GEMM using only 3 real multiplies instead of 4 (25% faster)
- `?gemmt` - Triangular GEMM (fuses `gemm` + extract triangle)
- `mkl_?omatcopy` - Out-of-place matrix transpose/copy with scaling (fuses copy + scale + transpose)
- `mkl_?imatcopy` - In-place matrix transpose with scaling
- `cblas_?gemm_batch` - Batch with variable sizes (group GEMM)
- `lapack::geinv_batch` - Batched matrix inverse (fuses `getrf` + `getri`)
- `lapack::getrfnp` - LU factorization WITHOUT pivoting (faster when pivoting not needed)

**rocBLAS/AMD Extensions**:
- `rocblas_?geam` - Matrix addition with transpose: `C = α*op(A) + β*op(B)` (fuses transpose + axpy)
- `rocblas_gemm_ext2` - Extended GEMM with additional options
- `rocblas_axpy_ex` - Mixed precision AXPY with batched variants
- `rocblas_dot_ex` / `nrm2_ex` / `scal_ex` - Mixed precision Level 1 operations  
- `rocblas_?syrkx` / `?herkx` - Symmetric/Hermitian rank-k update with separate C matrix
- `rocblas_trsm_ex` - Mixed precision triangular solve

**cuBLAS/NVIDIA Extensions**:
- `cublas<t>gemm3m` - Complex GEMM optimized (NVIDIA implementation)
- `cublasGemmGroupedBatched` - Variable-size batched GEMM (different m/n/k per batch)
- `cublasTrsmBatched` with out-of-place option
- `cublasGemmStridedBatchedEx` - Strided batched + mixed precision
- `cublas<t>tpttr` / `trttp` - Packed ↔ full format conversion (fused copy + reformat)

**MAGMA-Specific**:
- Tile algorithms that internally fuse multiple BLAS/LAPACK calls
- Mixed precision iterative refinement solvers (e.g., `magma_dsgesv` - FP32 compute, FP64 refinement)

**Value Proposition**: Fused operations can be **2-10x faster** than calling separate functions due to:
- Reduced memory transfers (intermediate results stay in cache/registers)
- Kernel launch overhead amortization  
- Better instruction fusion and pipelining

**Recommendation**: Add ~25 high-value fused operations to Phase 1, prioritizing those available across multiple backends.

---

## Executive Summary

This document identifies the complete superset of BLAS and LAPACK operations available across all backends (GPU and CPU) to establish our comprehensive implementation target.

### Complete Backend Inventory (All Potential Targets)

**GPU Backends:**
1. **cuBLAS + cuSOLVER** (NVIDIA CUDA) - NVIDIA GPU standard
2. **rocBLAS + rocSOLVER** (AMD ROCm) - AMD GPU standard  
3. **hipBLAS + hipSOLVER** (Unified CUDA/ROCm) - Portable GPU abstraction
4. **oneMKL** (Intel GPU via SYCL) - Intel Xe GPU architecture
5. **MAGMA** (Multi-platform GPU) - Dense linear algebra for heterogeneous systems
6. **clBLAS + clSPARSE** (OpenCL) - ⚠️ Deprecated but still in use

**CPU Backends - Vendor Optimized:**
7. **Intel MKL** (x86/x64) - Industry standard, comprehensive BLAS/LAPACK + extensions
8. **NVPL** (NVIDIA Performance Libraries) - ARM64/Grace CPU optimized
9. **Arm Performance Libraries (ArmPL)** - ARM architecture optimized
10. **AOCL** (AMD Optimizing CPU Libraries) - AMD Zen optimized
11. **IBM ESSL** (Engineering & Scientific Subroutine Library) - POWER architecture
12. **Apple Accelerate** (macOS/iOS) - Apple Silicon + Intel optimized

**CPU Backends - Open Source:**
13. **OpenBLAS** - Open source, highly optimized, widely used
14. **BLIS** (BLAS-like Library Instantiation Software) - Modern BLAS framework
15. **ATLAS** (Automatically Tuned Linear Algebra Software) - Auto-tuning framework
16. **FlexiBLAS** - Runtime switchable BLAS/LAPACK wrapper (can use any backend)
17. **Netlib Reference BLAS/LAPACK** - Reference implementation (500+ operations)

**Specialized Libraries:**
18. **PLASMA** (Parallel Linear Algebra Software for Multicore) - Multi-core CPU
19. **BLAS++** / **LAPACK++** - Modern C++ API wrappers
20. **Eigen** - Header-only C++ library (can use any BLAS/LAPACK backend)
21. **GNU Scientific Library (GSL)** - ⚠️ Different API, basic linear algebra only

**Total**: 21 potential backends across GPU, CPU vendor, CPU open-source, and specialized categories

### Backend-Specific Features & Operation Extensions

| Backend       | Unique Features                                              | Additional Operations Beyond Standard LAPACK |
| ------------- | ------------------------------------------------------------ | -------------------------------------------- |
| **Intel MKL** | VML (Vector Math), Sparse BLAS, PARDISO, FFT, RNG, ScaLAPACK | ~225 extended ops (VML, sparse, FFT, RNG)    |
| **NVPL**      | Grace CPU optimized, TENSOR ops, ScaLAPACK                   | Full LAPACK + ScaLAPACK + TENSOR             |
| **ArmPL**     | ARM SVE vectorization, Neon optimization                     | Full LAPACK + Sparse + FFT                   |
| **IBM ESSL**  | POWER optimization, parallel variants                        | Full LAPACK + specialized POWER ops          |
| **MAGMA**     | Hybrid CPU-GPU, batched ops, FP16 support                    | Batched LAPACK, hybrid solvers               |
| **PLASMA**    | Tile algorithms, multi-core parallelism                      | Tile-based LAPACK variants                   |
| **FlexiBLAS** | **Runtime backend switching**, universal wrapper             | Delegates to any BLAS/LAPACK backend         |
| **Eigen**     | Expression templates, header-only, can use BLAS backends     | Template-based, not direct BLAS/LAPACK       |
| **GSL**       | ⚠️ **Different API** (not BLAS/LAPACK compatible)             | Basic linear algebra with unique API         |
| **clBLAS**    | OpenCL GPU support                                           | ⚠️ Deprecated, limited to BLAS Level 1/2/3    |

**Key Insight**: FlexiBLAS is special - it's a **wrapper that allows runtime switching** between backends (OpenBLAS, MKL, BLIS, etc.) without recompilation. This is exactly what faster-blaster aims to do but with compile-time backend selection.

---

## LAPACK Operations Matrix

### Currently Implemented (28 operations)

| Operation | Description            | Types   | Status        |
| --------- | ---------------------- | ------- | ------------- |
| getrf     | LU factorization       | s,d,c,z | ✅ Implemented |
| getrs     | LU solve               | s,d,c,z | ✅ Implemented |
| potrf     | Cholesky factorization | s,d,c,z | ✅ Implemented |
| potrs     | Cholesky solve         | s,d,c,z | ✅ Implemented |
| geqrf     | QR factorization       | s,d,c,z | ✅ Implemented |
| gesvd     | SVD (standard)         | s,d,c,z | ✅ Implemented |
| syev/heev | Eigenvalues (standard) | s,d,c,z | ✅ Implemented |

### Extended LAPACK - Combined Solvers (12 operations)
| Operation | Description                        | cuSOLVER | rocSOLVER | hipSOLVER | oneMKL | MAGMA | MKL | OpenBLAS |
| --------- | ---------------------------------- | -------- | --------- | --------- | ------ | ----- | --- | -------- |
| sgesv     | Combined LU solve (float)          | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dgesv     | Combined LU solve (double)         | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cgesv     | Combined LU solve (complex)        | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zgesv     | Combined LU solve (dcomplex)       | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| sposv     | Combined Cholesky solve (float)    | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dposv     | Combined Cholesky solve (double)   | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cposv     | Combined Cholesky solve (complex)  | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zposv     | Combined Cholesky solve (dcomplex) | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| sgels     | Least squares solve (float)        | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dgels     | Least squares solve (double)       | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cgels     | Least squares solve (complex)      | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zgels     | Least squares solve (dcomplex)     | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |

### Extended LAPACK - General Eigenvalues (4 operations)
| Operation | Description                          | cuSOLVER | rocSOLVER | hipSOLVER | oneMKL | MAGMA | MKL | OpenBLAS |
| --------- | ------------------------------------ | -------- | --------- | --------- | ------ | ----- | --- | -------- |
| sgeev     | Non-symmetric eigenvalues (float)    | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dgeev     | Non-symmetric eigenvalues (double)   | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cgeev     | Non-symmetric eigenvalues (complex)  | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zgeev     | Non-symmetric eigenvalues (dcomplex) | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |

### Extended LAPACK - Eigenvalue Variants (16 operations)
| Operation | Description                              | cuSOLVER | rocSOLVER | hipSOLVER | oneMKL | MAGMA | MKL | OpenBLAS |
| --------- | ---------------------------------------- | -------- | --------- | --------- | ------ | ----- | --- | -------- |
| ssyevd    | Symmetric eigenvalues (D&C, float)       | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dsyevd    | Symmetric eigenvalues (D&C, double)      | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cheevd    | Hermitian eigenvalues (D&C, complex)     | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zheevd    | Hermitian eigenvalues (D&C, dcomplex)    | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| ssyevj    | Symmetric eigenvalues (Jacobi, float)    | ✅        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ⚠️        |
| dsyevj    | Symmetric eigenvalues (Jacobi, double)   | ✅        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ⚠️        |
| cheevj    | Hermitian eigenvalues (Jacobi, complex)  | ✅        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ⚠️        |
| zheevj    | Hermitian eigenvalues (Jacobi, dcomplex) | ✅        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ⚠️        |
| ssygvd    | Generalized sym eigenvalues (float)      | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dsygvd    | Generalized sym eigenvalues (double)     | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| chegvd    | Generalized herm eigenvalues (complex)   | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zhegvd    | Generalized herm eigenvalues (dcomplex)  | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |

### Extended LAPACK - SVD Variants (12 operations)
| Operation | Description                            | cuSOLVER | rocSOLVER | hipSOLVER | oneMKL | MAGMA | MKL | OpenBLAS |
| --------- | -------------------------------------- | -------- | --------- | --------- | ------ | ----- | --- | -------- |
| sgesvdj   | SVD Jacobi (better accuracy, float)    | ✅        | ✅         | ✅         | ⚠️      | ⚠️     | ⚠️   | ❌        |
| dgesvdj   | SVD Jacobi (better accuracy, double)   | ✅        | ✅         | ✅         | ⚠️      | ⚠️     | ⚠️   | ❌        |
| cgesvdj   | SVD Jacobi (better accuracy, complex)  | ✅        | ✅         | ✅         | ⚠️      | ⚠️     | ⚠️   | ❌        |
| zgesvdj   | SVD Jacobi (better accuracy, dcomplex) | ✅        | ✅         | ✅         | ⚠️      | ⚠️     | ⚠️   | ❌        |
| sgesvda   | SVD approximate (faster, float)        | ✅        | ⚠️         | ✅         | ❌      | ⚠️     | ❌   | ❌        |
| dgesvda   | SVD approximate (faster, double)       | ✅        | ⚠️         | ✅         | ❌      | ⚠️     | ❌   | ❌        |
| cgesvda   | SVD approximate (faster, complex)      | ✅        | ⚠️         | ✅         | ❌      | ⚠️     | ❌   | ❌        |
| zgesvda   | SVD approximate (faster, dcomplex)     | ✅        | ⚠️         | ✅         | ❌      | ⚠️     | ❌   | ❌        |
| sgesdd    | SVD D&C (fastest, float)               | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dgesdd    | SVD D&C (fastest, double)              | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cgesdd    | SVD D&C (fastest, complex)             | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zgesdd    | SVD D&C (fastest, dcomplex)            | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |

### Extended LAPACK - QR Utilities (16 operations)
| Operation | Description                       | cuSOLVER | rocSOLVER | hipSOLVER | oneMKL | MAGMA | MKL | OpenBLAS |
| --------- | --------------------------------- | -------- | --------- | --------- | ------ | ----- | --- | -------- |
| sorgqr    | Generate orthogonal Q (float)     | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dorgqr    | Generate orthogonal Q (double)    | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cungqr    | Generate unitary Q (complex)      | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zungqr    | Generate unitary Q (dcomplex)     | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| sormqr    | Multiply by orthogonal Q (float)  | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dormqr    | Multiply by orthogonal Q (double) | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cunmqr    | Multiply by unitary Q (complex)   | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zunmqr    | Multiply by unitary Q (dcomplex)  | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| sgeqp3    | QR with pivoting (float)          | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dgeqp3    | QR with pivoting (double)         | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cgeqp3    | QR with pivoting (complex)        | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zgeqp3    | QR with pivoting (dcomplex)       | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |

### Extended LAPACK - Other Factorizations (12 operations)
| Operation | Description                 | cuSOLVER | rocSOLVER | hipSOLVER | oneMKL | MAGMA | MKL | OpenBLAS |
| --------- | --------------------------- | -------- | --------- | --------- | ------ | ----- | --- | -------- |
| sgelqf    | LQ factorization (float)    | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dgelqf    | LQ factorization (double)   | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cgelqf    | LQ factorization (complex)  | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zgelqf    | LQ factorization (dcomplex) | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| sgerqf    | RQ factorization (float)    | ⚠️        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ✅        |
| dgerqf    | RQ factorization (double)   | ⚠️        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ✅        |
| cgerqf    | RQ factorization (complex)  | ⚠️        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ✅        |
| zgerqf    | RQ factorization (dcomplex) | ⚠️        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ✅        |
| sgeqlf    | QL factorization (float)    | ⚠️        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ✅        |
| dgeqlf    | QL factorization (double)   | ⚠️        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ✅        |
| cgeqlf    | QL factorization (complex)  | ⚠️        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ✅        |
| zgeqlf    | QL factorization (dcomplex) | ⚠️        | ✅         | ✅         | ✅      | ⚠️     | ✅   | ✅        |

### Extended LAPACK - Matrix Operations (16 operations)
| Operation | Description                                 | cuSOLVER | rocSOLVER | hipSOLVER | oneMKL | MAGMA | MKL | OpenBLAS |
| --------- | ------------------------------------------- | -------- | --------- | --------- | ------ | ----- | --- | -------- |
| strtri    | Triangular inverse (float)                  | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dtrtri    | Triangular inverse (double)                 | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| ctrtri    | Triangular inverse (complex)                | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| ztrtri    | Triangular inverse (dcomplex)               | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| sgetri    | General inverse (float)                     | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dgetri    | General inverse (double)                    | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cgetri    | General inverse (complex)                   | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zgetri    | General inverse (dcomplex)                  | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| sgecon    | Condition number estimate (float)           | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dgecon    | Condition number estimate (double)          | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cgecon    | Condition number estimate (complex)         | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zgecon    | Condition number estimate (dcomplex)        | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| spocon    | Sym positive definite condition (float)     | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| dpocon    | Sym positive definite condition (double)    | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| cpocon    | Herm positive definite condition (complex)  | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |
| zpocon    | Herm positive definite condition (dcomplex) | ✅        | ✅         | ✅         | ✅      | ✅     | ✅   | ✅        |

**LAPACK Total**: 28 (current) + 88 (extended) = **116 operations**

---

## Extended BLAS Operations Matrix

### Standard BLAS (Currently Implemented: ~152 operations)
- **Level 1**: axpy, scal, copy, swap, dot, nrm2, asum, iamax, rotg, rot, rotm, rotmg (~48-54 functions)
- **Level 2**: gemv, gbmv, hemv, symv, trmv, trsv, ger, her, syr, etc. (~60-70 functions)
- **Level 3**: gemm, symm, hemm, trmm, trsm, syrk, herk, syr2k, her2k (~36 functions)

### Batched Operations
| Operation     | Description                                | cuBLAS | rocBLAS | hipBLAS | oneMKL | MAGMA |
| ------------- | ------------------------------------------ | ------ | ------- | ------- | ------ | ----- |
| sgemmBatched  | Batched GEMM (float, array of pointers)    | ✅      | ✅       | ✅       | ✅      | ✅     |
| dgemmBatched  | Batched GEMM (double, array of pointers)   | ✅      | ✅       | ✅       | ✅      | ✅     |
| cgemmBatched  | Batched GEMM (complex, array of pointers)  | ✅      | ✅       | ✅       | ✅      | ✅     |
| zgemmBatched  | Batched GEMM (dcomplex, array of pointers) | ✅      | ✅       | ✅       | ✅      | ✅     |
| strsmBatched  | Batched TRSM (float)                       | ✅      | ✅       | ✅       | ✅      | ✅     |
| dtrsmBatched  | Batched TRSM (double)                      | ✅      | ✅       | ✅       | ✅      | ✅     |
| ctrsmBatched  | Batched TRSM (complex)                     | ✅      | ✅       | ✅       | ✅      | ✅     |
| ztrsmBatched  | Batched TRSM (dcomplex)                    | ✅      | ✅       | ✅       | ✅      | ✅     |
| sgetrfBatched | Batched LU factorization (float)           | ✅      | ✅       | ✅       | ✅      | ✅     |
| dgetrfBatched | Batched LU factorization (double)          | ✅      | ✅       | ✅       | ✅      | ✅     |
| cgetrfBatched | Batched LU factorization (complex)         | ✅      | ✅       | ✅       | ✅      | ✅     |
| zgetrfBatched | Batched LU factorization (dcomplex)        | ✅      | ✅       | ✅       | ✅      | ✅     |
| sgetriBatched | Batched matrix inverse (float)             | ✅      | ✅       | ✅       | ✅      | ✅     |
| dgetriBatched | Batched matrix inverse (double)            | ✅      | ✅       | ✅       | ✅      | ✅     |
| cgetriBatched | Batched matrix inverse (complex)           | ✅      | ✅       | ✅       | ✅      | ✅     |
| zgetriBatched | Batched matrix inverse (dcomplex)          | ✅      | ✅       | ✅       | ✅      | ✅     |

### Strided Batched Operations
| Operation           | Description                     | cuBLAS | rocBLAS | hipBLAS | oneMKL | MAGMA |
| ------------------- | ------------------------------- | ------ | ------- | ------- | ------ | ----- |
| sgemmStridedBatched | Strided batched GEMM (float)    | ✅      | ✅       | ✅       | ✅      | ✅     |
| dgemmStridedBatched | Strided batched GEMM (double)   | ✅      | ✅       | ✅       | ✅      | ✅     |
| cgemmStridedBatched | Strided batched GEMM (complex)  | ✅      | ✅       | ✅       | ✅      | ✅     |
| zgemmStridedBatched | Strided batched GEMM (dcomplex) | ✅      | ✅       | ✅       | ✅      | ✅     |

### Mixed Precision and Extended Operations
| Operation   | Description                                       | cuBLAS | rocBLAS | hipBLAS | oneMKL | Notes                        |
| ----------- | ------------------------------------------------- | ------ | ------- | ------- | ------ | ---------------------------- |
| gemmEx      | Mixed precision GEMM (FP16/BF16/TF32/FP32/FP64)   | ✅      | ✅       | ✅       | ✅      | Supports compute types       |
| gemm3m      | Complex GEMM optimized (3 matrix multiplications) | ✅      | ⚠️       | ⚠️       | ⚠️      | NVIDIA-specific optimization |
| gemmGrouped | Grouped/variable-batch GEMM                       | ✅      | ✅       | ✅       | ✅      | Different sizes per batch    |

**Extended BLAS Total**: ~44 new operations

---

## Summary: Target Operation Count

| Category             | Current | Target  | Gap     |
| -------------------- | ------- | ------- | ------- |
| Standard LAPACK      | 28      | 28      | 0       |
| Extended LAPACK      | 0       | 88      | 88      |
| Standard BLAS        | 152     | 152     | 0       |
| Extended BLAS        | 0       | 44      | 44      |
| **Fused Operations** | **2**   | **29**  | **27**  |
| **TOTAL**            | **182** | **341** | **159** |

---

## CPU-Specific LAPACK Operations

CPU backends (Intel MKL, OpenBLAS, BLIS, AOCL, Accelerate, ATLAS) provide the **full LAPACK reference implementation** with **500+ operations**. Key operations beyond what GPU backends provide:

### Additional CPU LAPACK Operations (not typically in GPU backends)
| Category                      | Operations                                               | Count | Availability |
| ----------------------------- | -------------------------------------------------------- | ----- | ------------ |
| **Band matrix operations**    | gbsv, gbtrf, gbtrs, pbsv, pbtrf, pbtrs, sbev, hbev, etc. | ~40   | ✅ All CPU    |
| **Packed storage**            | spsv, sptrf, sptrs, hpsv, hptrf, hptrs, spev, hpev, etc. | ~30   | ✅ All CPU    |
| **Tridiagonal**               | gtsv, gttrf, gttrs, stev, ptsv, pttrf, pttrs, etc.       | ~20   | ✅ All CPU    |
| **Refined iterative solvers** | gesvx, posvx, sysvx, gbsvx, pbsvx, etc.                  | ~15   | ✅ All CPU    |
| **Equilibration**             | geequ, poequ, gbequ, etc.                                | ~10   | ✅ All CPU    |
| **Condition estimates**       | gecon, pocon, trcon, etc.                                | ~8    | ✅ All CPU    |
| **Matrix norms**              | lange, lansy, lanhe, lantr, etc.                         | ~12   | ✅ All CPU    |
| **Advanced eigenvalue**       | gees, gges, geevx, gesvx, etc.                           | ~20   | ✅ All CPU    |
| **Generalized problems**      | ggev, gglse, ggglm, etc.                                 | ~15   | ✅ All CPU    |
| **Utilities**                 | lacpy, laset, lascl, etc.                                | ~100+ | ✅ All CPU    |

**Total CPU-only LAPACK**: ~270 operations

**Note**: GPU backends focus on dense matrix operations for maximum parallelism. CPU backends support full LAPACK specification including band, packed, and sparse formats which don't parallelize well on GPUs.

### Intel MKL Extended Operations (CPU-only)
| Category        | Description                                                              | Count |
| --------------- | ------------------------------------------------------------------------ | ----- |
| **VML**         | Vector Math Library (transcendental functions: sin, cos, exp, log, etc.) | 100+  |
| **Sparse BLAS** | Sparse matrix operations (CSR, CSC, COO formats)                         | 50+   |
| **PARDISO**     | Parallel Direct Sparse Solver                                            | 20+   |
| **FFT**         | Fast Fourier Transform (1D/2D/3D, real/complex)                          | 30+   |
| **RNG**         | Random Number Generators (uniform, normal, etc.)                         | 25+   |

**Total Intel MKL extensions**: ~225 operations

---

## Comprehensive Operation Count Summary

### Primary Target: GPU-Compatible Operations (All Backends)
| Category                        | Current | Target  | Gap     | Backends      |
| ------------------------------- | ------- | ------- | ------- | ------------- |
| Standard LAPACK                 | 28      | 28      | 0       | All           |
| Extended LAPACK (dense)         | 0       | 88      | 88      | GPU + CPU     |
| Standard BLAS                   | 152     | 152     | 0       | All           |
| Extended BLAS (batched/strided) | 0       | 44      | 44      | GPU only      |
| **PRIMARY TOTAL**               | **180** | **312** | **132** | **GPU focus** |

### Extended Target: CPU-Specific Operations
| Category                                     | Operations | Backends         |
| -------------------------------------------- | ---------- | ---------------- |
| Band/Packed/Tridiagonal LAPACK               | ~100       | CPU only         |
| Advanced LAPACK solvers                      | ~50        | CPU only         |
| LAPACK utilities                             | ~120       | CPU only         |
| Intel MKL extensions (VML, Sparse, FFT, RNG) | ~225       | Intel MKL only   |
| **EXTENDED TOTAL**                           | **~495**   | **CPU backends** |

### Grand Total: 312 (Primary GPU-compatible) + 495 (Extended CPU) = **~807 operations**

**Implementation Strategy**:

1. **Phase 1**: Implement 132 new GPU-compatible operations across all GPU backends (Priority)
2. **Phase 2**: Add CPU-specific LAPACK operations to CPU backends (band/packed/tridiagonal)
3. **Phase 3**: Intel MKL extensions (VML, Sparse, FFT) for specialized use cases

### Backend Coverage Priority (21 Total Backends)

**Tier 1 (Core GPU - Full Implementation Priority)**:

1. cuBLAS + cuSOLVER (NVIDIA CUDA)
2. rocBLAS + rocSOLVER (AMD ROCm)
3. hipBLAS + hipSOLVER (Unified portable)
4. oneMKL (Intel GPU)
5. MAGMA (Hybrid CPU-GPU)

**Tier 2 (Vendor CPU Libraries - High Performance)**:

6. Intel MKL (x86/x64) - Most comprehensive
7. NVPL (NVIDIA Grace ARM64)
8. Arm Performance Libraries (ARM architecture)
9. AOCL (AMD Zen CPUs)
10. IBM ESSL (POWER architecture)
11. Apple Accelerate (Apple Silicon + Intel)

**Tier 3 (Open Source CPU - Wide Compatibility)**:

12. OpenBLAS - Default open-source choice
13. BLIS - Modern BLAS framework
14. ATLAS - Auto-tuning legacy
15. Netlib Reference - Conformance testing

**Tier 4 (Wrappers & Specialized)**:

16. **FlexiBLAS** - Runtime backend switcher (can delegate to any BLAS/LAPACK)
17. PLASMA - Multi-core tile algorithms
18. BLAS++ / LAPACK++ - Modern C++ wrappers
19. Eigen - Template library (can use backends)
20. clBLAS (OpenCL) - ⚠️ Deprecated
21. GSL - ⚠️ Different API (not BLAS/LAPACK compatible)

**Special Note on FlexiBLAS**: This backend is particularly important as it provides **runtime switching** between backends. faster-blaster provides **compile-time selection**, while FlexiBLAS allows changing backends at runtime. Supporting FlexiBLAS as a backend means users get both compile-time AND runtime flexibility.

### Recommended Implementation Order

1. **Immediate** (Phase 1): GPU Tier 1 (cuBLAS, rocBLAS, hipBLAS, oneMKL, MAGMA) - 132 new ops
2. **High Priority** (Phase 2): CPU Tier 2 vendors (MKL, NVPL, ArmPL, AOCL) - Standard LAPACK
3. **Standard Priority** (Phase 3): CPU Tier 3 open-source (OpenBLAS, BLIS, ATLAS) - Standard LAPACK
4. **Future** (Phase 4): Specialized (PLASMA, FlexiBLAS, Eigen integration)

### Legend

- ✅ = Fully supported
- ⚠️ = Partial support or backend-specific
- ❌ = Not available

---

## Next Steps

1. ✅ **Trait interface expansion** - Add 132 new function pointers to `gpu_backend_trait.h`
2. ⬜ **cuBLAS/cuSOLVER implementation** - Implement all 132 operations
3. ⬜ **rocBLAS/rocSOLVER implementation** - Implement all 132 operations
4. ⬜ **hipBLAS/hipSOLVER implementation** - Implement all 132 operations
5. ⬜ **oneMKL implementation** - Implement all 132 operations
6. ⬜ **MAGMA implementation** - Implement subset (prioritize batched ops)
7. ⬜ **Comprehensive testing** - Test all operations across all backends
8. ⬜ **Documentation** - API documentation and coverage matrix
