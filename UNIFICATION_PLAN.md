# API Unification Implementation Plan

## Overview

This document tracks the implementation of the "Unified Implementation + Domain-Specific Aliases" architectural pattern across the FASTER-BLASTER-OPERATIONS-SUPERSET.md specification.

## Summary of Changes

| Category                  | Before (Operations)         | After (Impl + Aliases)                     | Reduction                          |
| ------------------------- | --------------------------- | ------------------------------------------ | ---------------------------------- |
| **GEMM Operations**       | 82 separate implementations | 1 implementation + 81 aliases              | -81 implementations                |
| **Normalization**         | 30 separate implementations | 1 implementation + 29 aliases              | -29 implementations                |
| **Reduction/Aggregation** | 20 separate implementations | 1 implementation + 19 aliases              | -19 implementations                |
| **TOTAL**                 | 3442 operations             | **3313 implementations** + **129 aliases** | **-129 duplicate implementations** |

## 1. GEMM Unification

### New Unified Section

Insert before "BLAS-LIKE EXTENSIONS" section:

**## 6. UNIFIED LINEAR ALGEBRA OPERATIONS**

```c
// Core unified GEMM implementation
fb_status_t fb_gemm_unified(
    // Precision selection
    fb_precision_t precision,        // FP64, FP32, BF16, FP16, FP8_E4M3, FP8_E5M2, INT8
    
    // Batching mode
    fb_batch_mode_t batch_mode,      // SINGLE, ARRAY (array of pointers), STRIDED (fixed stride)
    size_t batch_count,              // Number of matrices (1 for SINGLE)
    
    // Fusion operations
    fb_fusion_t fusion,              // NONE, RELU, GELU, SILU, SWISH, MISH, TANH, SIGMOID
    fb_fusion_t fusion2,             // Secondary fusion (e.g., BIAS, RESIDUAL, LAYERNORM)
    
    // Matrix dimensions
    fb_layout_t layout,              // ROW_MAJOR, COL_MAJOR
    fb_transpose_t trans_a,          // NO_TRANS, TRANS, CONJ_TRANS
    fb_transpose_t trans_b,
    size_t m, size_t n, size_t k,
    
    // Scaling factors (precision-dependent type via void*)
    const void* alpha,
    const void* beta,
    
    // Matrix data (arrays for batched modes)
    const void* a,  size_t lda,  size_t stride_a,
    const void* b,  size_t ldb,  size_t stride_b,
    void* c,        size_t ldc,  size_t stride_c,
    
    // Optional fusion parameters
    const fb_fusion_params_t* fusion_params   // Bias vectors, residual data, normalization params
);

// Precision enum
typedef enum {
    FB_PREC_FP64,      // Double precision (BLAS: D*)
    FB_PREC_FP32,      // Single precision (BLAS: S*)
    FB_PREC_C64,       // Complex double (BLAS: Z*)
    FB_PREC_C32,       // Complex single (BLAS: C*)
    FB_PREC_BF16,      // BFloat16 (ML training)
    FB_PREC_FP16,      // IEEE half precision
    FB_PREC_FP8_E4M3,  // FP8 E4M3 (AI inference)
    FB_PREC_FP8_E5M2,  // FP8 E5M2 (AI inference)
    FB_PREC_INT8,      // 8-bit integer quantized
    FB_PREC_INT4       // 4-bit integer quantized (weight-only)
} fb_precision_t;

// Batch mode enum
typedef enum {
    FB_BATCH_SINGLE,   // Single matrix operation
    FB_BATCH_ARRAY,    // Array of matrix pointers (non-contiguous)
    FB_BATCH_STRIDED   // Fixed-stride batching (contiguous)
} fb_batch_mode_t;

// Fusion enum
typedef enum {
    FB_FUSION_NONE,
    FB_FUSION_RELU,      // max(0, x)
    FB_FUSION_GELU,      // x * Φ(x) (Gaussian Error Linear Unit)
    FB_FUSION_SILU,      // x * σ(x) (Swish/SiLU)
    FB_FUSION_SWISH,     // Same as SILU
    FB_FUSION_MISH,      // x * tanh(softplus(x))
    FB_FUSION_TANH,      // tanh(x)
    FB_FUSION_SIGMOID,   // 1 / (1 + e^(-x))
    FB_FUSION_BIAS,      // Add broadcasted bias vector
    FB_FUSION_RESIDUAL,  // Add residual connection
    FB_FUSION_LAYERNORM  // Apply layer normalization
} fb_fusion_t;
```

**Backend Dispatch Logic**:
- AMD EPYC CPU → AOCL-DLP (low-precision), AOCL-BLIS (standard precision)
- Intel CPU → MKL
- NVIDIA GPU → cuBLAS (standard), cuBLASTLt (fused ops)
- AMD GPU → rocBLAS (standard), hipBLASLt (fused ops)

**Performance Notes**:
- Zero-cost abstraction: Compiler inlines alias wrappers
- Backend selection at runtime based on hardware detection
- Single kernel launch for fused operations (eliminates memory traffic)
- Mixed-precision achieves 3-10× speedup on modern hardware (Tensor Cores, Matrix Engines)

### Alias Locations

#### Section 3: Standard BLAS GEMM (4 aliases)

```markdown
### 3.8 BLAS Level 3: General Matrix Multiply

**SGEMM** - Single-precision matrix multiply: C := α*A*B + β*C

```c
void sgemm(...)  // → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_SINGLE, 1, FB_FUSION_NONE, FB_FUSION_NONE, ...)
```

⚠️ **Performance Note**: This is a zero-cost convenience alias. For batching, mixed-precision, or fused operations, use `fb_gemm_unified()` directly.

✅ **Migration Path**:
- **Drop-in replacement**: No code changes needed - this alias provides backward compatibility
- **Batched operations**: `fb_gemm_unified(FB_PREC_FP32, FB_BATCH_STRIDED, batch_count, ...)`
- **Fused activation**: `fb_gemm_unified(FB_PREC_FP32, FB_BATCH_SINGLE, 1, FB_FUSION_RELU, ...)`
- **Mixed precision**: `fb_gemm_unified(FB_PREC_BF16, ...)`

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations
```

*Repeat for DGEMM, CGEMM, ZGEMM*

#### Section 6 (renamed): BLAS-Like Extensions - Batched Operations (Aliases)

```markdown
## BLAS-LIKE EXTENSIONS (Aliases to Unified Implementations)

⚠️ **Note**: Most operations in this section are convenience aliases to unified implementations. For maximum performance and flexibility, use the unified APIs directly.

### Batched Operations

**Batched GEMM** - Array mode (4 aliases × 4 precisions = 16 aliases):

```c
void cblas_sgemm_batch(...)  // → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_ARRAY, ...)
void cblas_dgemm_batch(...)  // → fb_gemm_unified(FB_PREC_FP64, FB_BATCH_ARRAY, ...)
// ... C/Z variants
```

**Batched GEMM** - Strided mode (4 aliases × 4 precisions = 16 aliases):

```c
void cblas_sgemm_batch_strided(...)  // → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_STRIDED, ...)
void cblas_dgemm_batch_strided(...)  // → fb_gemm_unified(FB_PREC_FP64, FB_BATCH_STRIDED, ...)
// ... C/Z variants
```

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

### Mixed-Precision Operations (Aliases)

**BFloat16 GEMM** (2 aliases):
```c
void cblas_gemm_bf16bf16f32(...)  // → fb_gemm_unified(FB_PREC_BF16, ..., alpha_f32, beta_f32, ..., c_f32)
```

**FP16 GEMM** (2 aliases):
```c
void cblas_gemm_f16f16f32(...)  // → fb_gemm_unified(FB_PREC_FP16, ...)
```

**FP8 GEMM** (4 aliases):
```c
void cblas_gemm_e4m3e4m3f32(...)  // → fb_gemm_unified(FB_PREC_FP8_E4M3, ...)
void cblas_gemm_e5m2e5m2f32(...)  // → fb_gemm_unified(FB_PREC_FP8_E5M2, ...)
```

**INT8 Quantized GEMM** (2 aliases):
```c
void cblas_gemm_s8s8s32(...)  // → fb_gemm_unified(FB_PREC_INT8, ..., quantization_params)
```

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

**Total Aliases in this Section**: 48 (batched) + 12 (mixed-precision) = **60 aliases**
```

#### Section 7: Tensor Fusion Operations - Fused GEMM (Aliases)

```markdown
## 7. TENSOR FUSION OPERATIONS (Aliases to Unified Implementations)

### Fused GEMM Operations (28 aliases)

⚠️ **All operations in this section are aliases to `fb_gemm_unified()` with fusion flags.**

**GEMM + ReLU** (4 aliases):
```c
void fb_gemm_fused_relu(...)  // → fb_gemm_unified(..., FB_FUSION_RELU, FB_FUSION_NONE, ...)
```

**GEMM + GELU** (4 aliases):
```c
void fb_gemm_fused_gelu(...)  // → fb_gemm_unified(..., FB_FUSION_GELU, FB_FUSION_NONE, ...)
```

**GEMM + Bias** (4 aliases):
```c
void fb_gemm_fused_bias_add(...)  // → fb_gemm_unified(..., FB_FUSION_NONE, FB_FUSION_BIAS, ..., fusion_params)
```

**GEMM + Bias + Activation** (8 aliases - 2 activations × 4 precisions):
```c
void fb_gemm_fused_bias_relu(...)  // → fb_gemm_unified(..., FB_FUSION_RELU, FB_FUSION_BIAS, ...)
void fb_gemm_fused_bias_gelu(...)  // → fb_gemm_unified(..., FB_FUSION_GELU, FB_FUSION_BIAS, ...)
```

**GEMM + Scale + Activation** (8 aliases):
```c
void fb_gemm_fused_scale_act(...)  // → fb_gemm_unified(..., FB_FUSION_<ACT>, FB_FUSION_NONE, ..., scale_params)
```

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

**Total Aliases**: **28 fused GEMM aliases**
```

#### Section 16: Low-Precision GEMM (AOCL-DLP) - Aliases

```markdown
## 16. LOW-PRECISION GEMM & QUANTIZATION (Aliases to Unified Implementations)

⚠️ **AMD EPYC-Optimized Dispatch**: When `fb_gemm_unified()` detects AMD EPYC CPUs, it automatically dispatches to AOCL-DLP implementations for maximum performance.

### Low-Precision GEMM (12 aliases)

**LPGEMM Variants**:
```c
void fb_lpgemm_fp32(...)   // → fb_gemm_unified(FB_PREC_FP32, ...)
void fb_lpgemm_bf16(...)   // → fb_gemm_unified(FB_PREC_BF16, ...)
void fb_lpgemm_int8(...)   // → fb_gemm_unified(FB_PREC_INT8, ...)
```

**Batch GEMM** (3 aliases):
```c
void fb_batch_gemm_fp32(...)  // → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_ARRAY, ...)
void fb_batch_gemm_bf16(...)  // → fb_gemm_unified(FB_PREC_BF16, FB_BATCH_ARRAY, ...)
void fb_batch_gemm_int8(...)  // → fb_gemm_unified(FB_PREC_INT8, FB_BATCH_ARRAY, ...)
```

**Strided Batch GEMM** (3 aliases):
```c
void fb_batch_gemm_strided_fp32(...)  // → fb_gemm_unified(FB_PREC_FP32, FB_BATCH_STRIDED, ...)
void fb_batch_gemm_strided_bf16(...)  // → fb_gemm_unified(FB_PREC_BF16, FB_BATCH_STRIDED, ...)
void fb_batch_gemm_strided_int8(...)  // → fb_gemm_unified(FB_PREC_INT8, FB_BATCH_STRIDED, ...)
```

**Quantized GEMM with Scaling** (1 alias):
```c
void fb_qgemm_int8_asymmetric(...)  // → fb_gemm_unified(FB_PREC_INT8, ..., quantization_params)
```

### Fused GEMM Operations (18 aliases)

**GEMM + Bias**:
```c
void fb_gemm_bias_fp32(...)  // → fb_gemm_unified(FB_PREC_FP32, ..., FB_FUSION_BIAS, ...)
// ... bf16, int8 variants
```

**GEMM + Bias + Activation**:
```c
void fb_gemm_bias_relu_fp32(...)    // → fb_gemm_unified(..., FB_FUSION_RELU, FB_FUSION_BIAS, ...)
void fb_gemm_bias_gelu_fp32(...)    // → fb_gemm_unified(..., FB_FUSION_GELU, FB_FUSION_BIAS, ...)
void fb_gemm_bias_sigmoid_fp32(...) // → fb_gemm_unified(..., FB_FUSION_SIGMOID, FB_FUSION_BIAS, ...)
void fb_gemm_bias_tanh_fp32(...)    // → fb_gemm_unified(..., FB_FUSION_TANH, FB_FUSION_BIAS, ...)
// ... bf16 variants (4 more)
```

**Residual Connection Fusion** (2 aliases):
```c
void fb_gemm_bias_residual_relu_fp32(...)  // → fb_gemm_unified(..., FB_FUSION_RELU, FB_FUSION_RESIDUAL, ...)
// ... bf16 variant
```

**Layer Normalization Fusion** (2 aliases):
```c
void fb_gemm_bias_layernorm_fp32(...)  // → fb_gemm_unified(..., FB_FUSION_LAYERNORM, FB_FUSION_BIAS, ...)
// ... bf16 variant
```

→ **Unified Implementation**: See Section 6 - Unified Linear Algebra Operations

**Total Aliases**: 12 (low-precision) + 18 (fused) = **30 aliases**
```

### GEMM Unification Summary

- **Before**: 82 separate GEMM implementations across 4 sections
- **After**: 1 unified implementation + 81 aliases
- **Reduction**: **-81 duplicate implementations**

---

## 2. Normalization Unification

*(To be documented - similar structure to GEMM)*

---

## 3. Reduction Unification

*(To be documented - similar structure to GEMM)*

---

## Implementation Status

- [x] Architectural pattern documented in copilot-instructions.md
- [ ] GEMM unification (Section 6 unified + 4 alias sections)
- [ ] Normalization unification
- [ ] Reduction unification
- [ ] Operation count table updates
- [ ] API Design Principles appendix added to spec

---

## Validation Checklist

For each unified operation:

1. ✅ Unified implementation covers all variants
2. ✅ Enums provide clear variant selection
3. ✅ Aliases match original signatures exactly
4. ✅ Performance notes document zero-cost abstraction
5. ✅ Migration paths clear and actionable
6. ✅ Backend dispatch documented
7. ✅ Operation counts updated (implementations vs aliases)
8. ✅ Tests verify alias equivalence

---

## Final Operation Count

| Category                  | Implementations | Aliases | Total API Surface |
| ------------------------- | --------------- | ------- | ----------------- |
| BLAS Level 1-3            | 174             | 0       | 174               |
| LAPACK                    | 1488            | 0       | 1488              |
| **Unified GEMM**          | **1**           | **81**  | **82**            |
| **Unified Normalization** | **1**           | **29**  | **30**            |
| **Unified Reduction**     | **1**           | **19**  | **20**            |
| Other Operations          | 1648            | 0       | 1648              |
| **TOTAL**                 | **3313**        | **129** | **3442**          |

**Grand Total**: 3313 unique implementations + 129 convenience aliases = 3442 total API surface

---

**Note**: This document serves as implementation blueprint. Once changes are applied to FASTER-BLASTER-OPERATIONS-SUPERSET.md, this file can be archived or removed.
