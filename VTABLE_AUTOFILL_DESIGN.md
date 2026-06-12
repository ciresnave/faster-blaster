# Vtable Auto-Fill System Design

## Overview

Implement runtime vtable completion that ensures **correctness everywhere, performance where available**. When a plugin provides only a subset of operations, faster-blaster automatically generates missing vtable entries using 4 strategies, ordered by overhead (zero → minimal → last resort).

**Philosophy**: The dispatch system naturally selects the fastest implementation, but fallbacks guarantee robust operation across all hardware.

## Architecture

### 1. Vtable Structure (Dual Entry Design)

```c
// include/faster-blaster/backend_interface.h (or similar)

typedef struct fb_backend_vtable {
    // ===== SPECIFIC ENTRIES (Zero-Overhead BLAS Paths) =====
    // BLAS Level 1
    void (*saxpy)(const int n, const float alpha, const float* x, const int incx, float* y, const int incy);
    void (*daxpy)(const int n, const double alpha, const double* x, const int incx, double* y, const int incy);
    // ... 56 Level 1 ops (S/D/C/Z variants)
    
    // BLAS Level 2
    void (*sgemv)(const char trans, const int m, const int n, const float alpha, 
                  const float* A, const int lda, const float* x, const int incx,
                  const float beta, float* y, const int incy);
    // ... 74 Level 2 ops
    
    // BLAS Level 3 - Standard
    void (*sgemm)(const char transa, const char transb, 
                  const int m, const int n, const int k,
                  const float alpha, const float* A, const int lda,
                  const float* B, const int ldb,
                  const float beta, float* C, const int ldc);
    void (*dgemm)(...);
    void (*cgemm)(...);
    void (*zgemm)(...);
    // ... 27 Level 3 ops
    
    // BLAS Level 3 - Batched
    void (*sgemm_batch_strided)(const char transa, const char transb,
                                const int m, const int n, const int k,
                                const float alpha, const float* A, const int lda, const int strideA,
                                const float* B, const int ldb, const int strideB,
                                const float beta, float* C, const int ldc, const int strideC,
                                const int batch_count);
    // ... 4 batched variants (S/D/C/Z)
    
    // ===== UNIFIED ENTRIES (Advanced Features) =====
    fb_status_t (*gemm_unified)(
        fb_precision_t precision,
        fb_batch_mode_t batch_mode,
        fb_fusion_t fusion,
        const char transa, const char transb,
        const size_t m, const size_t n, const size_t k,
        const void* alpha, const void* A, const size_t lda, const size_t strideA,
        const void* B, const size_t ldb, const size_t strideB,
        const void* beta, void* C, const size_t ldc, const size_t strideC,
        const void* bias, fb_activation_t activation,
        const size_t batch_count, void* stream
    );
    
    fb_status_t (*normalize_unified)(
        fb_norm_mode_t mode,
        fb_norm_phase_t phase,
        const size_t* dims, size_t ndims,
        const int* normalize_axes, size_t num_axes,
        const void* input, void* output,
        const void* scale, const void* bias,
        const void* running_mean, const void* running_var,
        float momentum, float epsilon, int num_groups,
        float scale_min, float scale_max,
        const void* grad_output, void* grad_input, void* grad_scale, void* grad_bias
    );
    
    fb_status_t (*reduce_unified)(
        fb_reduce_op_t operation,
        fb_reduce_scope_t scope,
        const void* input, void* output,
        const size_t* input_dims, size_t ndims,
        const int* reduce_axes, size_t num_axes,
        bool keepdims,
        fb_scan_mode_t scan_mode,
        const void* scan_init,
        fb_comm_t communicator, int root_rank,
        void* scratch_buffer, size_t scratch_size,
        void* stream
    );
    
    // ... More unified operations as needed
    
} fb_backend_vtable_t;
```

### 2. Auto-Fill Strategies

Implemented in `src/core/plugin_registry.c`:

```c
/**
 * Strategy 1: Unified ↔ Specific (Zero Overhead)
 * 
 * If plugin provides unified but not specific: Generate specific wrappers
 * If plugin provides specific but not unified: Generate unified dispatcher
 * 
 * Example: vtable->sgemm = NULL + vtable->gemm_unified = impl
 *          → auto-generate sgemm wrapper that calls gemm_unified
 */
fb_status_t fb_autofill_unified_to_specific(fb_backend_vtable_t* vtable);
fb_status_t fb_autofill_specific_to_unified(fb_backend_vtable_t* vtable);

/**
 * Strategy 2: Batched → Single (Zero Overhead)
 * 
 * If plugin provides batched: Generate single-item version with batch_count=1
 * 
 * Example: vtable->sgemm = NULL + vtable->sgemm_batch = impl
 *          → sgemm(...) { return sgemm_batch(..., 1); }
 */
fb_status_t fb_autofill_batched_to_single(fb_backend_vtable_t* vtable);

/**
 * Strategy 3: Array → Strided (Lightweight)
 * 
 * If plugin provides array-of-pointers batching: Generate strided version
 * Build {A, A+stride, A+2*stride, ...} array, call array version
 */
fb_status_t fb_autofill_array_to_strided(fb_backend_vtable_t* vtable);

/**
 * Strategy 4: Higher Precision → Lower Precision (Last Resort)
 * 
 * If plugin provides only higher precision: Generate lower precision with
 * promotion/demotion + overflow checking
 * 
 * Example: vtable->sgemm = NULL + vtable->dgemm = impl
 *          → promote float→double, compute, demote with range check
 * 
 * Trade-off: 2× memory overhead + allocation, but guarantees correctness
 */
fb_status_t fb_autofill_precision_promotion(fb_backend_vtable_t* vtable);
```

### 3. Entry Point: Plugin Finalization

```c
// src/core/plugin_registry.c

/**
 * Finalize vtable after plugin registration
 * 
 * Applies auto-fill strategies in order (zero overhead → last resort)
 * Logs warnings for Strategy 4 usage (slow path active)
 * 
 * Called automatically by fb_load_plugin()
 */
fb_status_t fb_finalize_plugin_vtable(fb_backend_vtable_t* vtable) {
    fb_status_t status;
    
    // Strategy 1: Unified ↔ Specific (zero overhead)
    status = fb_autofill_unified_to_specific(vtable);
    if (status != FB_STATUS_SUCCESS) return status;
    
    status = fb_autofill_specific_to_unified(vtable);
    if (status != FB_STATUS_SUCCESS) return status;
    
    // Strategy 2: Batched → Single (zero overhead)
    status = fb_autofill_batched_to_single(vtable);
    if (status != FB_STATUS_SUCCESS) return status;
    
    // Strategy 3: Array → Strided (lightweight)
    status = fb_autofill_array_to_strided(vtable);
    if (status != FB_STATUS_SUCCESS) return status;
    
    // Strategy 4: Precision promotion (last resort)
    status = fb_autofill_precision_promotion(vtable);
    if (status != FB_STATUS_SUCCESS) return status;
    
    return FB_STATUS_SUCCESS;
}
```

## Code Generation Tasks

### Task 1: Generate Unified → Specific Wrappers

**Input**: Unified operations (gemm_unified, normalize_unified, reduce_unified)  
**Output**: Specific wrappers (sgemm, dgemm, batch_norm_forward, etc.)

**Example Generated Code**:

```c
// codegen/generated/wrappers_gemm_specific.c

void fb_sgemm_from_unified(
    const char transa, const char transb,
    const int m, const int n, const int k,
    const float alpha, const float* A, const int lda,
    const float* B, const int ldb,
    const float beta, float* C, const int ldc
) {
    // Get plugin vtable
    const fb_backend_vtable_t* vtable = fb_get_active_vtable();
    if (!vtable || !vtable->gemm_unified) {
        return; // Error handling
    }
    
    // Call unified with appropriate flags
    vtable->gemm_unified(
        FB_PREC_FP32,           // precision
        FB_BATCH_SINGLE,        // batch_mode
        FB_FUSION_NONE,         // fusion
        transa, transb,
        m, n, k,
        &alpha, A, lda, 0,      // strideA=0 (not used for single)
        B, ldb, 0,              // strideB=0
        &beta, C, ldc, 0,       // strideC=0
        NULL,                   // bias (no fusion)
        FB_ACT_NONE,            // activation
        1,                      // batch_count=1
        NULL                    // stream (default)
    );
}

// Vtable filler - called during finalization
void fb_vtable_fill_sgemm_from_unified(fb_backend_vtable_t* vtable) {
    if (vtable->sgemm == NULL && vtable->gemm_unified != NULL) {
        vtable->sgemm = fb_sgemm_from_unified;
    }
}
```

**Codegen Template** (codegen/templates/unified_to_specific.template):

```
// Wrapper: {specific_op} from {unified_op}
{return_type} fb_{specific_op}_from_unified({params}) {{
    const fb_backend_vtable_t* vtable = fb_get_active_vtable();
    if (!vtable || !vtable->{unified_op}) {{
        return {error_value};
    }}
    
    vtable->{unified_op}(
        {unified_params}
    );
}}

void fb_vtable_fill_{specific_op}_from_unified(fb_backend_vtable_t* vtable) {{
    if (vtable->{specific_op} == NULL && vtable->{unified_op} != NULL) {{
        vtable->{specific_op} = fb_{specific_op}_from_unified;
    }}
}}
```

### Task 2: Generate Specific → Unified Dispatchers

**Input**: Specific operations (sgemm, dgemm, cgemm, zgemm)  
**Output**: Unified dispatcher (gemm_unified)

**Example Generated Code**:

```c
// codegen/generated/wrappers_gemm_unified.c

fb_status_t fb_gemm_unified_from_specific(
    fb_precision_t precision,
    fb_batch_mode_t batch_mode,
    fb_fusion_t fusion,
    const char transa, const char transb,
    const size_t m, const size_t n, const size_t k,
    const void* alpha, const void* A, const size_t lda, const size_t strideA,
    const void* B, const size_t ldb, const size_t strideB,
    const void* beta, void* C, const size_t ldc, const size_t strideC,
    const void* bias, fb_activation_t activation,
    const size_t batch_count, void* stream
) {
    const fb_backend_vtable_t* vtable = fb_get_active_vtable();
    if (!vtable) return FB_STATUS_ERROR;
    
    // Basic validation
    if (batch_mode != FB_BATCH_SINGLE) {
        return FB_STATUS_NOT_SUPPORTED; // Batched requires specific batched ops
    }
    if (fusion != FB_FUSION_NONE || activation != FB_ACT_NONE) {
        return FB_STATUS_NOT_SUPPORTED; // Fusion requires unified implementation
    }
    
    // Dispatch by precision
    switch (precision) {
        case FB_PREC_FP32:
            if (vtable->sgemm) {
                vtable->sgemm(transa, transb, m, n, k, 
                             *(const float*)alpha, A, lda, B, ldb, 
                             *(const float*)beta, C, ldc);
                return FB_STATUS_SUCCESS;
            }
            break;
        case FB_PREC_FP64:
            if (vtable->dgemm) {
                vtable->dgemm(transa, transb, m, n, k,
                             *(const double*)alpha, A, lda, B, ldb,
                             *(const double*)beta, C, ldc);
                return FB_STATUS_SUCCESS;
            }
            break;
        // ... C64, C128 cases
    }
    
    return FB_STATUS_NOT_SUPPORTED;
}

void fb_vtable_fill_gemm_unified_from_specific(fb_backend_vtable_t* vtable) {
    if (vtable->gemm_unified == NULL &&
        (vtable->sgemm != NULL || vtable->dgemm != NULL ||
         vtable->cgemm != NULL || vtable->zgemm != NULL)) {
        vtable->gemm_unified = fb_gemm_unified_from_specific;
    }
}
```

### Task 3: Generate Batched → Single Wrappers

**Example**:

```c
// codegen/generated/wrappers_batched_to_single.c

void fb_sgemm_from_batched(
    const char transa, const char transb,
    const int m, const int n, const int k,
    const float alpha, const float* A, const int lda,
    const float* B, const int ldb,
    const float beta, float* C, const int ldc
) {
    const fb_backend_vtable_t* vtable = fb_get_active_vtable();
    if (!vtable || !vtable->sgemm_batch_strided) {
        return; // Error
    }
    
    // Call batched with batch_count=1
    vtable->sgemm_batch_strided(
        transa, transb, m, n, k,
        alpha, A, lda, 0,  // strideA doesn't matter for single
        B, ldb, 0,         // strideB doesn't matter
        beta, C, ldc, 0,   // strideC doesn't matter
        1                  // batch_count=1
    );
}
```

### Task 4: Generate Precision Promotion Wrappers

**Example**:

```c
// codegen/generated/wrappers_precision_promotion.c

void fb_sgemm_from_dgemm(
    const char transa, const char transb,
    const int m, const int n, const int k,
    const float alpha, const float* A, const int lda,
    const float* B, const int ldb,
    const float beta, float* C, const int ldc
) {
    const fb_backend_vtable_t* vtable = fb_get_active_vtable();
    if (!vtable || !vtable->dgemm) {
        return; // Error
    }
    
    // Allocate double-precision buffers
    double* A_d = fb_promote_array_f32_to_f64(A, m * k);
    double* B_d = fb_promote_array_f32_to_f64(B, k * n);
    double* C_d = fb_promote_array_f32_to_f64(C, m * n);
    
    if (!A_d || !B_d || !C_d) {
        // Cleanup and error
        fb_free_temp_buffer(A_d);
        fb_free_temp_buffer(B_d);
        fb_free_temp_buffer(C_d);
        return;
    }
    
    // Promote scalar parameters
    double alpha_d = (double)alpha;
    double beta_d = (double)beta;
    
    // Call double-precision implementation
    vtable->dgemm(transa, transb, m, n, k,
                  alpha_d, A_d, lda, B_d, ldb,
                  beta_d, C_d, ldc);
    
    // Demote with overflow checking
    fb_status_t status = fb_demote_array_f64_to_f32_checked(C_d, C, m * n);
    
    // Cleanup
    fb_free_temp_buffer(A_d);
    fb_free_temp_buffer(B_d);
    fb_free_temp_buffer(C_d);
    
    if (status == FB_STATUS_OVERFLOW) {
        // Log warning: "FP64→FP32 demotion overflow detected"
    }
}

// Helper: Promote float array to double
double* fb_promote_array_f32_to_f64(const float* input, size_t count) {
    double* output = (double*)malloc(count * sizeof(double));
    if (!output) return NULL;
    
    for (size_t i = 0; i < count; i++) {
        output[i] = (double)input[i];
    }
    return output;
}

// Helper: Demote double array to float with overflow checking
fb_status_t fb_demote_array_f64_to_f32_checked(const double* input, float* output, size_t count) {
    fb_status_t status = FB_STATUS_SUCCESS;
    
    for (size_t i = 0; i < count; i++) {
        double val = input[i];
        
        // Check for overflow
        if (val > (double)FLT_MAX || val < (double)-FLT_MAX) {
            status = FB_STATUS_OVERFLOW;
            // Clamp to float range
            output[i] = (val > 0) ? FLT_MAX : -FLT_MAX;
        } else {
            output[i] = (float)val;
        }
    }
    
    return status;
}
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Define `fb_backend_vtable_t` with dual entries (specific + unified)
- [ ] Implement `fb_finalize_plugin_vtable()` entry point
- [ ] Add vtable auto-fill hooks to `fb_load_plugin()`
- [ ] Create utility functions:
  - `fb_promote_array_f32_to_f64()`
  - `fb_demote_array_f64_to_f32_checked()`
  - `fb_allocate_temp_buffer()` / `fb_free_temp_buffer()`

### Phase 2: Strategy 1 - Unified ↔ Specific (Week 2)
- [ ] Create codegen templates for unified→specific wrappers
- [ ] Generate 81 GEMM wrappers (sgemm, dgemm, etc. from gemm_unified)
- [ ] Generate 17 normalization wrappers (batch_norm_forward, etc. from normalize_unified)
- [ ] Generate 24 reduction wrappers (tensor_reduce_sum, etc. from reduce_unified)
- [ ] Implement `fb_autofill_unified_to_specific()`
- [ ] Create codegen templates for specific→unified dispatchers
- [ ] Implement `fb_autofill_specific_to_unified()`
- [ ] **Total**: ~122 wrappers generated

### Phase 3: Strategy 2 - Batched → Single (Week 3)
- [ ] Identify all batched operations in vtable
- [ ] Generate batched→single wrappers for GEMM, GEMV, etc.
- [ ] Implement `fb_autofill_batched_to_single()`
- [ ] **Total**: ~48 wrappers

### Phase 4: Strategy 3 - Array → Strided (Week 3)
- [ ] Generate array→strided wrappers for batched operations
- [ ] Implement pointer array construction logic
- [ ] Implement `fb_autofill_array_to_strided()`
- [ ] **Total**: ~16 wrappers

### Phase 5: Strategy 4 - Precision Promotion (Week 4)
- [ ] Generate FP64→FP32 wrappers (~87 BLAS ops)
- [ ] Generate C128→C64 wrappers (~87 BLAS ops)
- [ ] Generate INT32→INT8 wrappers (~15 quantization ops)
- [ ] Implement overflow detection and logging
- [ ] Implement `fb_autofill_precision_promotion()`
- [ ] **Total**: ~189 wrappers

### Phase 6: Testing & Validation (Week 5)
- [ ] Unit tests for each strategy
- [ ] Integration tests with minimal plugins (provide only specific or only unified)
- [ ] Performance validation: Zero overhead for Strategies 1-3
- [ ] Correctness validation: Strategy 4 matches expected results
- [ ] Documentation: Update PLUGIN_ARCHITECTURE.md with auto-fill system

## Testing Strategy

### Unit Tests

```c
// tests/test_vtable_autofill.c

void test_unified_to_specific_gemm() {
    // Create vtable with only unified gemm
    fb_backend_vtable_t vtable = {0};
    vtable.gemm_unified = mock_gemm_unified_impl;
    
    // Run auto-fill
    fb_autofill_unified_to_specific(&vtable);
    
    // Verify sgemm, dgemm, cgemm, zgemm are populated
    assert(vtable.sgemm != NULL);
    assert(vtable.dgemm != NULL);
    
    // Verify they call unified correctly
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0, 0, 0, 0};
    vtable.sgemm('N', 'N', 2, 2, 2, 1.0f, A, 2, B, 2, 0.0f, C, 2);
    
    // Check mock was called with correct parameters
    assert(mock_last_call.precision == FB_PREC_FP32);
    assert(mock_last_call.batch_mode == FB_BATCH_SINGLE);
}

void test_precision_promotion_overflow() {
    fb_backend_vtable_t vtable = {0};
    vtable.dgemm = mock_dgemm_impl;
    
    fb_autofill_precision_promotion(&vtable);
    assert(vtable.sgemm != NULL);
    
    // Test with values that cause overflow
    double large_val = 1e100; // Larger than FLT_MAX
    // ... verify overflow is detected and clamped
}
```

### Integration Tests

```c
// tests/test_plugin_minimal.c

void test_plugin_provides_only_unified() {
    // Register plugin that provides ONLY gemm_unified
    fb_backend_plugin_t plugin = {
        .metadata = &minimal_plugin_metadata,
        .probe = minimal_probe,
        .init = minimal_init,
        .get_vtable = minimal_get_vtable_unified_only,
        .shutdown = minimal_shutdown
    };
    
    fb_register_plugin(&plugin);
    fb_load_best_plugin();
    
    // Verify traditional BLAS calls work via auto-filled wrappers
    float A[4], B[4], C[4];
    sgemm('N', 'N', 2, 2, 2, 1.0f, A, 2, B, 2, 0.0f, C, 2);
    
    // Should succeed even though plugin didn't provide sgemm
    assert(fb_get_last_status() == FB_STATUS_SUCCESS);
}
```

## Performance Considerations

### Zero-Overhead Guarantee (Strategies 1-3)

Inlining and compiler optimization eliminate wrapper overhead:

```c
// Wrapper (source):
void fb_sgemm_from_unified(...) {
    vtable->gemm_unified(FB_PREC_FP32, FB_BATCH_SINGLE, FB_FUSION_NONE, ...);
}

// After optimization (assembly):
// Direct jump to vtable->gemm_unified, no stack frame for wrapper
// Parameter setup is same as direct call
```

**Benchmark target**: <1% overhead vs direct vtable call

### Last Resort Warning (Strategy 4)

When precision promotion is used:

```c
if (status == FB_STATUS_OVERFLOW) {
    log_warn("[faster-blaster] FP64→FP32 demotion overflow in sgemm. "
             "Consider using dgemm directly or install FP32-native backend.");
}
```

**Benchmark expectation**: 2-3× slower than native precision (acceptable for "better than crashing")

## Anti-Patterns

**Never Auto-Route**:
- ❌ Lower precision → higher precision (precision loss in computation)
- ❌ Real ↔ Complex (type system incompatibility)
- ❌ Dense → Sparse / Sparse → Dense (semantic differences)
- ❌ Format conversions (CSR ↔ COO ↔ ELL) - expensive, user should convert explicitly
- ❌ Transpose emulation (GEMV from GEMM) - defeats specialized kernels

## Success Criteria

1. **Correctness**: All auto-filled operations produce identical results to native implementations
2. **Performance**: Strategies 1-3 have <1% overhead vs direct calls
3. **Coverage**: 100% of vtable entries have either native or auto-filled implementation
4. **Robustness**: No `FB_STATUS_NOT_SUPPORTED` failures for operations that are mathematically possible
5. **Transparency**: Clear logging when Strategy 4 (slow path) is used

## Next Steps After Code Generation

After implementing vtable auto-fill codegen:

**Option D: Backend Plugin Implementation**
- Update cuBLAS plugin with dual vtable entries
- Update MKL plugin with dual vtable entries
- Update rocBLAS plugin with dual vtable entries
- Test runtime auto-fill with real backends
- Benchmark zero-overhead validation

---

**Document Status**: Design specification for Option C implementation  
**Last Updated**: January 2026  
**Related**: FASTER-BLASTER-OPERATIONS-SUPERSET.md (Section 6.1-6.3), copilot-instructions.md (Runtime Vtable Auto-Fill section)
