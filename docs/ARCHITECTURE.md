# Faster-BLASTER Architecture Design Document

## Executive Summary

Faster-BLASTER is a high-performance BLAS abstraction layer that automatically selects optimal backend implementations based on hardware-specific calibration. This document describes the architectural decisions, implementation strategy, and design rationale.

## Core Design Principles

### 1. Zero-Overhead Dispatch
After calibration, function calls should have **zero dispatch overhead** - achieved through direct function pointers in a lookup table. No runtime branching or decision-making in the hot path.

### 2. Hardware-Specific Optimization
Different hardware has different performance characteristics. A calibration run benchmarks all backends on the actual hardware and caches results for instant future initialization.

### 3. Correctness First, Performance Second
All backend implementations are verified for correctness before performance data is trusted. Numerical precision is measured and tracked.

### 4. Crowdsourced Performance Database
Users can contribute their calibration data to build a comprehensive database of performance characteristics across hardware configurations.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                         │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │   Public C API (BLAS)        │
         │  - faster_blaster.h          │
         │  - faster_blaster_ext.h      │
         │  - faster_blaster_config.h   │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │      Dispatch System          │
         │  - Function table lookup      │
         │  - Thread-local contexts      │
         │  - Statistics (optional)      │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼────────────────────────────┐
         │         Backend Registry                   │
         │  - Plugin management                       │
         │  - Enable/disable backends                 │
         │  - Runtime reconfiguration                 │
         └───────────────┬────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────────────┐
    │                    │                            │
┌───▼────┐     ┌─────────▼──────┐     ┌─────────────▼─────┐
│OpenBLAS│     │  Intel MKL     │     │   NVIDIA cuBLAS   │
│ (BSD)  │     │ (Proprietary)  │     │  (Proprietary)    │
└───┬────┘     └────────┬───────┘     └─────────┬─────────┘
    │                   │                        │
    └───────────────────┴────────────────────────┘
                        │
                ┌───────▼────────┐
                │   Hardware     │
                │  CPU / GPU     │
                └────────────────┘
```

## Module Breakdown

### 1. Public API Layer (`include/`)

#### `faster_blaster.h`
Standard CBLAS-compatible interface. Drop-in replacement for any BLAS library.

**Functions**: All BLAS Level 1, 2, 3 operations
- Level 1: Vector-vector (swap, scal, copy, axpy, dot, nrm2, asum, iamax)
- Level 2: Matrix-vector (gemv, symv, trmv, trsv, ger)
- Level 3: Matrix-matrix (gemm, symm, syrk, syr2k, trmm, trsm)

**Data types**: float, double, complex float, complex double

#### `faster_blaster_ext.h`
Extended operations beyond standard BLAS:
- **Batched operations**: Process multiple matrices in one call
- **Strided batched**: Regularly-spaced matrix arrays
- **Mixed precision**: Different types for input/compute/output
- **Fused operations**: GEMM + activation, GEMM + bias + activation
- **Operation chains**: Multi-operation fusion for reduced memory traffic

#### `faster_blaster_config.h`
Runtime configuration and diagnostics:
- Backend enable/disable
- Force specific backend
- Query dispatch decisions
- Calibration control
- Hardware information
- Performance statistics
- Logging and debugging

### 2. Core System (`src/core/`)

#### Hardware Detection (`hardware_detect.h`)
Uses cpuinfo and OpenCL for cross-platform detection.

**Detects**:
- CPU: vendor, model, cores, cache sizes, instruction sets
- GPU: vendor, model, compute capability, memory (all vendors via OpenCL)
- Memory: total RAM

**Generates**: Unique hardware fingerprint (hash) for calibration caching

**CPU Features**: SSE2/3/4, AVX, AVX2, AVX-512, FMA, NEON, SVE, AMX

**GPU Detection**: OpenCL for vendor-neutral enumeration (NVIDIA, AMD, Intel, Apple)

#### Calibration System (`calibration.h`)
Comprehensive benchmarking framework.

**Problem Size Matrix**:
- Level 1: Vectors from 16 to 16M elements
- Level 2: Matrices from 16×16 to 16384×16384
- Level 3: Matrices from 64×64 to 8192×8192

**Metrics Collected**:

*Essential:*
1. Execution time (median, min, max, stddev)
2. Correctness (vs reference implementation)
3. Numerical stability (ill-conditioned test matrices)

*Important:*
4. Memory footprint (peak, allocated, freed)
5. Memory bandwidth (GB/s)
6. Warmup overhead (first run vs steady state)

*Advanced:*
7. Power consumption (if available)
8. Precision degradation
9. Cache efficiency

**Output**: JSON database with full performance matrix

#### Dispatch System (`dispatch.h`)
Zero-overhead function routing.

**Dispatch Key**: (operation, dtype, layout, m, n, k, trans_a, trans_b)

**Lookup Process**:
1. Hash dispatch key → table index
2. Load function pointer from table
3. Direct call (no branching!)

**Thread Safety**: Thread-local contexts with per-backend handles

**Runtime Reconfiguration**: When backends are enabled/disabled, rebuild dispatch table from full performance matrix

### 3. Backend Plugin System (`src/backends/`)

#### Plugin Interface (`backend_interface.h`)
Standardized API that all backends must implement.

**Lifecycle Functions**:
- `init()`: Initialize backend
- `destroy()`: Cleanup
- `get_info()`: Query capabilities
- `create_context()`: Thread-local setup
- `destroy_context()`: Thread cleanup

**Operation Functions**: Function pointers for all BLAS operations

**Capabilities Flags**:
- Level 1/2/3 support
- Batched operations
- Strided operations
- Mixed precision
- GPU acceleration
- Sparse matrices
- Asynchronous execution

**Backends**:
- **Reference**: Naive implementation (always available)
- **OpenBLAS**: BSD licensed (bundled)
- **BLIS**: BSD licensed (bundled)
- **Intel MKL**: Proprietary (user installs)
- **NVIDIA cuBLAS**: CUDA required (user installs)
- **AMD rocBLAS**: ROCm required (user installs)
- **Apple Accelerate**: macOS only (system provided)

### 4. API Implementation (`src/api/`)

Thin wrappers that:
1. Get thread-local context
2. Create dispatch key from parameters
3. Lookup function pointer
4. Forward call to backend
5. (Optional) Record statistics

Example:
```c
void fb_sgemm(const FB_LAYOUT Layout,
              const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
              const int M, const int N, const int K,
              const float alpha,
              const float *A, const int lda,
              const float *B, const int ldb,
              const float beta,
              float *C, const int ldc)
{
    fb_thread_context_t *ctx = fb_dispatch_get_thread_context();
    
    fb_dispatch_key_t key;
    fb_dispatch_key_init(&key, FB_OP_GEMM, FB_DTYPE_FLOAT32, Layout, M, N, K);
    key.trans_a = TransA;
    key.trans_b = TransB;
    
    const fb_dispatch_entry_t *entry = fb_dispatch_lookup(global_table, &key);
    
    fb_gemm_fn func = (fb_gemm_fn)entry->function_ptr;
    func(ctx->backend_contexts[entry->backend_id],
         FB_DTYPE_FLOAT32, Layout, TransA, TransB,
         M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
```

## Calibration Process

### Phase 1: Hardware Detection (< 1 second)
1. Detect CPU model, cores, cache, instruction sets
2. Detect GPU (if present)
3. Generate hardware fingerprint
4. Check cache for existing calibration data

### Phase 2: Backend Discovery (< 1 second)
1. Scan for available backend plugins
2. Load plugin libraries
3. Query capabilities
4. Initialize backends

### Phase 3: Correctness Testing (~ 1 minute)
For each backend:
1. Run small test cases for each operation
2. Compare results to reference implementation
3. Verify bit-exact (integers) or epsilon-close (floats)
4. Fail fast if correctness issues found

### Phase 4: Performance Benchmarking (~ 5-10 minutes)
For each operation × backend × problem size:
1. Allocate test matrices/vectors
2. Initialize with random data
3. Warmup iterations (3×)
4. Timing iterations (10×)
5. Measure: time, memory, bandwidth
6. Test numerical stability (optional)

### Phase 5: Analysis and Dispatch Table Generation (< 1 second)
1. For each (operation, size, dtype), find fastest backend
2. Populate dispatch table with function pointers
3. Save full performance matrix to disk
4. Generate calibration report

## Performance Optimization Strategies

### 1. Size-Based Dispatch
Small matrices may be faster on CPU (low kernel launch overhead)
Large matrices benefit from GPU parallelism

Example dispatch decision:
```
SGEMM 128×128: OpenBLAS (CPU)  - 0.05ms
SGEMM 1024×1024: cuBLAS (GPU)  - 0.8ms
SGEMM 4096×4096: cuBLAS (GPU)  - 45ms
```

### 2. Layout-Aware Dispatch
Some backends optimize for row-major, others for column-major

### 3. Operation Fusion
Combine multiple operations to reduce memory traffic:
```
Traditional:  C = GEMM(A, B)
              C = ReLU(C)
              (2 memory passes)

Fused:        C = GEMM_ReLU(A, B)
              (1 memory pass, ~1.5× faster)
```

### 4. Batched Operations
Group multiple independent operations:
```
Traditional:  for i in 0..N:
                  GEMM(A[i], B[i], C[i])
              (N kernel launches)

Batched:      GEMM_BATCHED(A_array, B_array, C_array, N)
              (1 kernel launch, ~10× faster on GPU)
```

## Memory Management

### CPU Memory
- Standard malloc/free
- Alignment for SIMD (typically 64-byte for AVX-512)

### GPU Memory
- Managed by backend (cuBLAS, rocBLAS)
- Automatic host↔device transfers when needed
- Consider unified memory for newer GPUs

### Thread Safety
Each thread maintains its own:
- Backend context handles
- Temporary buffers
- Statistics counters

## Error Handling

### Initialization Errors
- Hardware detection failure
- Backend load failure
- Calibration failure

**Fallback**: Reference implementation always available

### Runtime Errors
- Invalid parameters (negative dimensions, NULL pointers)
- Memory allocation failure
- Backend execution failure

**Handling**: Return error codes, log to stderr (or user callback)

## Future Extensions

### 1. Sparse Matrix Support
Add sparse variants of operations (SpGEMM, SpMV, etc.)

### 2. Automatic Work Distribution
Split large operations across CPU+GPU for maximum throughput

### 3. Operation Fusion Compiler
Analyze sequences of BLAS calls and generate fused kernels

### 4. Online Adaptation
Monitor runtime performance and re-optimize dispatch if patterns change

### 5. Cloud Integration
Download optimal calibration from cloud database on first run

## Build System

### CMake Build
```bash
cmake -DENABLE_CUBLAS=ON -DENABLE_MKL=ON ..
make
```

### Rust Integration
No Rust dependency required. Hardware detection uses pure C with cpuinfo and OpenCL.

### Plugin System
Backends are separate shared libraries loaded at runtime:
```
libfasterblaster_openblas.so
libfasterblaster_cublas.so
libfasterblaster_mkl.so
```

## Testing Strategy

### Unit Tests
- Dispatch system correctness
- Hardware fingerprinting consistency
- Calibration data serialization

### Integration Tests
- All BLAS operations with multiple backends
- Thread safety (concurrent calls)
- Backend enable/disable

### Performance Tests
- Verify dispatch overhead is negligible
- Compare against individual backends
- Scalability tests

### Correctness Tests
- Compare all backends against reference
- Numerical stability with ill-conditioned matrices
- Edge cases (zero dimensions, unit strides, etc.)

## Licensing Strategy

**Core Library**: MIT OR Apache-2.0 (dual license)
- Maximum compatibility with other projects
- Same license as Rust ecosystem

**Bundled Backends**: BSD-3-Clause
- OpenBLAS, BLIS

**Optional Backends**: Various
- Intel MKL: Proprietary (not redistributed)
- NVIDIA cuBLAS: CUDA Toolkit license
- AMD rocBLAS: MIT

**User Responsibility**: Install proprietary backends separately

## Documentation Plan

1. **README.md**: Quick start, examples
2. **API Reference**: Doxygen-generated from headers
3. **Architecture Guide**: This document
4. **Calibration Guide**: How to contribute data
5. **Backend Development Guide**: How to add new backends
6. **Performance Tuning Guide**: Best practices for users

## Success Metrics

1. **Performance**: Match or exceed best individual backend for each operation
2. **Overhead**: < 1% dispatch overhead after calibration
3. **Compatibility**: Drop-in replacement for standard BLAS
4. **Coverage**: Support all common hardware (Intel/AMD CPU, NVIDIA/AMD GPU)
5. **Adoption**: Used in real-world projects
6. **Database**: Calibration data for 100+ hardware configurations

## Conclusion

Faster-BLASTER provides a comprehensive solution to the BLAS backend fragmentation problem. By automatically selecting optimal implementations based on hardware-specific calibration, it delivers maximum performance without requiring users to manually tune their code for different platforms.

The zero-overhead dispatch mechanism ensures that this convenience comes at no runtime cost, making it suitable for performance-critical applications in scientific computing, machine learning, and numerical analysis.
