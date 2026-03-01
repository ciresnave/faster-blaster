# faster-blaster AI Agent Guide

## Project Overview

faster-blaster is a **plugin-based BLAS/LAPACK/related dispatch library** that automatically builds backend libraries from source with hardware-specific optimizations, then selects the optimal backend at runtime based on CPU/GPU detection. Think of it as a "universal adapter" that provides maximum linear algebra performance without manual configuration.

**Key Innovation**: Operation-level backend selection (not device-level). Different operations can use different backends on the same device simultaneously for best-tool-for-job optimization.

## Architecture: 3-Layer Plugin System

### Layer 1: Plugin Interface ([backend_plugin.h](include/faster-blaster/backend_plugin.h))
- **Probe-Score-Init Lifecycle**: Each plugin probes hardware, returns 0-100 compatibility score
- **9 Active Plugins**: 5 CPU (AOCL-BLIS, BLIS, OpenBLAS, MKL, Accelerate) + 4 GPU (cuBLAS, rocBLAS, oneMKL, Metal)
- **Capability Flags**: Plugins declare supported operations (BLAS L1/L2/L3, LAPACK, precisions)
- **Hardware Scoring Examples**: 
  - AOCL-BLIS on AMD Zen3: score=95 (vendor-optimized)
  - OpenBLAS on AMD Zen3: score=75 (generic, but functional)
  - MKL on Intel i7: score=95 (Intel's own)

### Layer 2: Backend Registry ([src/core/plugin_registry.c](src/core/plugin_registry.c))
- Maintains linked list of registered plugins
- `fb_load_best_plugin()`: Probes all → selects highest score → initializes
- Plugins register on `fb_init_plugins()` via `fb_register_plugin()`

### Layer 3: Operation Dispatch ([src/api/](src/api/))
- Queries registry for selected backend's vtable
- Routes `fb_saxpy()` → backend's implementation via function pointers
- Zero runtime overhead for single-backend scenarios

## Critical Build System Details

### Source-Built Backends (Primary Path)
**Entry Point**: [configure_optimized.ps1](configure_optimized.ps1) (Windows) or [configure_optimized.sh](configure_optimized.sh) (Linux/macOS)

**What Happens**:
1. Auto-detects CPU vendor via CPUID (AMD Zen2/3/4, Intel Skylake/Ice Lake, ARM)
2. Auto-detects GPUs (CUDA, ROCm, OpenCL via device APIs)
3. Triggers [cmake/BuildBackendsFromSource.cmake](cmake/BuildBackendsFromSource.cmake)
4. Builds BLIS, OpenBLAS, CLBlast with `-march=native`, architecture-specific kernels
5. Installs to `build/backends-install/<backend>/`
6. **Performance**: 10-30% faster than generic prebuilt binaries

**Key CMake Variables**:
- `FB_BUILD_BACKENDS_FROM_SOURCE=ON`: Enable source builds (default)
- `FB_BUILD_BLIS_FROM_SOURCE=ON`: Build BLIS with Zen4/AVX512 kernels
- `FB_BUILD_OPENBLAS_FROM_SOURCE=ON`: Build OpenBLAS with clang-cl + OpenMP 5.0
- `FB_BUILD_CLBLAST_FROM_SOURCE=ON`: Build CLBlast for GPU acceleration

### System-Installed Backends (Fallback)
**Entry Point**: [cmake/BackendInstaller.cmake](cmake/BackendInstaller.cmake)

Uses custom Find modules ([cmake/Find*.cmake](cmake/)):
- `FindAOCL.cmake`: Searches AMD AOCL installation (BLIS + libFLAME)
- `FindROCBLAS.cmake`: Searches ROCm installation (rocBLAS + rocSOLVER + hipBLAS)
- `FindMKL.cmake`: Searches Intel oneAPI MKL installation

**Important**: vcpkg auto-detection is built-in ([CMakeLists.txt:6-15](CMakeLists.txt#L6-L15)):
```cmake
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
```

## Build-Time Code Generation (Advanced)

### Automated Wrapper Generation ([codegen/](codegen/))
**Problem**: 1248+ BLAS/LAPACK operations across 9 backends = 11,232 potential wrappers to hand-write  
**Solution**: `fb_codegen` C tool extracts signatures from headers, generates wrappers automatically

**Workflow**:
1. [codegen/src/extractor.c](codegen/src/extractor.c): Parses `cblas_*` and `*_` (Fortran) signatures
2. [codegen/src/generator.c](codegen/src/generator.c): Generates wrapper code via templates
3. [codegen/src/library_loader.c](codegen/src/library_loader.c): Creates runtime DLL/SO loader
4. **Output**: `backends/generated/operations_<backend>.h` (included by plugins)

**When to Regenerate**: Only when adding new BLAS/LAPACK operations or new backends. CMake runs codegen automatically during build.

## Development Patterns

### Adding a New Plugin
1. Create [src/plugins/plugin_mynewbackend.c](src/plugins/)
2. Implement `fb_backend_plugin_t` interface (probe, init, shutdown, get_vtable)
3. Register in [src/core/plugin_registry.c](src/core/plugin_registry.c): `fb_register_mynewbackend_plugin()`
4. Add CMake Find module: [cmake/FindMyNewBackend.cmake](cmake/)
5. **Reference**: [docs/PLUGIN_ARCHITECTURE.md](docs/PLUGIN_ARCHITECTURE.md) lines 200-350 show complete plugin template

### Testing Convention
- CPU backends: [tests/test_<backend>_backend.c](tests/)
- GPU backends: [tests/test_<backend>_basic.c](tests/) (may be `.cpp` for C++ APIs)
- Use GoogleTest for C++ backends, plain C assertions for C backends
- **Run tests**: `cd build && ctest` (or `ctest -C Release` on Windows)

### Precision Variants
**BLAS naming**: `{s|d|c|z}{operation}` → `saxpy` (single), `daxpy` (double), `caxpy` (complex single), `zaxpy` (complex double)  
**Wrapper macros** in [src/plugins/](src/plugins/): See `DEFINE_BLAS_LEVEL1_OP`, `DEFINE_BLAS_LEVEL3_OP` patterns

### Important: CBLAS vs Fortran Conventions
- **CBLAS**: `cblas_saxpy(n, alpha, x, incx, y, incy)` — pass-by-value scalars
- **Fortran**: `saxpy_(&n, &alpha, x, &incx, y, &incy)` — pass-by-reference everything
- Many backends use Fortran → wrappers convert (see [ARCHITECTURE_REFERENCE_INTEGRATION.txt:61-71](ARCHITECTURE_REFERENCE_INTEGRATION.txt#L61-L71))

## API Design Principles

### Unified Implementation + Domain-Specific Aliases Pattern

**Core Philosophy**: When multiple sections provide functionally equivalent operations (same underlying computation, different naming conventions or contexts), we:

1. **Centralize the implementation** in the most appropriate unified section
2. **Create zero-cost inline aliases** in original locations for backward compatibility
3. **Document the relationship** with performance notes and migration paths
4. **Count implementations, not aliases** for accurate complexity metrics

#### Example: GEMM Unification

**Unified Implementation** (Section 6: Unified Linear Algebra):
```c
fb_gemm_unified(
    fb_precision_t precision,     // FP64, FP32, BF16, FP16, FP8, INT8
    fb_fusion_t fusion,           // NONE, RELU, GELU, BIAS, etc.
    fb_batch_mode_t batch_mode,   // SINGLE, ARRAY, STRIDED
    const void* A, const void* B, void* C, ...
);
```

**BLAS Alias** (Section 3 - zero-cost inline):
```c
static inline void sgemm(...) {
    fb_gemm_unified(FB_PREC_FP32, FB_FUSION_NONE, FB_BATCH_SINGLE, ...);
}
```

**Benefits**:
- Single implementation to maintain and optimize
- Familiar APIs preserved (BLAS users find `sgemm`, DNN users find `fb_gemm_fused_relu`)
- Zero runtime overhead (inlining eliminates function call)
- Clear upgrade path documented

#### When to Apply This Pattern

✅ **DO unify when**:
- Operations perform identical mathematical computation
- Different sections use different names for same functionality
- Parameter differences are trivial (type conversions, flag enums)
- Example: `sgemm` (BLAS) ≈ `fb_lpgemm_fp32` (AOCL-DLP) ≈ `fb_gemm_fused` (Tensor Fusion)

❌ **DON'T unify when**:
- Algorithms fundamentally differ (dense vs sparse matrix multiply)
- Data structures incompatible (row-major vs CSR format)
- Performance characteristics diverge significantly
- Example: `fb_gemv()` (dense) ≠ `fb_spmv()` (sparse) - keep separate

#### Categories Already Unified

1. **GEMM Operations**: 82 variants → 1 implementation + 81 aliases
   - Standard BLAS GEMM (S/D/C/Z)
   - Batched GEMM (array/strided)
   - Mixed-precision GEMM (BF16, FP16, FP8, INT8)
   - Fused GEMM (activation, bias, residual)
   - Low-precision GEMM (AOCL-DLP quantization)
   
   **Vtable Design**:
   ```c
   typedef struct {
       // Specific entries for zero-overhead BLAS calls
       void (*sgemm)(...);  // Direct BLAS params → vtable->sgemm → backend
       void (*dgemm)(...);
       void (*cblas_sgemm_batch_strided)(...);
       // ... 79 more specific variants
       
       // Unified entry for advanced features
       fb_status_t (*gemm_unified)(
           fb_precision_t precision,
           fb_batch_mode_t batch_mode,
           fb_fusion_t fusion,
           // ... unified params
       );
   } fb_backend_vtable_t;
   ```
   
   **Backend Implementation Strategies**:
   - **cuBLAS/MKL/rocBLAS**: Implement all specific wrappers + unified dispatcher
   - **Future backends**: Implement ONLY unified, use fb_codegen to auto-generate specific wrappers
   
   **User Call Paths**:
   - BLAS user: `sgemm(...)` → `vtable->sgemm(...)` → `cublasSgemm(...)` (2 calls, 0 conversions)
   - Advanced user: `fb_gemm_unified(...)` → `vtable->gemm_unified(...)` → plugin dispatch (2-3 calls)

#### Runtime Vtable Auto-Fill

When a plugin loads, faster-blaster automatically generates missing vtable entries as fallbacks. This ensures **correctness everywhere, performance where available** - the dispatch system naturally selects the fastest implementation, but always has a working fallback.

**Auto-Fill Strategies** (applied in order after plugin registration):

1. **Unified ↔ Specific (Zero Overhead)**
   - If plugin provides unified but not specific: Generate specific wrappers from unified
   - If plugin provides specific but not unified: Generate unified dispatcher from specific ops
   - Example: `vtable->sgemm = NULL` + `vtable->gemm_unified = impl` → auto-generate `sgemm` wrapper
   - **~81 operations** per unified category (GEMM, normalization, reduction)

2. **Batched → Single (Zero Overhead)**
   - If plugin provides batched operation: Generate single-item version with `batch_count=1`
   - Example: `vtable->sgemm = NULL` + `vtable->sgemm_batch = impl` → `sgemm(...) { return sgemm_batch(1, ...); }`
   - **~48 operations** (all batched BLAS variants)

3. **Array → Strided (Lightweight)**
   - If plugin provides array-of-pointers batching: Generate strided version with pointer array construction
   - Example: Build `{A, A+stride, A+2*stride, ...}` array, call array version
   - **~16 operations** (strided batched variants)

4. **Higher Precision → Lower Precision (Last Resort)**
   - If plugin provides only higher precision: Generate lower precision with promotion/demotion + overflow checking
   - Example: `vtable->sgemm = NULL` + `vtable->dgemm = impl` → promote float→double, compute, demote with range check
   - **Trade-off**: 2× memory + allocation overhead, but ensures correctness when no native implementation exists
   - **~189 operations** (FP64→FP32: 87 ops, C128→C64: 87 ops, INT32→INT8: 15 ops)
   - **Key**: faster-blaster's dispatch system naturally avoids these (slow), but they prevent `FB_STATUS_NOT_SUPPORTED` failures

**Implementation**:
```c
// src/core/plugin_registry.c

fb_status_t fb_finalize_plugin_vtable(fb_backend_vtable_t* vtable) {
    // Strategy 1: Unified ↔ Specific
    if (vtable->gemm_unified == NULL && vtable->sgemm != NULL) {
        vtable->gemm_unified = fb_generate_unified_from_specific(vtable);
    } else if (vtable->sgemm == NULL && vtable->gemm_unified != NULL) {
        vtable->sgemm = fb_generate_specific_from_unified(vtable, FB_PREC_FP32);
    }
    
    // Strategy 2: Batched → Single
    if (vtable->sgemm == NULL && vtable->sgemm_batch != NULL) {
        vtable->sgemm = fb_generate_single_from_batched(vtable->sgemm_batch);
    }
    
    // Strategy 3: Array → Strided
    if (vtable->sgemm_batch_strided == NULL && vtable->sgemm_batch != NULL) {
        vtable->sgemm_batch_strided = fb_generate_strided_from_array(vtable->sgemm_batch);
    }
    
    // Strategy 4: Precision promotion (last resort)
    if (vtable->sgemm == NULL && vtable->dgemm != NULL) {
        vtable->sgemm = fb_generate_promoted_precision(vtable->dgemm, 
                                                        FB_PREC_FP64, FB_PREC_FP32);
    }
    
    return FB_STATUS_SUCCESS;
}

// Precision promotion wrapper (example)
fb_status_t fb_sgemm_from_dgemm_wrapper(/* sgemm params */) {
    // 1. Promote float→double (lossless)
    double* A_d = fb_promote_array_f32_to_f64(A, m*k);
    double* B_d = fb_promote_array_f32_to_f64(B, k*n);
    double* C_d = fb_promote_array_f32_to_f64(C, m*n);
    
    // 2. Call double-precision implementation
    fb_status_t status = vtable->dgemm(..., A_d, B_d, C_d, ...);
    
    // 3. Demote with overflow checking
    if (status == FB_STATUS_SUCCESS) {
        status = fb_demote_array_f64_to_f32_checked(C_d, C, m*n);
    }
    
    // 4. Cleanup
    fb_free_temp_buffer(A_d);
    fb_free_temp_buffer(B_d);
    fb_free_temp_buffer(C_d);
    
    return status;
}
```

**Anti-Patterns (Never Auto-Route)**:
- ❌ Lower precision → higher precision (precision loss in computation)
- ❌ Real ↔ Complex (type system incompatibility)
- ❌ Dense → Sparse or Sparse → Dense (semantic differences)
- ❌ Format conversions (CSR ↔ COO ↔ ELL) - expensive, user should convert explicitly
- ❌ Transpose emulation (`GEMV` from `GEMM`) - defeats specialized kernels

**Philosophy**: faster-blaster provides correctness guarantees (operations never fail if mathematically possible) while letting the dispatch system optimize for performance. Auto-generated wrappers are "slow but correct" fallbacks that rarely win performance comparisons, but ensure robust operation across all hardware.

2. **Normalization Operations**: ~30 variants → 1 implementation + 29 aliases
   - Z-score normalization (Statistics)
   - Batch/Layer/Instance/Group normalization (DNN)
   - L1/L2/max normalization (Preprocessing)
   - Min-max scaling

3. **Reduction Operations**: ~20 variants → 1 implementation + 19 aliases
   - Local reduction (Parallel Primitives: sum, min, max, product)
   - Statistical reduction (Statistics: sum, mean, variance)
   - Collective reduction (NCCL/RCCL: AllReduce, ReduceScatter)

#### Implementation Checklist

When implementing a unified operation:

1. **Design unified API signature**
   - Use enums for variant selection (precision, fusion type, etc.)
   - Single entry point with comprehensive parameters
   - Internal dispatch to optimized kernels

2. **Create dual vtable entries**
   - **Specific entries**: `sgemm`, `dgemm`, etc. (direct BLAS param order for zero-overhead)
   - **Unified entry**: `gemm_unified` (takes enum flags for advanced features)
   - Backends choose: implement all specific ops + unified dispatcher, OR only unified (with generated wrappers)

3. **Create inline alias wrappers**
   - Zero-cost: use `static inline` or macro
   - Match original signature exactly
   - Forward to vtable->specific_op (NOT to unified)
   - Example: `sgemm(...)` → `vtable->sgemm(...)` (2 calls: dispatch + backend)
   - NOT: `sgemm(...)` → `fb_gemm_unified(...)` → `vtable->gemm_unified(...)` → `backend->sgemm(...)` (4 calls!)

4. **Update documentation**
   - Mark original sections with "→ Unified in Section X"
   - Add performance note: "This is an alias. For full control, use `fb_*_unified()`"
   - Document migration path

4. **Update operation counts**
   - Specification: "Implementations: X, Aliases: Y, Total API Surface: X+Y"
   - Internal tracking: Count implementations only

5. **Test both paths**
   - Verify alias behaves identically to legacy API
   - Benchmark to ensure zero overhead
   - Check compiler generates identical assembly

#### Documentation Template for Aliases

```markdown
### 3.8 BLAS Level 3: SGEMM (Alias)

**Single-precision matrix multiply**: C := α*A*B + β*C

```c
void sgemm(...)  // Alias → fb_gemm_unified(FB_PREC_FP32, ...)
```

⚠️ **Performance Note**: This is a zero-cost convenience alias. For maximum control 
over batching, fusion, and mixed-precision, use `fb_gemm_unified()` directly.

✅ **Migration Path**: 
- Drop-in replacement: No code changes needed
- Performance optimization: `fb_gemm_unified(FB_PREC_FP32, FB_FUSION_NONE, FB_BATCH_SINGLE, ...)`
- Advanced features: Enable fusion with `FB_FUSION_RELU`, batching with `FB_BATCH_STRIDED`

→ **Unified Implementation**: See Section 6.2 - Unified GEMM Operations
```

---

## Auto-Convention Vtable Loading

### Overview

When a backend DLL/SO is loaded, faster-blaster can **automatically enumerate every
exported symbol**, classify each by its naming convention, and fill the matching
`ext_ops[op_id][conv]` slot — with no manual `typedef` or `GetProcAddress` boilerplate
required in the plugin.  For any convention slot that the library does not export, a
static cross-convention **thunk** is installed automatically, so all three conventions
are always reachable.

### Calling Convention Enum (`fb_conv_t`)

```c
// src/backends/backend_interface.h
typedef enum {
    FB_CONV_CBLAS    = 0,  // cblas_saxpy(n, alpha, x, incx, y, incy)   — pass-by-value scalars
    FB_CONV_FORTRAN  = 1,  // saxpy_(&n, &alpha, x, &incx, y, &incy)    — everything by pointer, trailing _
    FB_CONV_REF      = 2,  // saxpy_ref(n, alpha, x, incx, y, incy)     — C convention, _ref suffix
    FB_CONV_COUNT    = 3
} fb_conv_t;
```

### 2-D `ext_ops` Table

```c
// Was (1D):
fb_generic_fn ext_ops[FB_JUDGE_MAX_OPERATIONS];

// Now (2D):
fb_generic_fn ext_ops[FB_JUDGE_MAX_OPERATIONS][FB_CONV_COUNT];
// Access:  vtable->ext_ops[FB_OP_SAXPY][FB_CONV_CBLAS]
```

Every op × convention pair has its own slot.  The dispatch layer picks the best
available slot: native slots win, thunk-generated slots serve as fallbacks.

### Symbol Classifier (`fb_classify_symbol`)

Three naming patterns cover all major BLAS/LAPACK libraries:

| Pattern | Convention | Example |
|---|---|---|
| `cblas_<stem>` | `FB_CONV_CBLAS` | `cblas_saxpy` |
| `<stem>_` (trailing underscore) | `FB_CONV_FORTRAN` | `saxpy_` |
| `<stem>_ref` | `FB_CONV_REF` | `saxpy_ref` |

```c
// src/core/backend_auto_detect.c
fb_conv_t fb_classify_symbol(const char *name, uint32_t *out_op_id);
// Returns FB_CONV_COUNT if the symbol is not a recognised BLAS/LAPACK operation.
```

The classifier strips the prefix/suffix, looks up the canonical stem in the
**reverse `stem → FB_OP_*` table** (see `src/core/op_vtable_map.c`), and returns the
convention.

### Export Enumerator (`fb_enumerate_and_populate`)

```c
// Enumerate all DLL/SO exports, classify each, fill ext_ops[op][conv]
void fb_enumerate_and_populate(fb_backend_vtable_t *vtable, void *lib_handle);
```

**Platform implementations** (all in `src/core/backend_auto_detect.c`):
- **Windows**: Walk the PE Export Directory (`IMAGE_EXPORT_DIRECTORY`) directly from
  the in-memory module.
- **Linux/BSD**: Iterate `.dynsym` section via `dl_iterate_phdr` + `ElfW(Sym)`.
- **macOS**: Walk Mach-O `LC_DYSYMTAB` / `nlist` table via `dlopen`+`dyld`.

Typical plugin `init()` replaces ~100 lines of typedef boilerplate with:

```c
// Before (manual):
typedef void (*saxpy_t)(int, float, const float*, int, float*, int);
saxpy_t saxpy_fn = (saxpy_t)GetProcAddress(handle, "cblas_saxpy");
vtable->saxpy = ...;
// ... × 200 operations

// After (automatic):
fb_enumerate_and_populate(vtable, handle);   // fills all 2266 × 3 slots in one pass
```

### ABI Compatibility: Why CBLAS Needs No Thunk

`CBLAS_TRANSPOSE` and `fb_transpose_t` share identical integer values by design:

```c
CBLAS_NO_TRANS=111 == FB_NO_TRANS=111
CBLAS_TRANS=112    == FB_TRANS=112
CBLAS_CONJ_TRANS=113 == FB_CONJ_TRANS=113
```

All vtable enum parameters match CBLAS values. A CBLAS function pointer can therefore
be assigned directly to the matching `ext_ops[op][FB_CONV_CBLAS]` slot with a plain
cast — no thunk, no conversion, zero overhead.

### Cross-Convention Thunks (`conv_thunks.c`)

**Fortran↔CBLAS thunks are necessary** because Fortran passes every argument by
pointer while CBLAS passes scalars by value.  Thunks are arity- and type-specific (one
per operation), but they are entirely **mechanical and codegen-able** from the typed
field declarations in `backend_interface.h`.

Static thunks live in `src/core/conv_thunks.c` and are installed by
`fb_finalize_plugin_vtable()` (Strategy 5 — new, runs after existing Strategies 1-4):

```c
// Strategy 5: fill empty conv slots from occupied ones via thunks
for (uint32_t op = 0; op < FB_JUDGE_MAX_OPERATIONS; op++) {
    if (vtable->ext_ops[op][FB_CONV_CBLAS] != NULL &&
        vtable->ext_ops[op][FB_CONV_FORTRAN] == NULL) {
        vtable->ext_ops[op][FB_CONV_FORTRAN] = k_cblas_to_fortran_thunks[op];
    }
    if (vtable->ext_ops[op][FB_CONV_FORTRAN] != NULL &&
        vtable->ext_ops[op][FB_CONV_CBLAS] == NULL) {
        vtable->ext_ops[op][FB_CONV_CBLAS] = k_fortran_to_cblas_thunks[op];
    }
    // FB_CONV_REF ↔ FB_CONV_CBLAS (same ABI; _ref implementations use C conv)
    if (vtable->ext_ops[op][FB_CONV_REF] != NULL &&
        vtable->ext_ops[op][FB_CONV_CBLAS] == NULL) {
        vtable->ext_ops[op][FB_CONV_CBLAS] = vtable->ext_ops[op][FB_CONV_REF];
    }
}
```

Example generated thunk for `saxpy`:

```c
// cblas_saxpy → saxpy_   (CBLAS→Fortran wrapper)
static void thunk_saxpy_cblas_to_fortran(
        int n, float alpha, const float *x, int incx, float *y, int incy) {
    // Fortran expects everything by pointer
    int    n_    = n;
    float  a_    = alpha;
    int    incx_ = incx;
    int    incy_ = incy;
    ((void(*)(int*,float*,const float*,int*,float*,int*))
        g_saxpy_fortran)(&n_, &a_, x, &incx_, y, &incy_);
}
```

### Reverse Stem Table (for classifier)

Located in `src/core/op_vtable_map.c` (appended after the existing forward table):

```c
// AUTO-GENERATED by gen_sym_tables.py — do not edit manually
static const struct { const char *stem; uint32_t op_id; } k_op_stem_map[] = {
    { "sasum",   FB_OP_SASUM   },
    { "dasum",   FB_OP_DASUM   },
    { "saxpy",   FB_OP_SAXPY   },
    { "daxpy",   FB_OP_DAXPY   },
    // ... 237 total entries
};
uint32_t fb_stem_to_op_id(const char *stem);  // binary search over k_op_stem_map
```

### `fb_sym_entry_t` Extension

The pre-generated symbol tables in `src/backends/sym_tables/` use `fb_sym_entry_t`.
After this change, the struct gains a `conv` field:

```c
// Was:
typedef struct { uint32_t op_id; const char *symbol; } fb_sym_entry_t;

// Now:
typedef struct { uint32_t op_id; fb_conv_t conv; const char *symbol; } fb_sym_entry_t;
```

`gen_sym_tables.py` is updated to emit the `conv` field.  Existing tables are all
`cblas_*` → `FB_CONV_CBLAS`; ScaLAPACK tables are `p*_` → `FB_CONV_FORTRAN`.

### Dead Code Removed by This Mechanism

| File | Reason |
|---|---|
| `src/backends/blas_lapack_reference_backend.c` | Dead DLL-load code; DLL was never built |
| `src/backends/blas_lapack_reference_backend.h` | Header for dead file |
| `src/plugins/plugin_blas_lapack_reference.c` | Plugin wiring the dead backend |

### Impact on Existing Plugins

In `plugin_openblas.c`, `plugin_aocl_blis.c`, `plugin_standard_blis.c`, `plugin_mkl.c`:

```c
// Remove: ~100 lines of per-symbol manual typedef + GetProcAddress
// Replace with single call in init():
fb_enumerate_and_populate(vtable, handle);
```

Named vtable fields (`vtable->saxpy`, `vtable->sgemm`, etc.) are still populated by
`fb_vtable_sync_ext_ops()` which mirrors `ext_ops[op][FB_CONV_CBLAS]` into them —
so no callers need to change.

### Key Files

| File | Role |
|---|---|
| `src/backends/backend_interface.h` | `fb_conv_t` enum; 2D `ext_ops[MAX_OPS][FB_CONV_COUNT]` |
| `src/backends/backend_auto_detect.h` | Updated `fb_sym_entry_t`; new function declarations |
| `src/core/backend_auto_detect.c` | `fb_enumerate_and_populate`, `fb_classify_symbol`, PE/ELF/Mach-O scanner |
| `src/core/op_vtable_map.c` | Reverse `stem→FB_OP_*` table; `fb_stem_to_op_id()` |
| `src/core/conv_thunks.c` | Static per-op Fortran↔CBLAS thunk arrays |
| `src/backends/sym_tables/` | Pre-generated `fb_sym_entry_t` tables (now with `conv` field) |

---

## Hardware Detection Implementation

### CPU Detection ([src/device_detection/](src/device_detection/))
- Uses cpuinfo library (fetched via CMake FetchContent)
- Detects: Vendor (AMD/Intel/ARM), microarchitecture (Zen3, Ice Lake), features (AVX2, AVX512)
- Scoring logic in each plugin's `probe()` function

### GPU Detection
- **NVIDIA**: CUDA driver API (`cuInit`, `cuDeviceGetCount`)
- **AMD**: HIP API (`hipGetDeviceCount`)
- **Intel**: Level Zero API (`zeDriverGet`)
- **Apple**: Metal framework

**Key File**: Each plugin implements hardware detection in its `probe()` function.

## Common Workflows

### Build Everything from Source (Maximum Performance)
```powershell
# Windows
.\configure_optimized.ps1
cd build && cmake --build . --config Release -j

# Linux/macOS
./configure_optimized.sh
cd build && cmake --build . -j$(nproc)
```

### Quick Development Build (Use System Libraries)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFB_BUILD_BACKENDS_FROM_SOURCE=OFF
cmake --build build -j
```

### Add New BLAS Operation to All Backends
1. Update [../FASTER-BLASTER-OPERATIONS-SUPERSET.md](../FASTER-BLASTER-OPERATIONS-SUPERSET.md) specification
2. Run codegen: `cmake --build build --target fb_codegen`
3. Wrappers auto-regenerate in `backends/generated/`
4. Update tests in [tests/](tests/)

### Debug Plugin Selection
Enable verbose logging in user code:
```c
#define FB_DEBUG 1  // Shows: "[DEBUG] Probing plugin: aocl-blis (score=95)"
#include <faster-blaster/faster_blaster.h>
```

## Key Files Reference

| File | Purpose |
|------|---------|
| [../FASTER-BLASTER-OPERATIONS-SUPERSET.md](../FASTER-BLASTER-OPERATIONS-SUPERSET.md) | Complete 1248-operation specification |
| [BACKEND_STATUS_SUMMARY.md](BACKEND_STATUS_SUMMARY.md) | What's implemented in each backend |
| [docs/PLUGIN_ARCHITECTURE.md](docs/PLUGIN_ARCHITECTURE.md) | Detailed plugin design patterns |
| [docs/BUILDING_FROM_SOURCE.md](docs/BUILDING_FROM_SOURCE.md) | Source build optimization guide |
| [ROADMAP.md](ROADMAP.md) | Future: operation-level selection, hybrid dispatch |
| [cmake/BuildBackendsFromSource.cmake](cmake/BuildBackendsFromSource.cmake) | Auto-build orchestration |

## Multi-Workspace Context

### [blis/](../blis/) 
Upstream BLIS library source. faster-blaster builds this as a backend. **Do not modify** — changes belong upstream.

### [faster-blaster-reference/](../../../projects/faster-blaster-reference/)
**Complete open-source BLAS/LAPACK/etc. reference implementation in modern C23.**

This is a **standalone project of significant value** — a fully compliant implementation of all 1248+ BLAS/LAPACK operations, and a large variety of related operations relevant to scientific computing and machine learning, from [FASTER-BLASTER-OPERATIONS-SUPERSET.md](../FASTER_BLASTER_OPERATIONS_SUPERSET.md), written entirely in modern C without Fortran dependencies.

#### Purpose & Design Philosophy
- **Gold Standard Testing**: Serves as the correctness baseline for validating faster-blaster backends
- **Precision over Performance**: Implementations prioritize numerical accuracy and IEEE 754 compliance
- **Platform Independence**: Pure C23 with no assembly, SIMD, or platform-specific code
- **Educational Value**: Readable, well-documented implementations of complex linear algebra algorithms
- **Standalone Utility**: Usable independently on platforms lacking vendor-optimized BLAS/LAPACK

#### Key Characteristics
- **Language**: C23 with C23 features where available (compound literals, constexpr)
- **No Fortran**: Unlike Netlib reference LAPACK, entirely C-based for modern toolchain compatibility
- **Complex Numbers**: Native C23 `_Complex` types (`float _Complex`, `double _Complex`)
- **IEEE 754 Compliance**: Proper handling of NaN, Inf, denormals, signed zeros
- **All Precision Variants**: Complete S/D/C/Z (single/double/complex single/complex double) for all operations

#### Integration with faster-blaster
1. **Correctness Testing**: Each backend operation is validated against faster-blaster-reference results
2. **Backend Plugin**: Can be loaded as a fallback backend via `plugin_reference.c` (slower but guaranteed correct)
3. **Fortran Compatibility Layer**: faster-blaster provides CBLAS-to-Fortran wrappers for backends expecting `*_` calling convention
4. **Shared Specification**: Both projects implement [FASTER-BLASTER-OPERATIONS-SUPERSET.md](../FASTER-BLASTER-OPERATIONS-SUPERSET.md)

#### Development Context
- **Build System**: Separate CMake project in `faster-blaster-reference/`
- **File Organization**: 
  - `src/blas/` - BLAS Level 1/2/3 implementations
  - `src/lapack/` - LAPACK computational, driver, auxiliary routines
  - `include/` - Public headers (cblas.h, lapacke.h style)
  - `tests/` - Operation-level correctness tests
- **Test Strategy**: Tests compare outputs against known-good results (typically from Netlib or vendor libraries)
- **No Modifications from faster-blaster**: Changes to reference implementations should be made in faster-blaster-reference workspace, not from faster-blaster code

#### Why This Matters
Most open-source BLAS/LAPACK implementations are either:
- Fortran-based (Netlib reference) - harder to integrate with modern C++ projects
- Performance-optimized (OpenBLAS, BLIS) - complex assembly, hard to verify correctness
- Vendor-specific (MKL, AOCL) - proprietary or architecture-locked

faster-blaster-reference fills the gap: **a readable, trustworthy, pure-C reference** that's easy to audit, port, and use as a correctness oracle.

## Code Completeness Requirements

**REQUIREMENT**: Before marking ANY BLAS or LAPACK operation as "complete," BOTH the reference implementation and CBLAS wrapper MUST be implemented, tested, and passing.

> **Architecture note**: faster-blaster-reference exports CBLAS as its primary API. Fortran ABI compatibility (`*_` trailing-underscore symbols) is provided automatically by consolidated per-level wrapper files (`blas_l1_fortran_wrappers.c`, etc.) that delegate to `cblas_*`. Do NOT add per-operation Fortran wrapper bodies to individual source files. faster-blaster's `conv_thunks.c` provides Fortran↔CBLAS conversion at dispatch time, so the reference library does not need to export Fortran symbols at all for correctness in normal use.

### Two Required Variants for Every Operation

For each operation (e.g., `saxpy`, `sgemv`, `csymv`), provide:

1. **Reference Implementation (`*_ref`)**
   - **Signature**: `void saxpy_ref(int n, float alpha, const float *x, int incx, float *y, int incy)`
   - **Location**: Source file (e.g., `src/blas/level1/saxpy.c`)
   - **Purpose**: Portable, correctness-focused, educational reference
   - **Calling Convention**: C convention (pass-by-value scalars)

2. **CBLAS Wrapper (`cblas_*`)**
   - **Signature**: `void cblas_saxpy(int n, float alpha, const float *x, int incx, float *y, int incy)`
   - **Location**: CBLAS wrappers file (e.g., `src/blas/level1/blas_l1_cblas_wrappers.c`)
   - **Purpose**: C BLAS standard interface — the primary exported API
   - **Calling Convention**: C convention (pass-by-value scalars, arrays as pointers)
   - **Header Declaration**: Must be in appropriate `blas_l*_reference.h` file

### Fortran ABI (Consolidated — Do NOT add per-operation wrappers)

Fortran `*_` wrappers are maintained in consolidated files, **not** in individual operation source files:
- `src/blas/level1/blas_l1_fortran_wrappers.c` — all L1 Fortran wrappers, calling `cblas_*`
- `src/blas/level2/blas_l2_fortran_wrappers.c` — all L2 Fortran wrappers, calling `cblas_*`
- `src/blas/level3/blas_l3_fortran_wrappers.c` — all L3 Fortran wrappers, calling `cblas_*`

When adding a new operation, add its `foo_` Fortran wrapper to the appropriate consolidated file. Do not declare `foo_` in headers.

### Verification Checklist

Before declaring operation complete:

- [ ] `*_ref` implementation exists and compiles
- [ ] `cblas_*` wrapper exists and compiles
- [ ] Both declarations in header file match implementations
- [ ] No "undeclared identifier" compiler errors
- [ ] All precision variants (S/D/C/Z where applicable) implemented
- [ ] Unit tests exist for the CBLAS variant
- [ ] **ALL tests passing** (0 failures, 100% pass rate)
- [ ] Integration tests pass (cross-module dependencies)
- [ ] Changes committed to Git with clear commit message

### Red Flags (Operation Incomplete If)

❌ `_ref` implementation missing  
❌ `cblas_*` wrapper missing  
❌ Fortran `*_` body added to an individual operation `.c` file (use consolidated wrapper file instead)  
❌ Function declared in header but not implemented  
❌ Tests pass locally but fail in CI  
❌ Partial coverage (e.g., S/D precisions but not C/Z)  
❌ Any compiler errors or test failures  
❌ Changes not committed to Git  
❌ Missing header declarations causing linker errors  

### Example: Complete Implementation

For `saxpy` (single-precision AXPY):

```c
// src/blas/level1/saxpy.c
BLAS_L1_KERNEL void saxpy_ref(int n, float alpha, const float *x, int incx, float *y, int incy) {
    // Reference implementation
}

// src/blas/level1/blas_l1_cblas_wrappers.c
void cblas_saxpy(int n, float alpha, const float *x, int incx, float *y, int incy) {
    saxpy_ref(n, alpha, x, incx, y, incy);
}

// src/blas/level1/blas_l1_fortran_wrappers.c  (consolidated — NOT in saxpy.c)
void saxpy_(int *n, float *alpha, float *x, int *incx, float *y, int *incy) {
    cblas_saxpy(*n, *alpha, x, *incx, y, *incy);  // delegate to cblas_, not _ref
}

// include/blas_l1_reference.h
BLAS_L1_KERNEL void saxpy_ref(int n, float alpha, const float *x, int incx, float *y, int incy);
void cblas_saxpy(int n, float alpha, const float *x, int incx, float *y, int incy);
// Note: saxpy_  is NOT declared in this header (lives only in fortran_wrappers.c)
```

### Project-Specific Notes

- **faster-blaster-reference**: CBLAS is the primary API; Fortran ABI is a compatibility layer in consolidated wrapper files
- **faster-blaster (main)**: References faster-blaster-reference as the correctness oracle; provides Fortran↔CBLAS thunks automatically via `conv_thunks.c`
- **Build verification**: `cd build-extended && ninja 2>&1 | Where-Object { $_ -match 'error:' } | Where-Object { $_ -notmatch 'test_cgesv|test_zgesv|test_chesv|test_zhesv' }` must produce no output
- **C23 standard**: All code must compile with `-std=c23` flag

---

## Critical Constraints

1. **C23/C++20 Standards**: Core is C23, GPU backends may use C++20 for SYCL/Metal
2. **No Global State**: Each plugin has isolated context (thread-safe)
3. **Zero Runtime Overhead Goal**: Function pointers resolve once at init, not per-call
4. **LAPACK Support**: Only MKL, AOCL (libFLAME), OpenBLAS, rocSOLVER have LAPACK — BLIS is BLAS-only
5. **Windows Quirks**: 
   - OpenBLAS requires clang-cl (not MSVC) due to inline assembly
   - BLIS requires Cygwin/WSL for configure script
   - ROCm not officially supported on Windows (HIP works via CUDA backend)

## When to Read Full Docs

- **Adding backends**: [docs/PLUGIN_ARCHITECTURE.md](docs/PLUGIN_ARCHITECTURE.md) Section 5 (lines 200-350)
- **Build failures**: [docs/BUILDING_FROM_SOURCE.md](docs/BUILDING_FROM_SOURCE.md) Troubleshooting (lines 300-460)
- **CMake mysteries**: [cmake/README.md](cmake/README.md) Backend discovery details
- **Code generation**: [codegen/README.md](codegen/README.md) Architecture and extension guide

## Testing Philosophy

**Unit tests** → Backend-specific ([tests/test_*_backend.c](tests/))  
**Integration tests** → Plugin selection logic ([tests/test_plugin_selection.c](tests/))  
**Performance tests** → [benchmarks/](benchmarks/) (GoogleBenchmark-based)

**Golden Rule**: New operations require tests in at least 3 backends (AOCL-BLIS, OpenBLAS, MKL) for validation.

---

**Next Steps for New Contributors**: Read [ROADMAP.md](ROADMAP.md) Phase 3 section to understand the vision for operation-level backend selection and hybrid CPU/GPU dispatch.
