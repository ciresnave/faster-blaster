# LAPACK/SOLVER Library Strategy for Multi-Vendor Support

## Problem Statement

The LAPACK/SOLVER libraries have **API incompatibilities** between vendors:
- **cuSOLVER** (NVIDIA): Uses workspace buffer queries, specific calling conventions
- **rocSOLVER** (AMD): Different API, now recommends using rocBLAS handles directly
- **oneMKL** (Intel): SYCL-based C++ API, different from both

## Research Findings (December 2025)

### 1. **hipSOLVER - The Portability Layer** ✅ RECOMMENDED

**Discovery**: AMD provides **hipSOLVER** as a compatibility layer!

From AMD documentation:
> "The hipSOLVER API is designed to be similar to the cuSOLVER and rocSOLVER interfaces, but it requires some minor adjustments to ensure the best portability."

**Key Benefits**:
- Single API works on both NVIDIA and AMD GPUs
- Automatically translates to cuSOLVER on NVIDIA GPUs
- Automatically translates to rocSOLVER on AMD GPUs
- Minimal code changes needed

**Architecture**:
```
Your Code → hipSOLVER API
                ↓
    ┌───────────┴───────────┐
    ↓                       ↓
cuSOLVER (NVIDIA)    rocSOLVER (AMD)
```

**File**: `src/backends/gpu/hipsolver_trait_impl.c`
- Write once against hipSOLVER API
- Compiles with NVCC → uses cuSOLVER
- Compiles with hipcc → uses rocSOLVER
- **Best of both worlds!**

### 2. **MAGMA - Vendor-Neutral High-Performance Library** 🌟 ALTERNATIVE

**Discovery**: MAGMA (Matrix Algebra for GPU and Multicore Architectures) is a portable, high-performance library.

From recent research (2024):
> "MAGMA is a pivotal open-source library... appealing to many applications that require portable high performance across current and future DOE systems accelerated by GPUs from NVIDIA (Perlmutter and Polaris), AMD (Frontier and El Capitan), and Intel (Aurora)."

**Key Benefits**:
- **Vendor-neutral**: Single codebase for NVIDIA, AMD, Intel
- **Hybrid CPU-GPU algorithms**: Often faster than vendor libraries!
- **Wide coverage**: BLAS, LAPACK, Sparse, Eigensolvers, SVD
- **Open source**: https://github.com/icl-utk-edu/magma
- **Actively maintained**: Used in TOP500 supercomputers

**Drawbacks**:
- Extra dependency to install
- May not always match vendor-optimized performance
- Requires building from source

### 3. **oneMKL Interfaces - Cross-Vendor Dispatch** 🎯 INTEL'S SOLUTION

**Discovery**: Intel's **oneMKL Interfaces** library can dispatch to vendor libraries!

From Intel documentation:
> "oneMKL interfaces can dispatch to Intel® MKL as well as other vendor libraries"

**Supported Backends**:
```
oneMKL Interfaces
    ├── Intel MKL (CPU + GPU)
    ├── cuBLAS + cuSOLVER (NVIDIA)
    ├── rocBLAS + rocSOLVER (AMD)
    └── Netlib (CPU fallback)
```

**Benefits**:
- Runtime backend selection
- SYCL-based modern C++ API
- Can use vendor-optimized libraries under the hood

**Drawbacks**:
- C++ only (SYCL)
- Requires Intel oneAPI toolkit
- Not all operations dispatch to all backends equally

## Recommended Architecture

### **Strategy: Hybrid Approach with hipSOLVER as Primary** ✅

```
faster-blaster/
├── BLAS Operations (✅ DONE)
│   ├── cuBLAS backend (NVIDIA)
│   ├── rocBLAS backend (AMD)
│   └── oneMKL backend (Intel)
│
├── LAPACK Operations (📝 RECOMMENDED)
│   ├── hipSOLVER backend (NVIDIA + AMD unified!)
│   │   - Single implementation
│   │   - Compiles to cuSOLVER OR rocSOLVER
│   │   - Best compatibility
│   │
│   ├── oneMKL backend (Intel)
│   │   - Intel-specific LAPACK
│   │   - SYCL-based
│   │
│   └── MAGMA backend (Optional - All vendors)
│       - Vendor-neutral fallback
│       - Hybrid CPU-GPU algorithms
│       - For when vendor libs unavailable
```

### Implementation Plan

#### Phase 1: hipSOLVER for NVIDIA + AMD (HIGHEST PRIORITY)

**Create**: `src/backends/gpu/hipsolver_lapack_impl.c`

```c
// Single file that works on BOTH NVIDIA and AMD!
#include <hipsolver/hipsolver.h>

// Initialize hipSOLVER (works on both vendors)
hipsolverHandle_t handle;
hipsolverCreate(&handle);

// LU Factorization (same code for NVIDIA/AMD)
hipsolverDgetrf(handle, m, n, A, lda, workspace, ipiv, devInfo);

// When compiled with NVCC: calls cusolverDnDgetrf()
// When compiled with hipcc: calls rocsolver_dgetrf()
```

**Benefits**:
- ✅ Single codebase for 2 vendors
- ✅ Vendor-optimized performance
- ✅ No runtime overhead
- ✅ Easy to maintain

#### Phase 2: oneMKL LAPACK for Intel

**Create**: `src/backends/gpu/onemkl_lapack_impl.cpp`

Use oneMKL's LAPACK domain:
```cpp
#include <oneapi/mkl/lapack.hpp>

// Intel-specific LAPACK using SYCL
oneapi::mkl::lapack::getrf(queue, m, n, A, lda, ipiv);
```

#### Phase 3: MAGMA as Optional Fallback

For exotic hardware or when vendor libs unavailable:
```c
#include <magma_v2.h>

// Vendor-neutral, works everywhere
magma_dgetrf_gpu(m, n, dA, lda, ipiv, &info);
```

## Comparison Matrix

| Library               | NVIDIA   | AMD       | Intel  | API Style  | Maintenance     | Performance          |
| --------------------- | -------- | --------- | ------ | ---------- | --------------- | -------------------- |
| **hipSOLVER** ✅       | ✅        | ✅         | ❌      | C (HIP)    | AMD maintains   | Vendor-optimized     |
| **MAGMA** 🌟           | ✅        | ✅         | ✅      | C          | Open-source     | Hybrid (often best!) |
| **oneMKL Interfaces** | ✅        | ✅         | ✅      | C++ (SYCL) | Intel maintains | Dispatch overhead    |
| **Vendor-specific**   | cuSOLVER | rocSOLVER | oneMKL | Mixed      | Fragmented      | Best per-vendor      |

## Other SOLVER Libraries to Consider

### Apple Accelerate Framework
- **Platforms**: macOS, iOS (Apple Silicon)
- **API**: C with Accelerate.framework
- **LAPACK**: Full implementation
- **Recommendation**: Include for Apple platform support

```c
#include <Accelerate/Accelerate.h>

// Apple's optimized LAPACK
dgetrf_(&m, &n, A, &lda, ipiv, &info);
```

### SLATE (Software for Linear Algebra Targeting Exascale)
- **Platforms**: Multi-GPU, multi-node
- **Focus**: Distributed linear algebra
- **Use case**: When scaling beyond single GPU
- **Recommendation**: Future consideration for HPC workloads

### PLASMA (Parallel Linear Algebra Software for Multicore Architectures)
- **Platforms**: CPU-focused
- **Recommendation**: CPU fallback only

## Final Recommendation

### **Use hipSOLVER for LAPACK operations** ✅

**Rationale**:
1. **Solves 2 vendors with 1 implementation** (NVIDIA + AMD = 90% of market)
2. **Zero runtime overhead** (compile-time backend selection)
3. **Vendor-optimized performance** (uses cuSOLVER/rocSOLVER under hood)
4. **Minimal API differences** from cuSOLVER
5. **AMD-maintained** (reliable, up-to-date)

**Implementation Steps**:

1. **Update trait interface** to support LAPACK operations:
   ```c
   // In gpu_backend_trait.h
   typedef struct {
       // ... existing BLAS operations ...
       
       // LAPACK - LU factorization
       int (*getrf)(void* handle, fb_gpu_stream_t stream, int m, int n, ...);
       int (*getrs)(void* handle, fb_gpu_stream_t stream, char trans, ...);
       // ... more LAPACK ops ...
   } fb_gpu_backend_trait_t;
   ```

2. **Create hipSOLVER implementation**:
   - File: `src/backends/gpu/hipsolver_lapack_impl.c`
   - Uses `#include <hipsolver/hipsolver.h>`
   - Compiles with hipcc (AMD) or NVCC with HIP support (NVIDIA)

3. **Add Intel oneMKL LAPACK separately**:
   - File: `src/backends/gpu/onemkl_lapack_impl.cpp`
   - Uses `#include <oneapi/mkl/lapack.hpp>`
   - Compiles with Intel DPC++ compiler

4. **Optional: Add MAGMA fallback**:
   - For when vendor libraries unavailable
   - Works on all GPUs + CPUs

5. **Platform-specific**:
   - Apple: Use Accelerate framework on macOS/iOS

## Code Example: Unified LAPACK Interface

```c
// Application code - works with ANY backend!
fb_gpu_backend_trait_t* backend;

#ifdef USE_NVIDIA_OR_AMD
    backend = fb_get_hipsolver_trait();  // hipSOLVER (works on both!)
#elif defined(USE_INTEL)
    backend = fb_get_onemkl_trait();     // Intel oneMKL
#elif defined(USE_MAGMA)
    backend = fb_get_magma_trait();      // MAGMA (vendor-neutral)
#elif defined(USE_APPLE)
    backend = fb_get_accelerate_trait(); // Apple Accelerate
#endif

// Same API regardless of backend
backend->getrf(handle, stream, m, n, A, lda, ipiv, &info);
backend->potrf(handle, stream, 'U', n, A, lda, &info);
backend->geqrf(handle, stream, m, n, A, lda, tau, &info);
```

## Performance Expectations

Based on literature review:

| Operation            | cuSOLVER (NVIDIA)   | rocSOLVER (AMD)   | oneMKL (Intel) | MAGMA             |
| -------------------- | ------------------- | ----------------- | -------------- | ----------------- |
| **LU (getrf)**       | Fastest on RTX 4090 | Fastest on MI250X | Good on Arc    | Hybrid often wins |
| **Cholesky (potrf)** | Optimized           | Optimized         | Optimized      | Competitive       |
| **QR (geqrf)**       | Optimized           | Optimized         | Optimized      | Often fastest!    |
| **SVD (gesvd)**      | Good                | Good              | Good           | **MAGMA excels**  |

**Note**: MAGMA's hybrid CPU-GPU algorithms sometimes outperform vendor-specific implementations, especially for operations like SVD!

## Migration Path

1. ✅ **Immediate**: Use hipSOLVER for LAPACK (NVIDIA + AMD unified)
2. 🔄 **Soon**: Add oneMKL LAPACK (Intel support)
3. 📝 **Future**: Consider MAGMA for vendor-neutral fallback
4. 🍎 **Platform**: Add Apple Accelerate for macOS/iOS

This gives you **maximum coverage** with **minimum code duplication**!

## Resources

- **hipSOLVER Documentation**: https://rocm.docs.amd.com/projects/hipSOLVER/
- **MAGMA**: https://icl.utk.edu/magma/
- **oneMKL Interfaces**: https://github.com/oneapi-src/oneMKL
- **cuSOLVER**: https://docs.nvidia.com/cuda/cusolver/
- **rocSOLVER**: https://rocm.docs.amd.com/projects/rocSOLVER/

## Conclusion

**Recommended Strategy**: Use **hipSOLVER** as your primary LAPACK implementation.

**Why**:
- ✅ Solves NVIDIA + AMD with single codebase
- ✅ Vendor-optimized performance
- ✅ Maintained by AMD (reliable)
- ✅ Easy migration from cuSOLVER
- ✅ Zero abstraction overhead

**Add oneMKL for Intel, consider MAGMA for maximum portability!**
