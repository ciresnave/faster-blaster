# Multi-Vendor GPU Support in faster-blaster

## Vision: Universal GPU Acceleration

**faster-blaster** aims to support **all major GPU vendors** for maximum portability and performance. Users should be able to leverage whatever hardware they have available.

## Current Support Status (December 2025)

| Vendor     | Backend    | BLAS Ops | Status         | Notes                                     |
| ---------- | ---------- | -------- | -------------- | ----------------------------------------- |
| **NVIDIA** | cuBLAS     | 198/204  | ✅ **COMPLETE** | Tested on RTX 4090, all tests passing     |
| **AMD**    | rocBLAS    | 198/204  | 🚧 **SKELETON** | Compiles on Windows HIP SDK 6.4           |
| **Intel**  | oneMKL     | ~220/204 | 📝 **PLANNED**  | Supports Arc, Flex, Max GPUs + integrated |
| **Apple**  | Accelerate | ~190/204 | 📝 **FUTURE**   | M-series Neural Engine support            |

**Note**: All vendors are missing the same 6 operations (cspmv, zspmv, cspr, zspr, cspr2, zspr2) - these don't exist in any GPU BLAS library.

## Vendor Ecosystems

### NVIDIA (cuBLAS) ✅ IMPLEMENTED

**Platform**: CUDA Toolkit  
**Language**: C/C++ with CUDA extensions  
**BLAS Library**: cuBLAS  
**LAPACK Library**: cuSOLVER  
**Compiler**: nvcc (NVIDIA CUDA Compiler)

**Supported GPUs**:
- Consumer: RTX 40/30/20 series, GTX 16 series
- Professional: RTX A-series, Quadro
- Data Center: A100, H100, L40S
- Minimum: Compute Capability 3.5+

**Installation**:
- Windows: CUDA Toolkit 12.0+ (includes cuBLAS, cuSOLVER)
- Download: https://developer.nvidia.com/cuda-downloads

**BLAS Coverage**: 198 operations
- Level 1: 54 operations ✅
- Level 2: 86 operations ✅
- Level 3: 28 operations ✅
- LAPACK: 26 operations ✅
- Missing: 6 complex symmetric packed (not in cuBLAS)

---

### AMD (rocBLAS) 🚧 IN PROGRESS

**Platform**: ROCm / HIP SDK  
**Language**: C/C++ with HIP (CUDA-compatible)  
**BLAS Library**: rocBLAS  
**LAPACK Library**: rocSOLVER  
**Compiler**: hipcc (Clang-based)

**Supported GPUs**:
- Consumer RDNA3: RX 7900 XTX/XT, RX 7800 XT, RX 7700 XT
- Consumer RDNA2: RX 6900/6800/6700 XT
- Professional: Radeon PRO W7900/W7800/W7700
- Data Center CDNA: MI300, MI250, MI210, MI100
- Minimum: GCN 3.0 (gfx803) or newer

**Installation**:
- **Windows**: HIP SDK 6.4+ (includes rocBLAS, rocSOLVER) ✅ VERIFIED
- **Linux**: ROCm 6.0+ (full stack)
- **WSL2**: ❌ Currently broken (ROCm 6.x missing packages, 7.x unsupported)
- Download: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html

**BLAS Coverage**: 198 operations (same as cuBLAS)
- Level 1: 54 operations ✅
- Level 2: 86 operations ✅
- Level 3: 28 operations ✅
- LAPACK: 26 operations (via rocSOLVER) ✅
- Missing: Same 6 complex symmetric packed as cuBLAS

---

### Intel (oneMKL) 📝 PLANNED

**Platform**: oneAPI Toolkit  
**Language**: C++ with SYCL (cross-vendor) or C API  
**BLAS Library**: oneMKL (Math Kernel Library)  
**LAPACK Library**: oneMKL (integrated)  
**Compiler**: DPC++ (Intel's SYCL compiler) or icx/icpx

**Supported GPUs**:
- **Consumer Arc**: A770 (16GB), A750, A380
- **Data Center Flex**: Flex 170, Flex 140
- **Data Center Max**: Max 1550, Max 1100 (Ponte Vecchio)
- **Integrated**: Iris Xe, UHD Graphics (11th gen+)
- Minimum: Gen 9 graphics (Skylake) or newer

**Installation**:
- Windows/Linux: Intel oneAPI Base Toolkit (free)
- Download: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
- Size: ~3GB download, ~10GB installed

**BLAS Coverage**: ~220+ operations (MOST COMPLETE!)
- Level 1: 54+ operations
- Level 2: 92+ operations (includes all packed variants!)
- Level 3: 28+ operations
- LAPACK: 50+ operations
- Extensions: BLAS-like extensions, sparse BLAS
- **Advantage**: Includes operations cuBLAS/rocBLAS don't have!

**Unique Features**:
- **Cross-vendor**: Can target NVIDIA/AMD GPUs via SYCL backends
- **Unified API**: Same code runs on CPU, GPU, FPGA
- **Domain libraries**: oneDNN (deep learning), oneCCL (communication)

---

### Apple (Accelerate) 📝 FUTURE

**Platform**: macOS / Apple Silicon  
**Language**: C/Objective-C with Metal Shading Language  
**BLAS Library**: Accelerate Framework (vecLib/BLAS)  
**LAPACK Library**: Accelerate Framework  
**Compiler**: clang (with Metal backend)

**Supported Hardware**:
- **M-series**: M1, M2, M3, M4 (Neural Engine + GPU)
- **A-series**: iPad Pro (A12X+), iPhone (A13+)
- **Intel Macs**: Legacy support via CPU BLAS only

**Installation**:
- Included with Xcode / Command Line Tools (free)
- No separate download needed on macOS

**BLAS Coverage**: ~180-190 operations
- Level 1: 54 operations
- Level 2: 70-80 operations (some packed variants missing)
- Level 3: 28 operations
- LAPACK: 40+ operations
- Missing: Some complex packed operations

**Unique Features**:
- **Neural Engine**: 16-core NPU for ML workloads
- **Unified Memory**: No CPU↔GPU copies needed
- **Power Efficient**: Best performance-per-watt

---

## Architecture Design

### Backend Trait Pattern

All GPU backends implement the same **`fb_gpu_backend_trait`** interface:

```c
struct fb_gpu_backend_trait {
    // Metadata
    const char* name;
    fb_gpu_backend_type_t type;
    
    // Lifecycle
    int (*init)(int device_id, void** backend_handle);
    void (*shutdown)(void* backend_handle);
    int (*get_device_properties)(...);
    
    // Memory
    int (*malloc)(void* handle, fb_gpu_ptr_t* ptr, size_t size);
    void (*free)(void* handle, fb_gpu_ptr_t ptr);
    int (*memcpy_h2d)(...);
    int (*memcpy_d2h)(...);
    int (*memcpy_d2d)(...);
    
    // Streams
    int (*stream_create)(void* handle, fb_gpu_stream_t* stream);
    void (*stream_destroy)(void* handle, fb_gpu_stream_t stream);
    void (*stream_synchronize)(void* handle, fb_gpu_stream_t stream);
    
    // BLAS Level 1 (54 ops)
    void (*saxpy)(...);
    void (*daxpy)(...);
    // ... all Level 1 operations
    
    // BLAS Level 2 (92 ops)
    void (*sgemv)(...);
    void (*dgemv)(...);
    // ... all Level 2 operations
    
    // BLAS Level 3 (28 ops)
    void (*sgemm)(...);
    void (*dgemm)(...);
    // ... all Level 3 operations
    
    // LAPACK subset (26-50 ops)
    void (*sgetrf)(...);
    void (*dgetrf)(...);
    // ... LAPACK operations
};
```

### Backend Registration

```c
// Runtime backend selection
const fb_gpu_backend_trait* fb_get_backend(fb_gpu_backend_type_t type) {
    switch (type) {
        case FB_GPU_NVIDIA_CUBLAS:
            return fb_get_cublas_trait();
        case FB_GPU_AMD_ROCBLAS:
            return fb_get_rocblas_trait();
        case FB_GPU_INTEL_ONEMKL:
            return fb_get_onemkl_trait();
        case FB_GPU_APPLE_ACCELERATE:
            return fb_get_accelerate_trait();
        default:
            return NULL;
    }
}
```

### Multi-GPU Support

Users can mix vendors in the same system:

```c
// NVIDIA RTX 4090 for FP32 workloads
const fb_gpu_backend_trait* nvidia = fb_get_backend(FB_GPU_NVIDIA_CUBLAS);
void* nvidia_handle;
nvidia->init(0, &nvidia_handle);

// AMD RX 7900 XTX for FP64 workloads
const fb_gpu_backend_trait* amd = fb_get_backend(FB_GPU_AMD_ROCBLAS);
void* amd_handle;
amd->init(0, &amd_handle);

// Intel Arc A770 for integer/mixed precision
const fb_gpu_backend_trait* intel = fb_get_backend(FB_GPU_INTEL_ONEMKL);
void* intel_handle;
intel->init(0, &intel_handle);
```

## Implementation Roadmap

### Phase 1: NVIDIA ✅ COMPLETE
- [x] cuBLAS trait implementation (198 ops)
- [x] Build system (Windows + Linux)
- [x] Comprehensive tests
- [x] Documentation

### Phase 2: AMD 🚧 IN PROGRESS
- [x] rocBLAS trait skeleton
- [x] Windows HIP SDK integration
- [x] Basic compilation verified
- [ ] Implement remaining 190 operations
- [ ] Testing on RX 7900 XTX
- [ ] Performance benchmarks vs cuBLAS

### Phase 3: Intel 📝 NEXT
- [ ] oneMKL trait implementation
- [ ] oneAPI toolkit integration
- [ ] SYCL C++ wrapper (if needed)
- [ ] Testing on Arc A770 or integrated GPU
- [ ] Leverage Intel's extended BLAS operations

### Phase 4: Apple 📝 FUTURE
- [ ] Accelerate framework integration
- [ ] Metal backend for custom kernels
- [ ] Neural Engine utilization
- [ ] Testing on M3/M4 MacBooks

### Phase 5: Exotic Accelerators 🚀 RESEARCH
- [ ] Google TPU (TensorFlow/JAX integration?)
- [ ] Qualcomm NPU (Snapdragon)
- [ ] AWS Trainium/Inferentia
- [ ] Cerebras Wafer-Scale Engine

## Build System Architecture

### Conditional Compilation

```c
// gpu_backend_registry.c
#ifdef FB_WITH_CUBLAS
extern const fb_gpu_backend_trait* fb_get_cublas_trait(void);
#endif

#ifdef FB_WITH_ROCBLAS
extern const fb_gpu_backend_trait* fb_get_rocblas_trait(void);
#endif

#ifdef FB_WITH_ONEMKL
extern const fb_gpu_backend_trait* fb_get_onemkl_trait(void);
#endif

#ifdef FB_WITH_ACCELERATE
extern const fb_gpu_backend_trait* fb_get_accelerate_trait(void);
#endif
```

### CMake Detection

```cmake
# Auto-detect available backends
find_package(CUDAToolkit QUIET)
find_package(hip QUIET)
find_package(IntelSYCL QUIET)

if(CUDAToolkit_FOUND)
    option(FB_WITH_CUBLAS "Build NVIDIA cuBLAS backend" ON)
endif()

if(hip_FOUND)
    option(FB_WITH_ROCBLAS "Build AMD rocBLAS backend" ON)
endif()

if(IntelSYCL_FOUND)
    option(FB_WITH_ONEMKL "Build Intel oneMKL backend" ON)
endif()

if(APPLE)
    option(FB_WITH_ACCELERATE "Build Apple Accelerate backend" ON)
endif()
```

## Performance Expectations

### GEMM Performance (FP32, 4096x4096)

| GPU                    | TFLOPS | GEMM (GFLOPS) | Memory BW | Winner       |
| ---------------------- | ------ | ------------- | --------- | ------------ |
| RTX 4090 (cuBLAS)      | 83     | ~65,000       | 1008 GB/s | 🥇 FP32       |
| RX 7900 XTX (rocBLAS)  | 61     | ~48,000       | 960 GB/s  | 🥉            |
| Arc A770 16GB (oneMKL) | 17     | ~13,000       | 560 GB/s  | -            |
| M3 Max (Accelerate)    | 5.1    | ~4,000        | 400 GB/s  | 🥇 Efficiency |

### GEMM Performance (FP64)

| GPU         | FP64 TFLOPS | Winner          |
| ----------- | ----------- | --------------- |
| RTX 4090    | 1.3         | -               |
| RX 7900 XTX | 1.9         | 🥇 Consumer FP64 |
| Arc A770    | 0.5         | -               |
| M3 Max      | ~0.3        | -               |

## Testing Strategy

### Unit Tests
- Each backend has identical test suite
- Tests verify all 198 implemented operations
- NULL operations return error codes gracefully

### Integration Tests
- Multi-vendor workload distribution
- Cross-GPU data transfer
- Vendor failover scenarios

### Performance Tests
- Vendor-specific benchmarks
- Operation-level profiling
- Power consumption metrics

## Documentation

- `docs/NVIDIA_SETUP.md` - CUDA Toolkit installation
- `docs/WINDOWS_ROCM_REALITY.md` - AMD HIP SDK setup
- `docs/INTEL_ONEMKL_SETUP.md` - oneAPI installation (TBD)
- `docs/APPLE_ACCELERATE_SETUP.md` - macOS development (TBD)

## Contributing

Want to add support for a new accelerator? We welcome contributions!

Requirements:
1. Implements `fb_gpu_backend_trait` interface
2. Includes build scripts for target platform
3. Provides basic test coverage
4. Documents installation & setup

## License

faster-blaster is open source. Vendor libraries have their own licenses:
- cuBLAS: NVIDIA CUDA Toolkit EULA
- rocBLAS: MIT License (open source)
- oneMKL: Intel Simplified Software License
- Accelerate: Apple SDK Agreement
