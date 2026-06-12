# Backend Installation Guide

faster-blaster includes an **automatic backend detection and installation system** to help you get the best performance for your hardware.

## Automatic Detection

When you configure faster-blaster with CMake, it automatically:

1. **Detects your hardware** (CPU vendor, NVIDIA GPU, AMD GPU, OpenCL devices)
2. **Checks installed backends** (OpenBLAS, Intel MKL, AOCL, CUDA, ROCm, CLBlast)
3. **Recommends missing backends** with installation instructions
4. **Optionally builds optimized backends** from source

## Quick Start

### Default (Fully Automatic)

```bash
cmake -B build
cmake --build build
```

This will:
- ✅ Detect all hardware
- ✅ Show installation prompts for missing backends (defaults to YES)
- ✅ Build OpenBLAS from source with threading support
- ✅ Configure all available backends

### Silent Mode (No Prompts)

```bash
cmake -B build -DINTERACTIVE_BACKEND_SETUP=OFF
cmake --build build
```

### Custom Configuration

```bash
cmake -B build \
  -DBUILD_OPENBLAS_FROM_SOURCE=ON \
  -DINTERACTIVE_BACKEND_SETUP=ON \
  -DAUTO_DETECT_BACKENDS=ON
```

## Backend Options

### CPU Backends

#### OpenBLAS (Universal)
- **Status**: Auto-built from source by default
- **Features**: Multi-threaded, dynamic architecture
- **Performance**: Good baseline performance on all CPUs
- **Installation**: Automatic (BUILD_OPENBLAS_FROM_SOURCE=ON)

**Why build from source?**
- vcpkg's OpenBLAS is single-threaded (4x slower)
- Source build enables OpenMP threading
- Automatic optimization for your CPU

#### Intel MKL (Intel CPUs)
- **Status**: Detected if installed
- **Features**: Optimized for Intel CPUs (2-4x faster than generic)
- **Installation**:
  ```bash
  # Option 1: Intel oneAPI (free)
  # Download: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
  
  # Option 2: vcpkg
  vcpkg install intel-mkl
  ```

#### AMD AOCL (AMD CPUs)
- **Status**: Detected if installed
- **Features**: Optimized for AMD CPUs (2-4x faster than generic)
- **Installation**:
  ```bash
  # Download from: https://www.amd.com/en/developer/aocl.html
  # Windows: Run installer
  # Linux: Extract to /opt/AMD/aocl
  ```

### GPU Backends

#### cuBLAS (NVIDIA GPUs)
- **Status**: Detected if CUDA Toolkit installed
- **Features**: Fastest for NVIDIA GPUs (10-100x faster than CPU)
- **Installation**:
  ```bash
  # Download: https://developer.nvidia.com/cuda-downloads
  # Recommended: CUDA 12.0 or later
  ```

#### rocBLAS (AMD Discrete GPUs)
- **Status**: Detected if ROCm installed
- **Features**: Fast for AMD discrete GPUs (RX 6000/7000, Instinct MI)
- **⚠️ Note**: Does NOT support AMD integrated GPUs (use CLBlast instead)
- **Installation**:
  ```bash
  # Download: https://www.amd.com/en/graphics/servers-solutions-rocm
  # Supported: Radeon RX 6000/7000, Radeon Pro, Instinct MI series
  ```

#### CLBlast (All GPUs via OpenCL)
- **Status**: Detected if installed
- **Features**: Universal GPU support via OpenCL
- **Works with**: NVIDIA, AMD (integrated & discrete), Intel, Apple
- **Perfect for**: AMD Radeon 610M and other integrated GPUs
- **Installation**:
  ```bash
  # Option 1: vcpkg (recommended)
  vcpkg install clblast opencl
  
  # Option 2: Build from source
  git clone https://github.com/CNugteren/CLBlast
  cd CLBlast && mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
  cmake --build . --target install
  ```

## Hardware-Specific Recommendations

### Intel CPU System
```
Best setup:
  CPU: Intel MKL (primary) + OpenBLAS (fallback)
  GPU: cuBLAS (NVIDIA) or CLBlast (Intel/other)
```

### AMD CPU System  
```
Best setup:
  CPU: AMD AOCL (primary) + OpenBLAS (fallback)
  GPU: rocBLAS (discrete AMD) or CLBlast (integrated AMD/other)
```

### Laptop with Integrated GPU
```
Example: AMD Ryzen with Radeon 610M
  CPU: OpenBLAS (multi-threaded from source)
  GPU: CLBlast (works with integrated GPUs)
  
Note: ROCm does NOT support integrated GPUs!
```

### Multi-GPU Workstation
```
Example: Intel CPU + NVIDIA RTX 4070 + AMD Radeon 610M
  CPU: Intel MKL (primary) + OpenBLAS (fallback)
  GPU 0 (NVIDIA): cuBLAS (fastest)
  GPU 1 (AMD): CLBlast (via OpenCL)
```

## Build Options Reference

| Option                       | Default | Description               |
| ---------------------------- | ------- | ------------------------- |
| `AUTO_DETECT_BACKENDS`       | ON      | Enable hardware detection |
| `INTERACTIVE_BACKEND_SETUP`  | ON      | Show installation prompts |
| `BUILD_OPENBLAS_FROM_SOURCE` | ON      | Build threaded OpenBLAS   |
| `ENABLE_MKL`                 | ON      | Enable Intel MKL if found |
| `ENABLE_OPENBLAS`            | ON      | Enable OpenBLAS if found  |
| `ENABLE_AOCL`                | ON      | Enable AMD AOCL if found  |
| `FB_ENABLE_CUDA`             | AUTO    | Enable NVIDIA cuBLAS      |
| `FB_ENABLE_ROCM`             | AUTO    | Enable AMD rocBLAS        |

## Troubleshooting

### "OpenBLAS found but single-threaded"
**Solution**: Enable `BUILD_OPENBLAS_FROM_SOURCE=ON` (default)
```bash
cmake -B build -DBUILD_OPENBLAS_FROM_SOURCE=ON
```

### "AMD GPU not detected"
**Check**:
1. Is it an integrated GPU? (Like Radeon 610M)
   - ROCm doesn't support it → Use CLBlast instead
2. Is OpenCL installed?
   ```bash
   # Install OpenCL
   vcpkg install opencl
   ```

### "CUDA found but cuBLAS tests fail"
**Solution**: This is expected with MSVC generator. cuBLAS runtime works fine.

### "Build takes forever"
**Cause**: Building OpenBLAS from source takes 5-15 minutes
**Solutions**:
- Accept vcpkg's single-threaded version: `-DBUILD_OPENBLAS_FROM_SOURCE=OFF`
- Use pre-built MKL or AOCL instead

## Performance Comparison

Typical GEMM (matrix multiply) performance on an AMD Ryzen 9 7940HX:

| Backend                  | Threads | GFLOPS | Relative        |
| ------------------------ | ------- | ------ | --------------- |
| vcpkg OpenBLAS           | 1       | 45     | 1.0x (baseline) |
| OpenBLAS (source)        | 16      | 180    | 4.0x            |
| AMD AOCL                 | 16      | 280    | 6.2x            |
| NVIDIA RTX 4070 (cuBLAS) | GPU     | 7396   | 164x            |
| AMD 610M (CLBlast)       | GPU     | 150    | 3.3x            |

## Verification

After building, verify backends loaded correctly:

```bash
# Run device test
./build/tests/Release/test_device_api.exe

# Expected output:
# [INFO] Successfully loaded backend: openblas (version 0.3.27)
# [OpenBLAS] Threading capability verified: 16 threads
# ✅ All tests passed
```

## Getting Help

If automatic detection fails:
1. Check CMake output for detection results
2. Manually specify backend paths:
   ```bash
   cmake -B build \
     -DOpenBLAS_DIR=/custom/path/lib/cmake/openblas \
     -DMKL_ROOT=/opt/intel/oneapi/mkl
   ```
3. Report issues with full CMake output

---

**Last Updated**: December 2025
**faster-blaster Version**: 0.1.0
