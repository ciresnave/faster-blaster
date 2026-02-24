# Faster-Blaster Build Guide

This guide explains how to build faster-blaster with various backend configurations.

## Quick Start

### Default Build (All Backends - Recommended)
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

**This automatically detects and enables all available backends!** The build system will:
- Detect CUDA → Enable cuBLAS (NVIDIA GPUs)
- Detect ROCm → Enable rocBLAS (AMD GPUs)  
- Detect Intel MKL → Enable MKL (Intel CPUs)
- Detect OpenBLAS → Enable OpenBLAS (Portable CPUs)
- Detect Apple Accelerate → Enable Accelerate (macOS)
- Always include Reference backend (fallback)

Backends that aren't found will be automatically skipped with a warning.

### Minimal Build (Reference Backend Only)
```bash
mkdir build && cd build
cmake -DENABLE_CUBLAS=OFF -DENABLE_ROCBLAS=OFF -DENABLE_MKL=OFF -DENABLE_OPENBLAS=OFF -DENABLE_ACCELERATE=OFF ..
cmake --build .
```

This builds with only the reference backend - good for testing but not optimized.

## Backend Options

**Note**: All backends are **enabled by default**. The build system automatically detects which ones are available on your system. You only need to use these options to **disable** backends you don't want.

### Disabling Specific Backends

To disable a backend you don't want:
```bash
cmake -DENABLE_CUBLAS=OFF ..      # Disable NVIDIA cuBLAS
cmake -DENABLE_MKL=OFF ..          # Disable Intel MKL
cmake -DENABLE_OPENBLAS=OFF ..     # Disable OpenBLAS
```

### GPU Backends

#### NVIDIA cuBLAS (Enabled by default if CUDA found)
**Requirements**: CUDA Toolkit 11.0 or later

**Auto-Detection**: Automatically finds CUDA in standard locations:
- Linux: `/usr/local/cuda`, `/opt/cuda`
- Windows: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`

**Manual CUDA Path** (if auto-detection fails):
```bash
cmake -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda ..
```

**Disable if not needed**:
```bash
cmake -DENABLE_CUBLAS=OFF ..
```

#### AMD rocBLAS (Enabled by default if ROCm found)
**Requirements**: ROCm 5.0 or later

**Auto-Detection**: Looks for ROCm in:
- Linux: `/opt/rocm`, `/opt/rocm-5.x`

**Disable if not needed**:
```bash
cmake -DENABLE_ROCBLAS=OFF ..
```

#### Intel oneMKL GPU (Enabled by default if oneAPI found)
**Requirements**: Intel oneAPI Toolkit

**Disable if not needed**:
```bash
cmake -DENABLE_ONEMKL_GPU=OFF ..
```

### CPU Backends

#### Intel MKL (Enabled by default if MKL found)
**Requirements**: Intel MKL or oneAPI Base Toolkit

**Auto-Detection**: Searches for MKL in:
- Linux: `/opt/intel/oneapi/mkl/latest`
- Windows: `C:\Program Files (x86)\Intel\oneAPI\mkl\latest`

**Manual MKL Path** (if auto-detection fails):
```bash
cmake -DMKLROOT=/path/to/mkl ..
```

**Disable if not needed**:
```bash
cmake -DENABLE_MKL=OFF ..
```

**Features**:
- AVX-512 optimization
- Multi-threading (automatic)
- VML (Vector Math Library)

#### OpenBLAS (Enabled by default if found)
**Requirements**: OpenBLAS library

**Auto-Detection**: Automatically found if installed via package manager.

**Installation** (if not already installed):
- Ubuntu/Debian: `sudo apt install libopenblas-dev`
- Fedora/RHEL: `sudo dnf install openblas-devel`
- macOS: `brew install openblas`
- Windows: Download from https://github.com/xianyi/OpenBLAS/releases

**Features**:
- Multi-architecture support (x86, ARM, RISC-V)
- Automatic CPU detection
- Multi-threading

**Disable if not needed**:
```bash
cmake -DENABLE_OPENBLAS=OFF ..
```

#### Apple Accelerate (Enabled by default on macOS)
**Requirements**: macOS 10.13+ (automatically available)

**Auto-Detection**: Automatically enabled on macOS systems.

**Disable if not needed**:
```bash
cmake -DENABLE_ACCELERATE=OFF ..
```

**Features**:
- Optimized for Apple Silicon (M1/M2/M3)
- Also works on Intel Mac
- No installation needed

#### AMD BLIS (Optimized for AMD CPUs)
**Requirements**: BLIS library

```bash
cmake -DENABLE_BLIS=ON ..
```

**Installation**:
```bash
# From source
git clone https://github.com/amd/blis.git
cd blis
./configure --enable-cblas auto
make -j
sudo make install
```

#### AMD AOCL (AMD's Official Optimized Libraries)
**Requirements**: AMD AOCL package

```bash
cmake -DENABLE_AOCL=ON ..
```

## Complete Build Examples

### Default Build (Recommended - Just Works™)
```bash
mkdir build && cd build
cmake ..
cmake --build .
```
**Result**: Automatically detects and enables all available backends. Perfect for most users!

### Minimal Build (CPU Only, No GPU Support)
```bash
cmake -DENABLE_CUBLAS=OFF -DENABLE_ROCBLAS=OFF -DENABLE_ONEMKL_GPU=OFF ..
```
**Result**: Only CPU backends (MKL/OpenBLAS/Accelerate + Reference)

### GPU Only Build (No CPU Optimization)
```bash
cmake -DENABLE_MKL=OFF -DENABLE_OPENBLAS=OFF -DENABLE_ACCELERATE=OFF -DENABLE_BLIS=OFF ..
```
**Result**: Only GPU backends (cuBLAS/rocBLAS) + Reference

### Development Build (Exclude Proprietary Backends)
```bash
cmake -DENABLE_MKL=OFF -DENABLE_CUBLAS=OFF ..
```
**Result**: Only open-source backends (OpenBLAS, BLIS, rocBLAS, Reference)

## Build Configuration

### Debug Build
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Release Build (Optimized)
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### Release with Debug Info
```bash
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

## Testing

### Build Tests
```bash
cmake -DBUILD_TESTS=ON ..
cmake --build .
```

### Run Tests
```bash
# Run all tests
ctest

# Run specific test
./build/test_device_detection
./build/test_dispatch_system
./build/test_end_to_end
```

### Test Output Explanation
- `test_device_detection`: Enumerates all CPUs and GPUs
- `test_dispatch_system`: Tests device selection strategies
- `test_end_to_end`: Full integration test with backend execution

## Troubleshooting

### CMake Can't Find CUDA
```bash
# Set CUDA path manually
export CUDA_HOME=/usr/local/cuda-12
cmake -DENABLE_CUBLAS=ON ..
```

### CMake Can't Find MKL
```bash
# Set MKL root
export MKLROOT=/opt/intel/oneapi/mkl/latest
cmake -DENABLE_MKL=ON ..
```

### CMake Can't Find OpenBLAS
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Or set path manually
cmake -DENABLE_OPENBLAS=ON -DOpenBLAS_DIR=/path/to/openblas ..
```

### Linking Errors with GPU Backends
Make sure runtime libraries are in your library path:

**Linux**:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

**Windows**:
Add to PATH:
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

### Runtime "Backend Not Available" Errors
This means the backend library wasn't found at runtime:

1. **Verify library installation**: Run `ldd` (Linux) or `dumpbin` (Windows) on the executable
2. **Check library path**: Ensure backend .so/.dll files are in LD_LIBRARY_PATH or PATH
3. **Recompile with backend enabled**: Check CMake output shows backend enabled

## Performance Tips

### Multi-Threading
Most backends auto-detect optimal thread count. To set manually:

**Intel MKL**:
```bash
export MKL_NUM_THREADS=8
```

**OpenBLAS**:
```bash
export OPENBLAS_NUM_THREADS=8
```

**BLIS**:
```bash
export BLIS_NUM_THREADS=8
```

### GPU Performance
- Use Release build for optimal performance
- Ensure data is registered with data tracker for transfer optimization
- Use batched operations when possible

## Minimal System Requirements

### For CPU-Only Build
- CMake 3.15+
- C11 compiler (GCC 5+, Clang 3.4+, MSVC 2015+)
- OpenBLAS library (or other CPU BLAS)

### For GPU Build
- Above, plus:
- NVIDIA: CUDA Toolkit 11.0+, GPU with compute capability 3.5+
- AMD: ROCm 5.0+, RDNA or CDNA architecture
- Intel: oneAPI 2023+, Arc or Data Center GPU

## Installation

### System-Wide Install
```bash
sudo cmake --install build
```

This installs:
- Library: `/usr/local/lib/libfaster-blaster.a`
- Headers: `/usr/local/include/faster-blaster/`

### Custom Install Prefix
```bash
cmake -DCMAKE_INSTALL_PREFIX=/opt/faster-blaster ..
sudo cmake --install build
```

## Next Steps

After building:
1. Run tests to verify everything works: `ctest`
2. Check backend configuration: `./build/test_end_to_end`
3. Review performance: `./build/test_benchmarks` (if built)
4. Integrate into your project - see `INTEGRATION.md`

## Getting Help

- Check CMake configuration output for warnings
- Run tests with verbose output: `ctest --verbose`
- Check GPU device detection: `./build/test_device_detection`
- Review backend selection: `./build/test_end_to_end`

## Common Build Recipes

### Research/Development Machine (Default - All Backends)
```bash
cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON \
  ..
```
**Note**: No need to specify backends - all are enabled by default!

### Production Server (Optimize for Specific Hardware)
```bash
# NVIDIA GPU + Intel CPU only
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ROCBLAS=OFF \
  -DENABLE_OPENBLAS=OFF \
  -DENABLE_BLIS=OFF \
  -DBUILD_TESTS=OFF \
  ..
```

### Embedded/ARM Device (CPU Only)
```bash
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUBLAS=OFF \
  -DENABLE_ROCBLAS=OFF \
  -DENABLE_MKL=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  ..
```

### CI/CD (Reference Backend Only)
```bash
cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_CUBLAS=OFF \
  -DENABLE_ROCBLAS=OFF \
  -DENABLE_MKL=OFF \
  -DENABLE_OPENBLAS=OFF \
  -DENABLE_ACCELERATE=OFF \
  -DENABLE_BLIS=OFF \
  -DBUILD_TESTS=ON \
  ..
```
