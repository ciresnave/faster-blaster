# Building Faster-BLASTER

**Last Updated**: December 14, 2025  
**Build System**: CMake 3.15+  
**Architecture**: Plugin-based backend selection

## Quick Start

```bash
git clone https://github.com/yourusername/faster-blaster.git
cd faster-blaster
mkdir build && cd build
cmake ..
cmake --build . --config Release
ctest --output-on-failure  # Run tests
```

---

## Prerequisites

### Required Dependencies

1. **C11 Compiler**
   - GCC 4.9+ or Clang 3.5+ (Linux/macOS)
   - MSVC 2019+ (Windows)

2. **CMake 3.15+**
   - Download from [cmake.org](https://cmake.org/download/)

3. **OpenCL SDK** (for GPU detection)
   
   **Linux:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install opencl-headers ocl-icd-opencl-dev
   
   # Fedora/RHEL
   sudo dnf install opencl-headers ocl-icd-devel
   
   # Arch Linux
   sudo pacman -S opencl-headers ocl-icd
   ```
   
   **Windows:**
   - Install Intel OpenCL SDK or
   - NVIDIA CUDA Toolkit (includes OpenCL) or
   - AMD APP SDK
   
   **macOS:**
   - OpenCL is included in the system (no installation needed)

4. **cpuinfo** (automatically fetched by CMake if not installed)

### Optional Backend Dependencies

- **OpenBLAS**: Will be bundled/built automatically
- **BLIS**: Will be bundled/built automatically
- **Intel MKL**: Download from [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- **NVIDIA cuBLAS**: Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- **AMD rocBLAS**: Install [ROCm](https://rocm.docs.amd.com/)

## Building

### Basic Build

```bash
git clone https://github.com/yourusername/faster-blaster.git
cd faster-blaster
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Install System-Wide

```bash
sudo make install
```

This installs:
- Headers to `/usr/local/include/`
- Library to `/usr/local/lib/`
- CMake config to `/usr/local/lib/cmake/faster-blaster/`

### CMake Build Options

```bash
cmake \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=ON \
  -DFB_ENABLE_CUDA=AUTO \
  -DFB_ENABLE_ROCM=AUTO \
  -DFB_ENABLE_ONEMKL=AUTO \
  -DFB_ENABLE_METAL=AUTO \
  ..
```

**Core Build Options:**
- `BUILD_SHARED_LIBS` - Build shared library instead of static (default: ON)
- `BUILD_TESTS` - Build test suite (default: ON)
- `BUILD_BENCHMARKS` - Build benchmarks (default: ON)
- `BUILD_EXAMPLES` - Build example programs (default: ON)
- `CMAKE_BUILD_TYPE` - Debug, Release, RelWithDebInfo, MinSizeRel (default: Release)

**Plugin Backend Options:**

All plugin backends support three modes:
- `ON` - Force enable (fail if dependencies not found)
- `OFF` - Disable completely
- `AUTO` - Enable if dependencies detected (recommended)

| Option             | Backend            | Default | Platform      |
| ------------------ | ------------------ | ------- | ------------- |
| `FB_ENABLE_CUDA`   | NVIDIA cuBLAS      | AUTO    | Windows/Linux |
| `FB_ENABLE_ROCM`   | AMD rocBLAS        | AUTO    | Linux         |
| `FB_ENABLE_ONEMKL` | Intel oneMKL (GPU) | AUTO    | Windows/Linux |
| `FB_ENABLE_METAL`  | Apple Metal        | AUTO    | macOS only    |

**CPU Backend Plugins** (Always built):
- AMD AOCL BLIS (scores 95 on AMD CPUs, 70 on others)
- Standard BLIS (scores 85, portable)
- OpenBLAS (scores 80, portable)
- Intel MKL (scores 95 on Intel CPUs, 70 on others)
- Apple Accelerate (scores 98 on macOS, 0 on other platforms)

**Note**: Plugin system automatically selects the best backend at runtime based on hardware detection. All enabled plugins are built; selection happens at runtime, not compile-time.

### Windows Build (MSVC)

```powershell
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
cmake --install .
```

### Linux Build (Ubuntu/Debian)

**Basic build (CPU only)**:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake git

git clone https://github.com/yourusername/faster-blaster.git
cd faster-blaster
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

**With NVIDIA GPU support**:
```bash
# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Build with CUDA enabled
mkdir build && cd build
cmake -DFB_ENABLE_CUDA=ON ..
make -j$(nproc)
```

**With AMD GPU support (ROCm)**:
```bash
# Install ROCm (Ubuntu 22.04)
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_22.20.50200-1_all.deb
sudo apt-get install ./amdgpu-install_*_all.deb
sudo amdgpu-install --usecase=rocm

# Build with ROCm enabled
mkdir build && cd build
cmake -DFB_ENABLE_ROCM=ON ..
make -j$(nproc)
```

**With Intel GPU support (oneAPI)**:
```bash
# Install Intel oneAPI Base Toolkit
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/20f4e6a1-6b0b-4752-b8c1-e5eacba10e01/l_BaseKit_p_2024.0.0.49564.sh
sudo sh ./l_BaseKit_p_2024.0.0.49564.sh

# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build with oneMKL enabled
mkdir build && cd build
cmake -DFB_ENABLE_ONEMKL=ON ..
make -j$(nproc)
```

### macOS Build

**Basic build (CPU + Accelerate framework)**:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake
brew install cmake

# Build
git clone https://github.com/yourusername/faster-blaster.git
cd faster-blaster
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
sudo make install
```

**Apple Silicon (M1/M2/M3) - Metal GPU support**:
```bash
mkdir build && cd build
cmake -DFB_ENABLE_METAL=ON ..
make -j$(sysctl -n hw.ncpu)
```

**Intel Mac with discrete AMD GPU**:
```bash
mkdir build && cd build
cmake -DFB_ENABLE_METAL=ON ..  # Metal works on Intel Macs too
make -j$(sysctl -n hw.ncpu)
```

---

## Testing Plugin Detection

After building, verify which plugins are available:

**Windows (PowerShell)**:
```powershell
cd build\tests\Release
.\test_plugin_architecture.exe
```

**Linux/macOS**:
```bash
cd build/tests
./test_plugin_architecture
```

**Expected output**:
```
Plugin Architecture Test
========================

Registered Plugins:
  1. metal v3.0 (Apple Inc.) - Capabilities: 0x000002DE
  2. onemkl v2024.0 (Intel Corporation) - Capabilities: 0x000002DE
  3. rocblas v6.0 (Advanced Micro Devices (AMD)) - Capabilities: 0x000002DE
  4. cublas v12.0 (NVIDIA Corporation) - Capabilities: 0x000002DE
  5. accelerate v1.0 (Apple Inc.) - Capabilities: 0x000002DD
  6. mkl v2024.0 (Intel Corporation) - Capabilities: 0x000002DD
  7. openblas v0.3.27 (OpenBLAS Project) - Capabilities: 0x0000025D
  8. standard-blis v0.9.0 (BLIS Project) - Capabilities: 0x00000245
  9. aocl-blis v4.2.1 (AMD) - Capabilities: 0x0000025D

Total plugins registered: 9

[DEBUG] Probed aocl-blis: score=95 (AMD CPU detected)
[DEBUG] Initializing best plugin: aocl-blis (score=95)

Plugin architecture test PASSED
```

The system will automatically select the best backend for your hardware:
- **AMD Ryzen CPU** → AOCL BLIS (score=95)
- **Intel Core CPU** → Intel MKL (score=95)
- **Apple Silicon** → Metal GPU (score=99) or Accelerate (score=98)
- **NVIDIA GPU** → cuBLAS (score=95)
- **AMD GPU** → rocBLAS (score=95)
- **Intel GPU** → oneMKL (score=95)

---

## Verifying Installation

### Run Self-Test

```bash
cd build
ctest --output-on-failure
```

### Run Example

```bash
./examples/basic_example
```

Expected output:
```
=== Faster-BLASTER Basic Example ===

Initializing Faster-BLASTER...
Detected: Intel Core i9-13900K (24 cores)
Detected: NVIDIA GeForce RTX 4090
Loading calibration data... Done!
Library version: 0.1.0
Initialization complete!

Example 1: Matrix Multiplication (SGEMM)
...
```

## Troubleshooting

### OpenCL Not Found

**Error:** `Could not find OpenCL`

**Linux:**
```bash
# Check if OpenCL is installed
ls /usr/include/CL/cl.h
ls /usr/lib/x86_64-linux-gnu/libOpenCL.so

# If missing, install ICD loader
sudo apt-get install ocl-icd-opencl-dev
```

**Windows:**
Install one of:
- [Intel OpenCL Runtime](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html)
- NVIDIA CUDA Toolkit
- AMD Radeon Software

### cpuinfo Build Error

If cpuinfo fails to fetch automatically:

```bash
cd build
git clone https://github.com/pytorch/cpuinfo.git
cd cpuinfo
cmake -DCPUINFO_BUILD_TOOLS=OFF -DCPUINFO_BUILD_UNIT_TESTS=OFF .
make
sudo make install
```

Then rebuild faster-blaster.

### MKL Not Found

Set environment variable:

**Linux/macOS:**
```bash
export MKLROOT=/opt/intel/oneapi/mkl/latest
```

**Windows:**
```powershell
$env:MKLROOT="C:\Program Files (x86)\Intel\oneAPI\mkl\latest"
```

### CUDA Not Found

Add CUDA to PATH:

**Linux:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Windows:**
```powershell
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
```

## Using in Your Project

### With CMake

```cmake
find_package(faster-blaster REQUIRED)
target_link_libraries(your_target PRIVATE FasterBlaster::faster-blaster)
```

### With pkg-config

```bash
gcc myprogram.c $(pkg-config --cflags --libs faster-blaster) -o myprogram
```

### Manual Linking

```bash
gcc myprogram.c -I/usr/local/include -L/usr/local/lib -lfaster-blaster -o myprogram
```

## Cross-Compilation

### For ARM64 (Raspberry Pi, etc.)

```bash
cmake \
  -DCMAKE_TOOLCHAIN_FILE=cmake/arm64-toolchain.cmake \
  -DENABLE_CUBLAS=OFF \
  -DENABLE_MKL=OFF \
  ..
```

### For Android

Use Android NDK:

```bash
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  ..
```

## Development Build

For development with debug symbols:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ..
make -j$(nproc)
```

Run tests with Valgrind:

```bash
ctest -T memcheck
```

Enable compiler sanitizers:

```bash
cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined" \
  ..
```
