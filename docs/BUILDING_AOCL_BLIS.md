# Building AOCL-BLIS from Source for Maximum Performance

This guide explains how to build AOCL-BLIS from source to unlock maximum performance for faster-blaster.

## Why Build from Source?

The prebuilt AOCL-BLIS binaries have several limitations:

1. **No Native BLIS API**: Only exports CBLAS compatibility layer (slower)
2. **No Expert Interface**: Cannot use `_ex` variants for runtime threading control
3. **Static Threading**: Thread count fixed at compile time, no runtime control
4. **Generic Optimizations**: Not tuned for your specific CPU microarchitecture

Building from source enables:

- ✅ **Native BLIS Typed API** - Faster than CBLAS layer
- ✅ **Expert Interface** - Runtime threading control per operation
- ✅ **CPU-Specific Optimizations** - Kernel selection for your exact CPU
- ✅ **Full API Access** - All BLIS features available

## Performance Impact

According to BLIS documentation:
- Native API is **5-15% faster** than CBLAS layer for Level 2/3 operations
- Expert interface enables **per-operation threading** instead of global settings
- CPU-specific builds can be **20-50% faster** than generic builds

## Prerequisites

### Windows

```powershell
# Install required tools via winget or chocolatey
winget install Git.Git
winget install Python.Python.3.12
winget install Kitware.CMake

# Install MSYS2 for make and compiler tools
winget install MSYS2.MSYS2

# In MSYS2 terminal:
pacman -S make mingw-w64-x86_64-gcc mingw-w64-x86_64-toolchain
```

### Linux

```bash
sudo apt-get install build-essential python3 git cmake
```

### macOS

```bash
brew install python3 git cmake
xcode-select --install
```

## Automated Build (Recommended)

### Windows

```powershell
# Run the automated build script
.\scripts\build_aocl_blis.ps1

# Custom install location
.\scripts\build_aocl_blis.ps1 -InstallDir "C:\MyLibs\BLIS"

# View help
Get-Help .\scripts\build_aocl_blis.ps1 -Detailed
```

### Linux/macOS

```bash
# Coming soon - use manual build below
```

## Manual Build

### 1. Clone AOCL-BLIS Repository

```bash
git clone --branch amd-mainline https://github.com/amd/blis.git
cd blis
```

### 2. Detect Your CPU Architecture

**AMD CPUs:**
- Ryzen 7000/9000 series → `zen4`
- Ryzen 5000/6000 series → `zen3`
- Ryzen 3000/4000 series → `zen2`
- EPYC Milan/Genoa → `zen3` or `zen4`

**Intel CPUs:**
- Alder Lake/Raptor Lake → `haswell` (generic, not optimal)
- Consider using Intel MKL instead for better Intel performance

Check your CPU:
```powershell
# Windows
Get-WmiObject -Class Win32_Processor | Select-Object Name

# Linux
lscpu | grep "Model name"
```

### 3. Configure BLIS

```bash
# AMD Zen 4 (Ryzen 7000+)
./configure --prefix=/opt/blis \
            --enable-cblas \
            --enable-threading=openmp \
            --enable-shared \
            --enable-static \
            zen4

# AMD Zen 3 (Ryzen 5000)
./configure --prefix=/opt/blis \
            --enable-cblas \
            --enable-threading=openmp \
            --enable-shared \
            --enable-static \
            zen3

# Intel (generic fallback)
./configure --prefix=/opt/blis \
            --enable-cblas \
            --enable-threading=openmp \
            --enable-shared \
            --enable-static \
            haswell
```

**Key Configuration Options:**

- `--enable-cblas`: Build CBLAS compatibility layer (optional, for compatibility)
- `--enable-threading=openmp`: Enable OpenMP threading for parallelism
- `--enable-shared`: Build shared library (.dll/.so)
- `--enable-static`: Build static library (.lib/.a)
- Last argument: CPU architecture configuration

### 4. Build

```bash
# Use all available cores
make -j$(nproc)

# Or specify core count manually
make -j8
```

### 5. Install

```bash
# System-wide install (requires sudo on Linux)
sudo make install

# Or install to custom directory (already set with --prefix)
make install
```

### 6. Verify Installation

```bash
# Check installed files
ls -la /opt/blis/lib
ls -la /opt/blis/include

# Test linking
gcc test.c -I/opt/blis/include -L/opt/blis/lib -lblis -o test
```

## Integration with faster-blaster

### Update CMakeLists.txt

After building BLIS, update your faster-blaster build to find the custom BLIS:

```cmake
# Set custom BLIS location
set(BLIS_ROOT "C:/AOCL-BLIS-Custom" CACHE PATH "Custom BLIS installation")

# Find BLIS library
find_library(BLIS_LIBRARY
    NAMES blis libblis
    PATHS ${BLIS_ROOT}/lib
    NO_DEFAULT_PATH
)

find_path(BLIS_INCLUDE_DIR
    NAMES blis.h
    PATHS ${BLIS_ROOT}/include/blis
    NO_DEFAULT_PATH
)
```

### Rebuild faster-blaster

```powershell
cd faster-blaster/build
cmake .. -DBLIS_ROOT="C:/AOCL-BLIS-Custom"
cmake --build . --config Release
```

## Runtime Configuration

### Thread Control

With source-built BLIS, you can control threading at runtime:

```c
// Set global thread count (traditional method)
export BLIS_NUM_THREADS=8
export OMP_NUM_THREADS=8

// Or use API (faster-blaster will support this)
fb_set_num_threads(8);
```

### Performance Testing

```powershell
# Run benchmarks with different thread counts
$env:BLIS_NUM_THREADS=1; .\benchmark.exe
$env:BLIS_NUM_THREADS=4; .\benchmark.exe
$env:BLIS_NUM_THREADS=8; .\benchmark.exe
```

## Troubleshooting

### Configure Fails

**Error**: `configure: error: configuration not found`
- Check CPU detection output
- Try generic configuration: `haswell`, `zen`, or `generic`

### Build Fails on Windows

**Error**: `make: command not found`
- Install MSYS2 and add to PATH
- Or use WSL (Windows Subsystem for Linux)

**Error**: Compiler not found
- Install MinGW-w64 via MSYS2: `pacman -S mingw-w64-x86_64-gcc`
- Or use Visual Studio with nmake

### Runtime Errors

**Error**: DLL not found
- Add BLIS bin directory to PATH: `$env:PATH += ";C:\AOCL-BLIS-Custom\bin"`
- Or copy DLLs to application directory

**Error**: Symbol not found
- Verify BLIS was built with `--enable-cblas`
- Check that faster-blaster is linking against correct BLIS version

## Advanced Options

### Link-Time Optimization (LTO)

```bash
./configure --enable-lto zen4
```

### Static Linking

For distributable binaries without external dependencies:

```cmake
target_link_libraries(faster-blaster PRIVATE libblis.a)
```

### Debug Build

```bash
./configure --enable-debug zen4
make -j$(nproc)
```

## Verification

After building and linking, verify BLIS features are available:

```c
#include <blis.h>

int main() {
    // Check for expert interface
    void* test_ex = bli_sgemm_ex;
    printf("Expert interface available: %s\n", test_ex ? "YES" : "NO");
    
    // Check threading
    bli_thread_set_num_threads(4);
    int threads = bli_thread_get_num_threads();
    printf("Thread control working: %s (%d threads)\n", 
           threads == 4 ? "YES" : "NO", threads);
    
    return 0;
}
```

## Next Steps

Once BLIS is built and installed:

1. ✅ BLIS built from source with optimal configuration
2. ⏭️ Update faster-blaster to use native BLIS API (not CBLAS)
3. ⏭️ Implement expert interface wrappers for threading control
4. ⏭️ Expose threading API to faster-blaster users
5. ⏭️ Benchmark performance improvements

See [BLIS_BACKEND_DESIGN.md](BLIS_BACKEND_DESIGN.md) for implementation details.
