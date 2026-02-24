# Quick Reference: Building Backends From Source

## TL;DR - Get Maximum Performance Now

### Windows
```powershell
.\configure_optimized.ps1
cd build
cmake --build . --config Release -j
```

### Linux/macOS
```bash
./configure_optimized.sh
cd build
cmake --build . -j$(nproc)
```

## What Gets Built

| Backend      | Purpose         | Optimization              | Performance Gain |
| ------------ | --------------- | ------------------------- | ---------------- |
| **BLIS**     | CPU BLAS        | Zen4/Zen3/Skylake kernels | +10-25%          |
| **OpenBLAS** | CPU BLAS+LAPACK | ZEN/SKYLAKEX targets      | +15-30%          |
| **CLBlast**  | GPU OpenCL      | Any OpenCL device         | Native GPU speed |

## Configuration Options

```powershell
# Build everything (recommended)
.\configure_optimized.ps1

# Skip specific backends
.\configure_optimized.ps1 -SkipCLBlast      # CPU only
.\configure_optimized.ps1 -SkipBLIS         # No BLIS
.\configure_optimized.ps1 -SkipOpenBLAS     # No OpenBLAS

# Clean build
.\configure_optimized.ps1 -Clean

# Help
.\configure_optimized.ps1 -Help
```

## Build Time

| Backend   | Cores   | Time           |
| --------- | ------- | -------------- |
| BLIS      | 8 cores | ~3-5 min       |
| OpenBLAS  | 8 cores | ~5-8 min       |
| CLBlast   | 8 cores | ~2-3 min       |
| **Total** | 8 cores | **~10-15 min** |

## Hardware Support

### AMD CPUs
- ✅ Ryzen 7000/9000 (Zen 4) - AVX512 kernels
- ✅ Ryzen 5000/6000 (Zen 3) - AVX2 kernels  
- ✅ Ryzen 3000/4000 (Zen 2) - AVX2 kernels
- ✅ EPYC Milan/Genoa - Server optimizations

### Intel CPUs
- ✅ 12th/13th/14th Gen Core (Alder/Raptor Lake)
- ✅ Xeon Scalable (Skylake/Cascade/Ice Lake)
- ✅ Xeon 4th/5th Gen (Sapphire Rapids)

### GPUs
- ✅ NVIDIA (any with CUDA support)
- ✅ AMD Radeon (RDNA, RDNA2, RDNA3)
- ✅ AMD Instinct (CDNA2, CDNA3)
- ✅ Intel Arc/Iris (via OpenCL)
- ✅ Apple M1/M2/M3 (Metal - separate)

## Verify It Worked

```bash
# Check built libraries
ls build/backends-install/blis/lib/
ls build/backends-install/openblas/lib/
ls build/backends-install/clblast/lib/

# Run benchmarks
cd build
./benchmarks/benchmark_gemm --backend=blis --size=2048
```

## Disable Source Build

If you want to use system libraries instead:

```bash
cmake .. -DFB_BUILD_BACKENDS_FROM_SOURCE=OFF
```

## Common Issues

**"WSL not found"** (Windows)
```powershell
wsl --install
# Restart, then retry
```

**"make not found"** (Linux)
```bash
sudo apt-get install build-essential  # Ubuntu/Debian
sudo dnf install make gcc gcc-c++      # Fedora
```

**"OpenCL not found"**
```bash
# Install GPU drivers first, then:
sudo apt-get install opencl-headers ocl-icd-opencl-dev  # Ubuntu
sudo dnf install opencl-headers ocl-icd-devel           # Fedora
```

## Full Documentation

- [Complete Build Guide](BUILDING_FROM_SOURCE.md)
- [BLIS Build Details](BUILDING_AOCL_BLIS.md)
- [BLIS Backend Design](BLIS_BACKEND_DESIGN.md)
