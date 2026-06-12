# ROCm/HIP Setup for AMD GPU Support

This guide explains how to set up AMD's HIP/ROCm stack to build and use the rocBLAS backend with your AMD Radeon RX 7900 XTX.

## Overview

The `rocblas_trait_impl.c` backend provides **204 BLAS operations** for AMD GPUs - that's **6 more operations** than NVIDIA cuBLAS supports!

**Exclusive rocBLAS operations** (not in cuBLAS):

- `cspmv` / `zspmv` - Complex symmetric packed matrix-vector multiply
- `cspr` / `zspr` - Complex symmetric packed rank-1 update
- `cspr2` / `zspr2` - Complex symmetric packed rank-2 update

## Understanding ROCm vs HIP

**Important Clarification:**

- **ROCm** = AMD's complete platform (like NVIDIA's "CUDA Toolkit")
- **HIP** = Programming API within ROCm (like NVIDIA's "CUDA Runtime API")
- **rocBLAS** = BLAS library within ROCm (like NVIDIA's "cuBLAS")

When AMD documentation refers to "HIP SDK on Windows," it means the **ROCm components needed for HIP development**. Installing HIP means installing the relevant parts of ROCm.

## RX 7900 XTX Support Status (December 2025)

✅ **OFFICIALLY SUPPORTED** on Windows 11!

- **GPU**: AMD Radeon RX 7900 XTX (RDNA3 / gfx1100)
- **ROCm Version**: 7.1.1 components
- **Windows**: Windows 11 22H2+
- **Linux**: Full ROCm 7.1.1 support (Ubuntu 22.04+)

**Important Limitation:**

> AMD states: *"PyTorch on Windows includes ROCm 7.1.1 components; however, the entire ROCm stack is not yet supported on Windows."*

This means:

- ✅ HIP runtime works on Windows
- ✅ rocBLAS library accessible via PyTorch installation
- ⚠️ Standalone HIP SDK has **LIMITED** native Windows support
- ✅ **WSL2 provides FULL ROCm support** (recommended path)

## Prerequisites

- **Windows 11** (22H2 or later)
- **AMD Radeon RX 7900 XTX** (gfx1100) or other supported RDNA3/RDNA4 GPU
- **WSL2** installed (for full ROCm stack support)
- **16GB+ RAM** recommended for compilation

## Installation Options

### Option 1: WSL2 + Ubuntu (RECOMMENDED - Full ROCm Support)

WSL2 provides **complete ROCm 7.1.1 support** for RX 7900 XTX. This is the most reliable path.

**Step 1: Install WSL2**

```powershell
# In Windows PowerShell (Admin)
wsl --install -d Ubuntu-22.04
wsl --update
```

**Step 2: Install AMD GPU Driver for WSL**

```powershell
# Download and install AMD driver for WSL2
# Visit: https://www.amd.com/en/support/download/linux-drivers.html
# Look for "AMD GPU Driver for WSL" or install from Radeon Software Adrenalin
```

**Step 3: Install ROCm in WSL Ubuntu**

```bash
# Inside WSL Ubuntu terminal
# Add ROCm repository
sudo mkdir -p /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.1.1 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

# Update and install ROCm
sudo apt update
sudo apt install rocm-hip-sdk rocm-libs -y

# Install rocBLAS and rocSOLVER specifically
sudo apt install rocblas rocsolver -y
```

**Step 4: Configure User Permissions**

```bash
# Add user to render and video groups
sudo usermod -a -G render,video $USER
newgrp render
```

**Step 5: Verify Installation**

```bash
# Check ROCm installation
rocm-smi
rocminfo | grep "Name:"

# Check HIP compiler
hipcc --version

# Verify RX 7900 XTX is detected
rocminfo | grep "gfx1100"
```

Expected output:

```text
Name:                    gfx1100
```

**Step 6: Set Environment Variables**

```bash
# Add to ~/.bashrc
echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
echo 'export PATH=$ROCM_PATH/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: Windows Native (Limited - PyTorch Bundled Only)

**WARNING**: As of December 2025, standalone HIP SDK for Windows has **limited support**. The recommended path is PyTorch-bundled ROCm.

## Building the rocBLAS Backend

### Windows Build

Create `build_rocblas_tests.ps1`:

```powershell
param(
    [string]$RocmPath = "C:\Program Files\AMD\ROCm"
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building rocBLAS Backend Tests" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Using ROCm at: $RocmPath"

$hipcc = "$RocmPath\bin\hipcc.exe"
if (-not (Test-Path $hipcc)) {
    Write-Host "ERROR: hipcc not found at $hipcc" -ForegroundColor Red
    exit 1
}

Write-Host "HIPCC: $hipcc"

# Compile rocBLAS trait implementation
Write-Host "`nCompiling rocBLAS trait implementation..."
$cmd = "$hipcc -c " +
       "-I include " +
       "-I `"$RocmPath\include`" " +
       "src\backends\gpu\rocblas_trait_impl.c " +
       "-o build\tests\rocblas_trait_impl.obj"

Write-Host "Command: $cmd"
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to compile rocblas_trait_impl.c" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Compiled rocblas_trait_impl.c" -ForegroundColor Green

# Compile test file
Write-Host "`nCompiling test_rocblas_basic.c..."
$cmd = "$hipcc -c " +
       "-I include " +
       "-I `"$RocmPath\include`" " +
       "tests\test_rocblas_basic.c " +
       "-o build\tests\test_rocblas_basic.obj"

Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to compile test_rocblas_basic.c" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Compiled test_rocblas_basic.c" -ForegroundColor Green

# Link executable
Write-Host "`nLinking executable..."
$cmd = "$hipcc " +
       "build\tests\rocblas_trait_impl.obj " +
       "build\tests\test_rocblas_basic.obj " +
       "-lrocblas -lrocsolver " +
       "-o build\tests\test_rocblas_basic.exe"

Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to link executable" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Linked test_rocblas_basic.exe" -ForegroundColor Green

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Executable: build\tests\test_rocblas_basic.exe"
Write-Host "`nTo run tests:"
Write-Host "  cd build\tests"
Write-Host "  .\test_rocblas_basic.exe"
```

**Step 1: Install PyTorch with ROCm on Windows**

```powershell
# Create a Python virtual environment
python -m venv venv_rocm
.\venv_rocm\Scripts\Activate.ps1

# Install PyTorch with ROCm 7.1.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**Step 2: Extract rocBLAS Libraries**

PyTorch bundles rocBLAS. You can access it via:

```powershell
# Find PyTorch ROCm libraries
python -c "import torch; import os; print(os.path.dirname(torch.__file__))"
# Example: C:\Users\YourName\venv_rocm\Lib\site-packages\torch
```

Libraries are in: `torch\lib\` (includes `rocblas.dll`, `hip_hcc.dll`)

**Step 3: Limitations**

This approach provides:

- ✅ HIP runtime (via PyTorch)
- ✅ rocBLAS library (via PyTorch)
- ❌ No `hipcc` compiler
- ❌ No standalone development

**For full development capabilities, use WSL2 (Option 1).**

## Building the rocBLAS Backend

### WSL2 Build (Recommended)

Create `build_rocblas_tests.sh` in WSL:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Building rocBLAS Backend Tests (WSL2)"
echo "=========================================="

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
echo "Using ROCm at: $ROCM_PATH"

# Check hipcc
if ! command -v hipcc &> /dev/null; then
    echo "ERROR: hipcc not found. Install ROCm first."
    exit 1
fi

hipcc --version

# Create build directory
mkdir -p build/tests

# Compile rocBLAS trait implementation
echo ""
echo "Compiling rocBLAS trait implementation..."
hipcc -c \
    -I include \
    -I "$ROCM_PATH/include" \
    src/backends/gpu/rocblas_trait_impl.c \
    -o build/tests/rocblas_trait_impl.o

echo "✓ Compiled rocblas_trait_impl.c"

# Compile test file
echo ""
echo "Compiling test_rocblas_basic.c..."
hipcc -c \
    -I include \
    -I "$ROCM_PATH/include" \
    tests/test_rocblas_basic.c \
    -o build/tests/test_rocblas_basic.o

echo "✓ Compiled test_rocblas_basic.c"

# Link executable
echo ""
echo "Linking executable..."
hipcc \
    build/tests/rocblas_trait_impl.o \
    build/tests/test_rocblas_basic.o \
    -L"$ROCM_PATH/lib" \
    -lrocblas -lrocsolver -lamdhip64 \
    -o build/tests/test_rocblas_basic

echo "✓ Linked test_rocblas_basic"

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Executable: build/tests/test_rocblas_basic"
echo ""
echo "To run tests:"
echo "  cd build/tests"
echo "  ./test_rocblas_basic"
```

**Build Command:**

```bash
chmod +x build_rocblas_tests.sh
./build_rocblas_tests.sh
```

### Windows Build (If you have standalone HIP SDK)

**PowerShell Script (if HIP SDK installed):**

```powershell
.\build_rocblas_tests.ps1 -RocmPath "C:\Program Files\AMD\ROCm"
```

**Note**: This requires a standalone HIP SDK installation on Windows, which has limited availability. **Use WSL2 for reliable builds.**

## Troubleshooting

### WSL2 Issues

**"GPU not detected"**

```bash
# Check if GPU is visible in WSL
rocm-smi

# Expected output:
# GPU[0] : Navi 31 [Radeon RX 7900 XTX]
```

If not detected:

1. Update Windows to latest version
2. Update WSL: `wsl --update`
3. Install AMD GPU driver for WSL from AMD website
4. Restart WSL: `wsl --shutdown` then restart

**"rocBLAS not found"**

```bash
# Verify rocBLAS installation
ls /opt/rocm/lib/librocblas.so

# Reinstall if missing
sudo apt install rocblas rocsolver -y
```

**"hipcc: command not found"**

```bash
# Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH

# Make permanent
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Windows Native Issues

**"HIP runtime error"**

If using PyTorch-bundled ROCm:

```powershell
# Verify PyTorch can see GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Expected: True, and "AMD Radeon RX 7900 XTX"
```

### GPU Detection

**Check RX 7900 XTX is recognized:**

WSL2:

```bash
rocminfo | grep -A 10 "gfx1100"
```

Windows (via AMD Software):

```powershell
# Open AMD Software: Adrenalin Edition
# Go to Performance → Tuning
# Should show RX 7900 XTX details
```

## Performance Expectations

### RX 7900 XTX Specifications (RDNA3)

- **GPU Architecture**: RDNA3 (gfx1100 / Navi 31)
- **Compute Units**: 96 CUs (6144 Stream Processors)
- **VRAM**: 24GB GDDR6
- **Memory Bandwidth**: 960 GB/s
- **FP32 Performance**: ~61 TFLOPS
- **FP64 Performance**: ~1.9 TFLOPS (1:32 ratio - consumer GPU)
- **TDP**: 355W

### Comparison with RTX 4090 (cuBLAS)

| Metric                   | RTX 4090 (cuBLAS) | RX 7900 XTX (rocBLAS) | Winner |
| ------------------------ | ----------------- | --------------------- | ------ |
| BLAS Operations          | 198               | **204** ✅             | AMD    |
| VRAM                     | 24GB              | 24GB                  | Tie    |
| FP32 TFLOPS              | ~83               | ~61                   | NVIDIA |
| FP64 Performance         | ~1.3 TFLOPS       | ~1.9 TFLOPS           | AMD    |
| Memory Bandwidth         | 1008 GB/s         | 960 GB/s              | NVIDIA |
| Complex Symmetric Packed | ❌ Not supported   | ✅ **Supported**       | AMD    |
| TDP                      | 450W              | 355W                  | AMD    |
| Linux Support            | Full              | Full                  | Tie    |
| Windows Support          | Full              | Limited (PyTorch)     | NVIDIA |
| WSL2 Support             | Full              | Full                  | Tie    |

**Summary**: RTX 4090 has better raw FP32 performance, but RX 7900 XTX has **more complete BLAS support** and better FP64 performance for scientific computing.

## Multi-GPU Strategy

With both GPUs, you can:

1. **NVIDIA RTX 4090** → Fast FP32 workloads, tensor operations
2. **AMD RX 7900 XTX** → FP64 workloads, complete BLAS (204 ops), complex symmetric packed operations

This gives you **best of both worlds**!

## Next Steps

### For WSL2 Path (Recommended)

1. ✅ **Install WSL2** with Ubuntu 22.04
2. ✅ **Install ROCm 7.1.1** in WSL (full stack)
3. ✅ **Build rocBLAS backend** using `build_rocblas_tests.sh`
4. ✅ **Test on RX 7900 XTX** - verify all 204 operations
5. ✅ **Benchmark** against cuBLAS on RTX 4090
6. ✅ **Implement multi-GPU dispatcher** to use both GPUs

### For Windows Native Path (Limited)

1. ⚠️ **Install PyTorch with ROCm** (`pip install torch --index-url ...`)
2. ⚠️ **Extract libraries** from PyTorch installation
3. ⚠️ **Link against bundled rocBLAS**
4. ❌ **Limited development capabilities** (no hipcc)

## Summary: ROCm vs HIP Naming

**Question**: "Has HIP replaced ROCm?"

**Answer**: No! They're different layers:

```text
┌─────────────────────────────────────┐
│         Your Application            │
├─────────────────────────────────────┤
│   HIP API (like CUDA Runtime API)   │  ← Programming interface
├─────────────────────────────────────┤
│  rocBLAS, rocSOLVER (like cuBLAS)   │  ← Math libraries
├─────────────────────────────────────┤
│   ROCm Runtime & Compiler (HCC)     │  ← Execution layer
├─────────────────────────────────────┤
│         AMD GPU Hardware            │
└─────────────────────────────────────┘
```

**ROCm** = The entire stack (platform)  
**HIP** = The programming API within ROCm

AMD's documentation sometimes focuses on "HIP SDK" because that's what developers interact with, but you're still installing/using ROCm.

**For RX 7900 XTX on Windows**: Use **WSL2 for full ROCm support**, or PyTorch-bundled ROCm for limited use.

## References

- [AMD ROCm Documentation (Main)](https://rocm.docs.amd.com/)
- [ROCm on Radeon GPUs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [rocBLAS API Reference](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/)
- [RX 7900 XTX Support Matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
- [RX 7900 XTX Specifications](https://www.amd.com/en/products/graphics/amd-radeon-rx-7900xtx)
