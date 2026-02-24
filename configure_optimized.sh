#!/bin/bash
# Configure and Build faster-blaster with Source-Built Backends
# This script builds all backend libraries from source with maximum performance optimizations

set -e

SKIP_BLIS=0
SKIP_OPENBLAS=0
SKIP_CLBLAST=0
CLEAN=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-blis)
            SKIP_BLIS=1
            shift
            ;;
        --skip-openblas)
            SKIP_OPENBLAS=1
            shift
            ;;
        --skip-clblast)
            SKIP_CLBLAST=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --help)
            cat <<EOF
Configure and Build faster-blaster with Optimized Backends

Usage: ./configure_optimized.sh [options]

Options:
  --skip-blis      Don't build BLIS from source
  --skip-openblas  Don't build OpenBLAS from source
  --skip-clblast   Don't build CLBlast from source
  --clean          Clean build directory before configuring
  --help           Show this help message

Examples:
  # Build everything with max optimizations:
  ./configure_optimized.sh

  # Clean build with everything:
  ./configure_optimized.sh --clean

  # Build only BLIS and OpenBLAS:
  ./configure_optimized.sh --skip-clblast

Description:
  This script configures faster-blaster to build all backend libraries
  from source with CPU/GPU-specific optimizations for maximum performance.
  
  The build process will:
  - Auto-detect your CPU (AMD Zen2/3/4, Intel Skylake/Ice Lake, etc.)
  - Auto-detect your GPU (NVIDIA CUDA, AMD ROCm, OpenCL)
  - Build BLIS with architecture-specific kernels
  - Build OpenBLAS with optimal targets
  - Build CLBlast for GPU acceleration
  - Apply maximum compiler optimizations (-march=native, etc.)
  
  All built libraries are installed to: build/backends-install/
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================================================"
echo "  faster-blaster - Maximum Performance Configuration"
echo "============================================================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake not found. Please install CMake 3.15 or later."
    exit 1
fi
echo "  ✓ CMake: $(cmake --version | head -1)"

if ! command -v make &> /dev/null; then
    echo "  ⚠ make not found - install build tools (build-essential on Debian/Ubuntu)"
else
    echo "  ✓ make: Available"
fi

if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
    echo "  ⚠ No C compiler found - install gcc or clang"
else
    if command -v gcc &> /dev/null; then
        echo "  ✓ gcc: $(gcc --version | head -1)"
    fi
    if command -v clang &> /dev/null; then
        echo "  ✓ clang: $(clang --version | head -1)"
    fi
fi

if command -v clinfo &> /dev/null; then
    echo "  ✓ OpenCL: Available"
elif command -v nvidia-smi &> /dev/null || command -v rocminfo &> /dev/null; then
    echo "  ✓ GPU detected"
else
    echo "  ⚠ OpenCL not detected - CLBlast will be skipped"
    SKIP_CLBLAST=1
fi

echo ""

# Clean build directory if requested
if [ $CLEAN -eq 1 ]; then
    if [ -d "build" ]; then
        echo "Cleaning build directory..."
        rm -rf build
        echo "  ✓ Build directory cleaned"
    fi
fi

# Create build directory
mkdir -p build
cd build

# Configure CMake options
CMAKE_ARGS=(
    ".."
    "-DFB_BUILD_BACKENDS_FROM_SOURCE=ON"
)

if [ $SKIP_BLIS -eq 0 ]; then
    echo "BLIS will be built from source with CPU-specific optimizations"
    CMAKE_ARGS+=("-DFB_BUILD_BLIS_FROM_SOURCE=ON")
else
    CMAKE_ARGS+=("-DFB_BUILD_BLIS_FROM_SOURCE=OFF")
fi

if [ $SKIP_OPENBLAS -eq 0 ]; then
    echo "OpenBLAS will be built from source with CPU-specific optimizations"
    CMAKE_ARGS+=("-DFB_BUILD_OPENBLAS_FROM_SOURCE=ON")
else
    CMAKE_ARGS+=("-DFB_BUILD_OPENBLAS_FROM_SOURCE=OFF")
fi

if [ $SKIP_CLBLAST -eq 0 ]; then
    echo "CLBlast will be built from source for GPU acceleration"
    CMAKE_ARGS+=("-DFB_BUILD_CLBLAST_FROM_SOURCE=ON")
else
    CMAKE_ARGS+=("-DFB_BUILD_CLBLAST_FROM_SOURCE=OFF")
fi

echo ""
echo "Configuring faster-blaster..."
echo "CMake command: cmake ${CMAKE_ARGS[*]}"
echo ""

# Run CMake configure
cmake "${CMAKE_ARGS[@]}"

echo ""
echo "============================================================================"
echo "  Configuration complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. Build backends: cmake --build . --target blis-backend openblas-backend"
echo "  2. Build faster-blaster: cmake --build . --config Release"
echo "  3. Run tests: ctest -C Release"
echo ""
echo "Or build everything at once:"
echo "  cmake --build . -j\$(nproc)"
echo ""
echo "Built libraries will be in: build/backends-install/"
echo ""

cd ..
