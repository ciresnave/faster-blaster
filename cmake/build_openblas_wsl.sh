#!/bin/bash
# Build OpenBLAS using MinGW-w64 cross-compiler in WSL
# Produces Windows PE/COFF binaries compatible with MSVC
set -e

SOURCE_DIR="$1"
INSTALL_DIR="$2"
TARGET="$3"
NUM_CORES="$4"

echo "================================================================"
echo "  Building OpenBLAS with MinGW-w64 Cross-Compiler"
echo "================================================================"
echo "Source:  $SOURCE_DIR"
echo "Install: $INSTALL_DIR"
echo "Target:  $TARGET"
echo "Cores:   $NUM_CORES"
echo "Compiler: x86_64-w64-mingw32-gcc (produces Windows binaries)"
echo ""

# Create build directory in /tmp (native Linux filesystem)
BUILD_DIR="/tmp/openblas-mingw-build-$$"
echo "Build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake using MinGW-w64 cross-compiler
echo ""
echo "Configuring with CMake..."

# Get toolchain file path (convert Windows path to WSL path if needed)
TOOLCHAIN_FILE="/mnt/c/Users/cires/OneDrive/Documents/projects/faster-blaster/cmake/mingw-w64-toolchain.cmake"

cmake -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DTARGET="$TARGET" \
    -DUSE_OPENMP=ON \
    -DUSE_THREAD=ON \
    -DNUM_THREADS=128 \
    -DDYNAMIC_ARCH=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_STATIC_LIBS=ON \
    -DNO_LAPACKE=OFF \
    "$SOURCE_DIR"

# Build
echo ""
echo "Building OpenBLAS..."
make -j"$NUM_CORES"

# Install
echo ""
echo "Installing to $INSTALL_DIR..."
make install

# Cleanup
cd /
rm -rf "$BUILD_DIR"

echo ""
echo "================================================================"
echo "  Build complete!"
echo "================================================================"
