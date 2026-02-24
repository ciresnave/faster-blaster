#!/bin/bash
set -e

# AOCL-BLIS Build Script for WSL
# This script builds BLIS from source with maximum performance optimizations

INSTALL_DIR="${1:-/mnt/c/AOCL-BLIS-Custom}"
BUILD_DIR="/tmp/aocl-blis-build"

echo "============================================"
echo " Building AOCL-BLIS from Source (WSL)"
echo "============================================"
echo ""

# Detect CPU architecture
CPU_INFO=$(cat /proc/cpuinfo | grep "model name" | head -n1)
echo "Detected CPU: $CPU_INFO"

# Auto-detect AMD Zen architecture
if echo "$CPU_INFO" | grep -iq "Zen 4"; then
    BLIS_CONFIG="zen4"
    echo "Using Zen4 optimized configuration"
elif echo "$CPU_INFO" | grep -iq "Zen 3"; then
    BLIS_CONFIG="zen3"
    echo "Using Zen3 optimized configuration"
elif echo "$CPU_INFO" | grep -iq "Zen 2"; then
    BLIS_CONFIG="zen2"
    echo "Using Zen2 optimized configuration"
else
    BLIS_CONFIG="zen3"
    echo "Using Zen3 default configuration"
fi
echo ""

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone BLIS repository
echo "Cloning BLIS repository..."
git clone --depth 1 --branch master https://github.com/amd/blis.git
cd blis

# Configure BLIS
echo ""
echo "Configuring BLIS..."
echo "Configuration: $BLIS_CONFIG"
echo "Install directory: $INSTALL_DIR"
echo ""

./configure \
    --prefix="$INSTALL_DIR" \
    --enable-cblas \
    --enable-threading=openmp \
    --enable-shared \
    --enable-static \
    "$BLIS_CONFIG"

# Build with all cores
NUM_CORES=$(nproc)
echo ""
echo "Building BLIS with $NUM_CORES cores..."
make -j"$NUM_CORES"

# Install
echo ""
echo "Installing BLIS to $INSTALL_DIR..."
make install

echo ""
echo "============================================"
echo " Build Complete!"
echo "============================================"
echo ""
echo "BLIS has been installed to: $INSTALL_DIR"
echo ""
echo "Next steps:"
echo "1. Add to PATH: export PATH=\"$INSTALL_DIR/bin:\$PATH\""
echo "2. Add to LD_LIBRARY_PATH: export LD_LIBRARY_PATH=\"$INSTALL_DIR/lib:\$LD_LIBRARY_PATH\""
echo "3. Rebuild faster-blaster with: cmake -DBLIS_ROOT=\"$INSTALL_DIR\""
echo ""
echo "Verify installation:"
echo "  ls -la $INSTALL_DIR/lib"
echo "  ls -la $INSTALL_DIR/include"
echo ""
