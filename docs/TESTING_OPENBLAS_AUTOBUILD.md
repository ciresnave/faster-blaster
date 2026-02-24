# Testing OpenBLAS Auto-Build from Source

## Step 1: Uninstall vcpkg OpenBLAS

Run this in PowerShell:

```powershell
cd C:\libraries\vcpkg
.\vcpkg remove openblas
.\vcpkg remove openblas:x64-windows
```

## Step 2: Clean and Reconfigure

```powershell
cd C:\Users\cires\OneDrive\Documents\projects\faster-blaster
Remove-Item build -Recurse -Force
mkdir build
cd build
cmake ..
```

## Expected Output:

You should see:
```
══════════════════════════════════════════════════════════════
 Backend Detection & Configuration
══════════════════════════════════════════════════════════════
CPU: AMD Ryzen 9 7940HX with Radeon Graphics
CPU Vendor: AMD

CPU Backends:
  ✗ OpenBLAS: Not found
  ✓ Intel MKL: (found)
  ✗ AOCL-BLIS: Not found

💡 OpenBLAS not found
   We can build OpenBLAS from source with full threading support
   → Building threaded OpenBLAS from source (BUILD_OPENBLAS_FROM_SOURCE=ON)

-- Building OpenBLAS from source with threading support...
-- OpenBLAS will be built with threading support enabled
-- Build will occur during 'cmake --build'
```

## Step 3: Build

```powershell
cmake --build . --config Release
```

This will:
1. Download OpenBLAS 0.3.27 from GitHub
2. Build it with `-DUSE_THREAD=1 -DUSE_OPENMP=1`
3. Install to `build/external/install`
4. Build faster-blaster linking to the threaded OpenBLAS

**Note**: First build will take 10-15 minutes to compile OpenBLAS.

## Step 4: Test Threading

```powershell
$env:Path = "build\Release;build\external\install\bin;" + $env:Path
cd build\tests\Release
.\test_device_api.exe
```

Expected output:
```
[OpenBLAS] Thread control: get=0x<address>, set=0x<address>
[OpenBLAS] Threading capability verified: 16 threads
✅ All 28/28 tests passed (100%)!
```

## Step 5: Verify Performance

The custom-built OpenBLAS should show:
- Multi-threaded execution (16 threads on Ryzen 9 7940HX)
- 4-8x faster than single-threaded vcpkg version
- Thread control working properly

## Rollback (if needed)

To restore vcpkg OpenBLAS:

```powershell
cd C:\libraries\vcpkg
.\vcpkg install openblas

cd C:\Users\cires\OneDrive\Documents\projects\faster-blaster
cmake -B build -DBUILD_OPENBLAS_FROM_SOURCE=OFF
```
