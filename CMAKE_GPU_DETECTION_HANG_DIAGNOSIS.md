# CMake GPU Detection Hang — Root Cause Analysis & Fixes

## Summary

Three root causes identified, in order of impact:

| # | Location | Operation | Worst-case cost |
|---|----------|-----------|-----------------|
| 1 | `cmake/BackendInstaller.cmake:62` | `find_package(hip QUIET)` with no guard | minutes on broken ROCm registry |
| 2 | `cmake/BackendInstaller.cmake:17-28` | `detect_cpu_vendor()` → PowerShell WMI | 5–30 s every reconfigure |
| 3 | `cmake/BackendInstaller.cmake:120+` | Duplicate `find_package(CUDAToolkit)` | seconds pre-cache |

---

## Cause 1 (Primary): `find_package(hip QUIET)` without existence guard

**File**: `cmake/BackendInstaller.cmake` line 62  
**Called from**: `detect_gpus()` → top-level `if(AUTO_DETECT_BACKENDS)` block (line 365)  
**Runs on every**: `cmake ..` reconfigure

### Why it hangs

`find_package(hip QUIET)` (lowercase "hip", config-mode) makes CMake search:

1. `CMAKE_PREFIX_PATH` entries
2. `HIP_DIR` / `hip_DIR` env/cache vars
3. **Windows package registry**: `HKLM\SOFTWARE\Kitware\CMake\Packages\hip`

If any stale/partial AMD ROCm installation left entries in the Windows registry, CMake _finds_ and then _loads_ `hipConfig.cmake`. AMD's hip package config internally attempts to:
- Run `hipcc --version` or `clang-cl --version` to detect the HIP compiler
- On some versions: call a python helper that probes GPU device count via HIP API

When the driver is present but the toolkit is corrupt, or when `hipcc` is missing from PATH while its registry entry exists, these sub-processes can block indefinitely waiting for device enumeration.

**Confirmation**: If you see cmake hanging with no output after `"cuda Toolkit: ..."`, this is the culprit.

### Fix

```cmake
# cmake/BackendInstaller.cmake  — detect_gpus() function, ~line 62
# BEFORE (hangs):
find_package(hip QUIET)
if(hip_FOUND)
    ...
endif()

# AFTER (guarded):
# Only attempt HIP discovery if ROCm appears to be physically installed.
# Avoids loading a broken hipConfig.cmake from a stale registry entry.
set(_HIP_CANDIDATE_PATHS
    "$ENV{ROCM_PATH}"
    "$ENV{HIP_PATH}"
    "C:/Program Files/AMD/ROCm"
    "/opt/rocm"
)
set(_HIP_LOOKS_PRESENT FALSE)
foreach(_p IN LISTS _HIP_CANDIDATE_PATHS)
    if(EXISTS "${_p}")
        set(_HIP_LOOKS_PRESENT TRUE)
        break()
    endif()
endforeach()

if(_HIP_LOOKS_PRESENT)
    find_package(hip QUIET)
    if(hip_FOUND)
        set(${OUT_HAS_AMD} TRUE PARENT_SCOPE)
        message(STATUS "  AMD ROCm/HIP: Found")
    endif()
else()
    message(STATUS "  AMD ROCm/HIP: Not present (no ROCm directory found)")
endif()
unset(_HIP_LOOKS_PRESENT)
unset(_HIP_CANDIDATE_PATHS)
```

---

## Cause 2: PowerShell WMI query in `detect_cpu_vendor()`

**File**: `cmake/BackendInstaller.cmake` lines 17–28  
**Called from**: top-level `if(AUTO_DETECT_BACKENDS)` block (line 361)  
**Runs on every**: `cmake ..` reconfigure — **never cached**

### Why it is slow

```cmake
execute_process(
    COMMAND powershell -NoProfile -Command "Get-CimInstance -ClassName Win32_Processor ..."
    ...
)
```

Every cmake reconfigure:
1. Spawns a fresh `powershell.exe` process (~1–3 s startup)
2. Makes a WMI query via `Get-CimInstance` (~2–5 s, longer under enterprise group policies)
3. Total: **5–30 s per reconfigure** — worse under VMs or OneDrive-backed workspaces

Additionally, PowerShell `Get-CimInstance` can hang up to 60 s if the WMI service is busy or under restrictive Windows Security policy.

### Fix — cache the result

```cmake
function(detect_cpu_vendor OUT_VENDOR OUT_MODEL)
    # Cache so we only pay this cost once per CMake cache lifetime.
    if(DEFINED FB_CACHED_CPU_VENDOR AND DEFINED FB_CACHED_CPU_MODEL)
        set(${OUT_VENDOR} "${FB_CACHED_CPU_VENDOR}" PARENT_SCOPE)
        set(${OUT_MODEL}  "${FB_CACHED_CPU_MODEL}"  PARENT_SCOPE)
        return()
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x86")
        if(WIN32)
            # Use wmic (always available, faster than powershell + WMI) 
            execute_process(
                COMMAND wmic cpu get Name /value
                OUTPUT_VARIABLE CPU_INFO
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                TIMEOUT 10   # <-- don't block cmake forever
            )
            # wmic output: "Name=Intel(R) Core(TM) ..." -- strip key
            string(REGEX REPLACE ".*Name=" "" CPU_INFO "${CPU_INFO}")
        else()
            execute_process(
                COMMAND sh -c "grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2"
                OUTPUT_VARIABLE CPU_INFO
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                TIMEOUT 5
            )
        endif()
        ...
    endif()

    # Write to CMake cache so subsequent reconfigures skip the query entirely.
    set(FB_CACHED_CPU_VENDOR "${vendor}" CACHE INTERNAL "Detected CPU vendor")
    set(FB_CACHED_CPU_MODEL  "${model}"  CACHE INTERNAL "Detected CPU model string")
    set(${OUT_VENDOR} "${vendor}" PARENT_SCOPE)
    set(${OUT_MODEL}  "${model}"  PARENT_SCOPE)
endfunction()
```

Key changes:
- **Cache guard** using `CACHE INTERNAL` variables — second and subsequent `cmake ..` calls return instantly
- **Replace `powershell Get-CimInstance` with `wmic cpu get Name`** — `wmic.exe` launches in ~200 ms (no JIT, no profile loading)
- **Add `TIMEOUT 10`** to all `execute_process` calls so a hung subprocess can never block cmake forever

---

## Cause 3: Duplicate `find_package(CUDAToolkit)` calls

**Files**: `cmake/BackendInstaller.cmake` (inside `detect_gpus` and `check_gpu_backends`) and `CMakeLists.txt` line 123

### Impact

CMake caches `find_package` results _within a single cmake run_ after the first call, so duplicate calls in the same project are normally cheap. However, the **first** call (inside `detect_gpus`) occurs before caching is established, and CMake's `FindCUDAToolkit.cmake` internally runs:

```
execute_process(COMMAND nvcc --version)
```

On a system where `nvcc` is in PATH but requires environment initialisation (e.g., the CUDA env-vars were set by a batch file that isn't in cmake's env), this process can take 5–15 s.

### Fix — deduplicate explicitly

In `CMakeLists.txt`, move the single authoritative `find_package(CUDAToolkit QUIET)` call to **before** `include(cmake/BackendInstaller.cmake)`. Then in `BackendInstaller.cmake`'s `detect_gpus` and `check_gpu_backends`, check the already-set cache variable instead of calling `find_package` again:

```cmake
# cmake/BackendInstaller.cmake  — detect_gpus(), ~line 55
# BEFORE:
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)

# AFTER (read the result already set by the main CMakeLists.txt):
if(CUDAToolkit_FOUND)   # reads CMake cache — zero cost
```

---

## Additional Observations

### `BackendInstaller.cmake` runs unconditionally

`cmake/BackendInstaller.cmake` is `include()`d at CMakeLists.txt line 24 with the default `AUTO_DETECT_BACKENDS=ON`. The entire backend detection block (CPU query + GPU probes + backend scans) runs on **every** `cmake ..` invocation, including trivially-triggered re-runs from ninja touching CMakeLists.txt.

To suppress for CI/scripted builds, pass `-DAUTO_DETECT_BACKENDS=OFF`.

### OneDrive amplification

The workspace is under `OneDrive\Documents\projects\`. CMake reads/writes many temporary files during configure. Files that are "online only" (not locally pinned) incur network round-trips for stat/open calls. This amplifies all the above costs. Consider pinning the project folder to "Always keep on this device" in OneDrive settings, or moving the `build-clang/` directory to a local (non-OneDrive) path.

---

## Recommended Apply Order

1. **Immediate (highest impact)**: Add `_HIP_LOOKS_PRESENT` guard around `find_package(hip QUIET)` in `detect_gpus()` — eliminates the hang entirely for machines without ROCm.

2. **Quick win**: Add `CACHE INTERNAL` guard + switch to `wmic` in `detect_cpu_vendor()` — eliminates 5–30 s per reconfigure.

3. **Cleanup**: Deduplicate `find_package(CUDAToolkit)` to eliminate the double-call pattern.

4. **Optional**: Add `-DAUTO_DETECT_BACKENDS=OFF` to CI/automation cmake invocations so the probe block never runs in headless environments.
