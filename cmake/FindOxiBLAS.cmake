# FindOxiBLAS.cmake
# Locate OxiBLAS FFI library (pure Rust BLAS/LAPACK with SIMD)
#
# This module defines:
#  OXIBLAS_FOUND        - System has OxiBLAS FFI
#  OXIBLAS_FFI_LIBRARY  - Library needed to use OxiBLAS
#  OXIBLAS_VERSION      - Version of OxiBLAS found (e.g. "0.1.0")
#
# OxiBLAS exports Fortran-convention symbols only (saxpy_, dgemm_, …).
# No C/CBLAS header is installed — no include dir is needed at compile time.
# The faster-blaster plugin uses fb_enumerate_and_populate() to auto-load
# all Fortran-convention symbols and fills CBLAS slots via conv_thunks.

# Search paths for OxiBLAS FFI
set(OXIBLAS_SEARCH_PATHS
    # Build-from-source installation (highest priority)
    "${CMAKE_BINARY_DIR}/backends-install/oxiblas"
    "${FB_BACKENDS_INSTALL_PREFIX}/oxiblas"

    # Environment variables
    $ENV{OXIBLAS_ROOT}
    $ENV{OXIBLAS_HOME}

    # Cargo default output locations
    "$ENV{CARGO_TARGET_DIR}/release"
    "${CMAKE_SOURCE_DIR}/../oxiblas/target/release"
    "${CMAKE_SOURCE_DIR}/extern/oxiblas/target/release"

    # vcpkg installations
    "$ENV{VCPKG_ROOT}/installed/x64-windows"
    "C:/vcpkg/installed/x64-windows"
    "C:/dev/vcpkg/installed/x64-windows"

    # Windows default locations
    "C:/libraries/oxiblas"
    "C:/Program Files/OxiBLAS"

    # Linux package manager / manual install
    /usr
    /usr/local
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /opt/oxiblas

    # macOS Homebrew/MacPorts
    /usr/local/opt/oxiblas
    /opt/homebrew/opt/oxiblas
    /opt/local
)

# Find library (prefer shared so that runtime dlopen probing works)
find_library(OXIBLAS_FFI_LIBRARY
    # Cargo builds liboxiblas_ffi.so / .dylib / .dll / .a
    NAMES oxiblas_ffi libboxiblas_ffi
          # Windows import lib name
          oxiblas_ffi.lib
    PATHS ${OXIBLAS_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 bin release
    DOC "OxiBLAS FFI library (shared or static)"
)

# OxiBLAS does not install a C header.  Version comes from the Cargo metadata
# embedded in the library path, or is hardcoded to the known shipped version.
if(NOT OXIBLAS_VERSION)
    if(OXIBLAS_FFI_LIBRARY)
        # Future: parse Cargo.toml next to the library
        set(OXIBLAS_VERSION "0.1.0")
    endif()
endif()

# Handle standard find_package arguments (QUIET / REQUIRED)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OxiBLAS
    REQUIRED_VARS OXIBLAS_FFI_LIBRARY
    VERSION_VAR   OXIBLAS_VERSION
)

if(OXIBLAS_FOUND)
    # Alias so callers can use target_link_libraries(… ${OXIBLAS_LIBRARIES})
    set(OXIBLAS_LIBRARIES ${OXIBLAS_FFI_LIBRARY})

    message(STATUS "Found OxiBLAS:")
    message(STATUS "  Library: ${OXIBLAS_FFI_LIBRARY}")
    message(STATUS "  Version: ${OXIBLAS_VERSION}")
    message(STATUS "  Convention: Fortran (saxpy_, dgemm_, …) — CBLAS slots via conv_thunks")
endif()

mark_as_advanced(
    OXIBLAS_FFI_LIBRARY
    OXIBLAS_LIBRARIES
)
