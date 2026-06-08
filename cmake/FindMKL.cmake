# FindMKL.cmake
# Locate Intel MKL (Math Kernel Library) for CPU
#
# This module defines:
#  MKL_FOUND - System has Intel MKL
#  MKL_INCLUDE_DIRS - MKL include directories
#  MKL_LIBRARIES - Libraries needed to use MKL
#  MKL_VERSION - Version of MKL found

# Search paths for Intel MKL
set(MKL_SEARCH_PATHS
    # Environment variables
    $ENV{MKLROOT}
    $ENV{ONEAPI_ROOT}/mkl
    
    # Windows default locations
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.3"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.2"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.1"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.0"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2024.2"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2024.1"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2024.0"
    "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl"
    
    # Linux default locations
    /opt/intel/oneapi/mkl/latest
    /opt/intel/oneapi/mkl
    /opt/intel/mkl
    /usr/local/intel/oneapi/mkl
    /usr/local/intel/mkl
    
    # macOS default locations
    /opt/intel/oneapi/mkl/latest
    /opt/intel/mkl
    ~/intel/oneapi/mkl/latest
)

# Find include directory
find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    PATHS ${MKL_SEARCH_PATHS}
    PATH_SUFFIXES include
    DOC "Intel MKL include directory"
    NO_DEFAULT_PATH
)

# Find library directory
find_path(MKL_LIB_DIR
    NAMES mkl_core.lib libmkl_core.so libmkl_core.a
    PATHS ${MKL_SEARCH_PATHS}
    PATH_SUFFIXES lib lib/intel64 lib/x64 lib64
    DOC "Intel MKL library directory"
)

# Determine threading layer based on platform
if(WIN32)
    set(MKL_THREADING_LIB "mkl_intel_thread")
    set(MKL_COMPILER_LIB "iomp5md")  # Intel OpenMP
elseif(APPLE)
    set(MKL_THREADING_LIB "mkl_intel_thread")
    set(MKL_COMPILER_LIB "iomp5")
else() # Linux
    set(MKL_THREADING_LIB "mkl_gnu_thread")
    set(MKL_COMPILER_LIB "gomp")  # GNU OpenMP
endif()

# Determine interface layer based on architecture
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(MKL_INTERFACE_LIB "mkl_intel_lp64")  # 64-bit
else()
    set(MKL_INTERFACE_LIB "mkl_intel")  # 32-bit
endif()

# Find MKL libraries (standard linking line)
set(MKL_LIB_NAMES
    ${MKL_INTERFACE_LIB}
    ${MKL_THREADING_LIB}
    mkl_core
)

set(MKL_LIBRARIES "")
foreach(lib_name ${MKL_LIB_NAMES})
    find_library(MKL_${lib_name}_LIBRARY
        NAMES ${lib_name}
        PATHS ${MKL_LIB_DIR}
        NO_DEFAULT_PATH
    )
    if(MKL_${lib_name}_LIBRARY)
        list(APPEND MKL_LIBRARIES ${MKL_${lib_name}_LIBRARY})
    endif()
endforeach()

# Find Intel OpenMP runtime library (separate search — iomp5md lives in the
# Intel compiler runtime dir, NOT in the MKL lib dir)
if(WIN32)
    find_library(MKL_OMP_LIBRARY
        NAMES iomp5md libiomp5md
        PATHS
            "C:/Program Files (x86)/Intel/oneAPI/2025.3/lib"
            "C:/Program Files (x86)/Intel/oneAPI/2025.2/lib"
            "C:/Program Files (x86)/Intel/oneAPI/2025.1/lib"
            "C:/Program Files (x86)/Intel/oneAPI/2025.0/lib"
            "C:/Program Files (x86)/Intel/oneAPI/2024.2/lib"
            "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib"
            "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64"
            $ENV{ONEAPI_ROOT}/compiler/latest/lib
    )
    if(MKL_OMP_LIBRARY)
        list(APPEND MKL_LIBRARIES ${MKL_OMP_LIBRARY})
        message(STATUS "  Intel OpenMP runtime: ${MKL_OMP_LIBRARY}")
    else()
        message(WARNING "  Intel OpenMP runtime (iomp5md.lib) not found — mkl_intel_thread requires it")
    endif()
elseif(UNIX AND NOT APPLE)
    # Linux: try to find gomp or iomp5
    find_library(MKL_OMP_LIBRARY NAMES gomp iomp5)
    if(MKL_OMP_LIBRARY)
        list(APPEND MKL_LIBRARIES ${MKL_OMP_LIBRARY})
    endif()
endif()

# Extract version from mkl_version.h
if(MKL_INCLUDE_DIR)
    find_file(MKL_VERSION_FILE
        NAMES mkl_version.h
        PATHS ${MKL_INCLUDE_DIR}
        NO_DEFAULT_PATH
    )
    
    if(MKL_VERSION_FILE)
        file(STRINGS ${MKL_VERSION_FILE} MKL_VERSION_STRINGS
            REGEX "__INTEL_MKL__|INTEL_MKL_VERSION")
        
        # Try to extract major.minor.patch
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" MKL_VERSION 
            "${MKL_VERSION_STRINGS}")
        
        # Fallback: try year format (2024.x)
        if(NOT MKL_VERSION)
            string(REGEX MATCH "20[0-9][0-9]\\.[0-9]" MKL_VERSION 
                "${MKL_VERSION_STRINGS}")
        endif()
    endif()
endif()

# Handle find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL
    REQUIRED_VARS MKL_INCLUDE_DIR MKL_LIBRARIES
    VERSION_VAR MKL_VERSION
)

# Set output variables
if(MKL_FOUND)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    
    # Print found information
    message(STATUS "Found Intel MKL:")
    message(STATUS "  Include: ${MKL_INCLUDE_DIR}")
    message(STATUS "  Libraries: ${MKL_LIBRARIES}")
    message(STATUS "  Interface: ${MKL_INTERFACE_LIB} (${CMAKE_SIZEOF_VOID_P}-byte pointers)")
    message(STATUS "  Threading: ${MKL_THREADING_LIB}")
    if(MKL_VERSION)
        message(STATUS "  Version: ${MKL_VERSION}")
    endif()
    
    # Provide linking advice
    message(STATUS "  Note: You may also need to link: ${MKL_COMPILER_LIB} (OpenMP runtime)")
endif()

mark_as_advanced(
    MKL_INCLUDE_DIR
    MKL_LIB_DIR
    MKL_LIBRARIES
    MKL_INTERFACE_LIB
    MKL_THREADING_LIB
)
