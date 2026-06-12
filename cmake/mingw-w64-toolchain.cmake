# Toolchain file for MinGW-w64 cross-compilation
# Produces Windows PE/COFF binaries from WSL/Linux

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)

# Specify the cross-compilers
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_Fortran_COMPILER x86_64-w64-mingw32-gfortran)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)

# Target environment location
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)

# Adjust the default behavior of FIND_XXX() commands to search in
# the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Make sure CMake doesn't try to run executables (we're cross-compiling)
set(CMAKE_CROSSCOMPILING_EMULATOR "")

# Set the target file extensions
set(CMAKE_EXECUTABLE_SUFFIX ".exe")
set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")

# Compiler flags for Windows target
set(CMAKE_C_FLAGS_INIT "-D_WIN32_WINNT=0x0601 -DWINVER=0x0601")
set(CMAKE_CXX_FLAGS_INIT "-D_WIN32_WINNT=0x0601 -DWINVER=0x0601")
