@echo off
REM Build oneMKL trait library separately with Intel DPC++ compiler

echo Initializing Intel oneAPI environment...
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

echo.
echo Configuring oneMKL trait with Intel compiler...
cmake -S . -B build -G "Ninja" -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% NEQ 0 (
    echo Configuration failed!
    exit /b %ERRORLEVEL%
)

echo.
echo Building oneMKL trait library...
cmake --build build --config Release

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b %ERRORLEVEL%
)

echo.
echo oneMKL trait library built successfully!
echo Output: build\onemkl_trait.lib
