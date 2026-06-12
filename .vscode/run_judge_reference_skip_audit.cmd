@echo off
setlocal

set "CMAKE_BUILD_PARALLEL_LEVEL=1"
set "CTEST_PARALLEL_LEVEL=1"
set "NINJAFLAGS=-j1"
set "OMP_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "BLIS_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "VECLIB_MAXIMUM_THREADS=1"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_judge_reference_skip_audit.ps1"
exit /b %ERRORLEVEL%