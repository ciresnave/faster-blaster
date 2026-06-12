@echo off
setlocal
set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
pushd "%REPO_ROOT%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%build_low_memory.ps1" -RunFocusedTests
set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%
