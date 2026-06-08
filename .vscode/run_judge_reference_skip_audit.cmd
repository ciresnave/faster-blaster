@echo off
setlocal

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_judge_reference_skip_audit.ps1"
exit /b %ERRORLEVEL%