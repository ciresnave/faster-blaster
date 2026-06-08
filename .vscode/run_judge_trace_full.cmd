@echo off
setlocal

set "FB_JUDGE_START_OP_ID="
set "FB_JUDGE_END_OP_ID="
set "FB_JUDGE_FAIL_ON_REFERENCE_SKIP="
set "FB_JUDGE_REFERENCE_OP_LIST="
set "FB_JUDGE_VALIDATE_HEAP=1"
set "FB_JUDGE_SUPPRESS_OP_RESULTS=1"
set "FB_JUDGE_TRACE_OP_START=1"
set "FB_JUDGE_TRACE_OP_RETURN=1"

cd /d "c:\Users\cires\OneDrive\Documents\projects\faster-blaster\build-clang\tests"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"

set "LOG=judge_trace_full_%STAMP%.txt"
set "ERR=judge_trace_full_%STAMP%.err.txt"
set "TAIL=judge_trace_full_latest_tail.txt"
set "STATUS=judge_trace_full_latest_status.txt"

del /q "%TAIL%" 2>nul
del /q "%STATUS%" 2>nul

test_judge_runner.exe > "%LOG%" 2> "%ERR%"
set "CODE=%ERRORLEVEL%"

powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 60 | Set-Content '%TAIL%' -Encoding utf8"
powershell -NoProfile -Command "if ((Test-Path '%ERR%') -and ((Get-Item '%ERR%').Length -gt 0)) { Add-Content '%TAIL%' '--- STDERR ---'; Get-Content '%ERR%' -Tail 60 | Add-Content '%TAIL%' }"

for %%F in ("%LOG%") do set "LOG_SIZE=%%~zF"
for %%F in ("%ERR%") do set "ERR_SIZE=%%~zF"

(
echo LOG=%LOG%
echo ERR=%ERR%
echo TAIL=%TAIL%
echo EXIT=%CODE%
echo SIZE=%LOG_SIZE%
echo ERR_SIZE=%ERR_SIZE%
) > "%STATUS%"

type "%TAIL%"
echo LOG:%LOG%
echo ERR:%ERR%
echo TAIL:%TAIL%
echo STATUS:%STATUS%
echo DONE:%CODE%

exit /b %CODE%