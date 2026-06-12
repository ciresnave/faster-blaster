param(
    [string]$BuildScript,
    [string]$SourceDir,
    [string]$InstallDir,
    [string]$Target = "ZEN",
    [int]$NumCores = 32
)

$ErrorActionPreference = "Stop"

$vsDevShell = "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/Tools/Launch-VsDevShell.ps1"
if (Test-Path $vsDevShell) {
    & $vsDevShell -Arch amd64 -HostArch amd64 | Out-Null
}

& $BuildScript -SourceDir $SourceDir -InstallDir $InstallDir -Target $Target -NumCores $NumCores
exit $LASTEXITCODE
