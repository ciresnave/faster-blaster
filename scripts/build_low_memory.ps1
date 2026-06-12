param(
    [string]$BuildDir = "build-clang",
    [switch]$RunFocusedTests,
    [string]$LogDir = "",
    [int]$SampleSeconds = 2,
    [switch]$EnableLowRamDumpTrigger,
    [int]$LowRamThresholdMb = 4096,
    [int]$MaxDumpsPerStep = 2,
    [string]$DumpProcRegex = "clang-cl|clang|lld-link|link|ninja|cmake",
    [string]$ProcdumpPath = "",
    [switch]$UseFullDump
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogDir = Join-Path $RepoRoot "build-logs/low-mem-$stamp"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$DumpDir = Join-Path $LogDir "dumps"
if ($EnableLowRamDumpTrigger) {
    New-Item -ItemType Directory -Force -Path $DumpDir | Out-Null
}

$MemoryCsv = Join-Path $LogDir "memory_samples.csv"
"timestamp,phase,proc,pid,ws_mb,pm_mb,cpu_s,sys_free_mb,sys_total_mb" | Out-File -FilePath $MemoryCsv -Encoding ascii -Force

$DumpManifestCsv = Join-Path $LogDir "dump_manifest.csv"
"timestamp,phase,pid,proc,ws_mb,sys_free_mb,dump_path" | Out-File -FilePath $DumpManifestCsv -Encoding ascii -Force

$LowRamEventsLog = Join-Path $LogDir "low_ram_events.log"

$script:StepStats = @{}
$script:StepDumpCounts = @{}
$script:CapturedDumpKeys = @{}
$script:ProcdumpExe = $null
$script:DumpTool = "none"

function Resolve-DumpTool {
    if (-not $EnableLowRamDumpTrigger) {
        return $null
    }

    if (-not [string]::IsNullOrWhiteSpace($ProcdumpPath)) {
        if (Test-Path $ProcdumpPath) {
            $script:DumpTool = "procdump"
            return (Resolve-Path $ProcdumpPath).Path
        }
        throw "EnableLowRamDumpTrigger is set but ProcdumpPath does not exist: $ProcdumpPath"
    }

    $candidates = @("procdump64.exe", "procdump.exe")
    foreach ($name in $candidates) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $script:DumpTool = "procdump"
            return $cmd.Source
        }
    }

    $comsvcsPath = Join-Path $env:WINDIR "System32/comsvcs.dll"
    if (Test-Path $comsvcsPath) {
        $script:DumpTool = "comsvcs"
        return $comsvcsPath
    }

    throw "EnableLowRamDumpTrigger is set, but neither procdump nor comsvcs.dll is available."
}

function Update-StepStats {
    param(
        [string]$Phase,
        [string]$Proc,
        [int]$ProcId,
        [double]$WsMb,
        [double]$FreeMb
    )

    if (-not $script:StepStats.ContainsKey($Phase)) {
        $script:StepStats[$Phase] = [ordered]@{
            samples = 0
            peak_ws_mb = 0.0
            peak_proc = ""
            peak_pid = 0
            min_sys_free_mb = [double]::PositiveInfinity
        }
    }

    $s = $script:StepStats[$Phase]
    $s.samples += 1
    if ($WsMb -gt [double]$s.peak_ws_mb) {
        $s.peak_ws_mb = $WsMb
        $s.peak_proc = $Proc
        $s.peak_pid = $ProcId
    }
    if ($FreeMb -lt [double]$s.min_sys_free_mb) {
        $s.min_sys_free_mb = $FreeMb
    }
}

function Try-CaptureLowRamDump {
    param(
        [string]$Phase,
        [double]$FreeMb,
        [array]$Procs
    )

    if (-not $EnableLowRamDumpTrigger) {
        return
    }
    if ($FreeMb -ge $LowRamThresholdMb) {
        return
    }

    if (-not $script:StepDumpCounts.ContainsKey($Phase)) {
        $script:StepDumpCounts[$Phase] = 0
    }
    if ([int]$script:StepDumpCounts[$Phase] -ge $MaxDumpsPerStep) {
        return
    }

    $targets = $Procs |
        Where-Object { $_.ProcessName -match $DumpProcRegex } |
        Sort-Object WorkingSet64 -Descending

    foreach ($p in $targets) {
        if ([int]$script:StepDumpCounts[$Phase] -ge $MaxDumpsPerStep) {
            break
        }

        $dumpKey = "$Phase-$($p.Id)"
        if ($script:CapturedDumpKeys.ContainsKey($dumpKey)) {
            continue
        }

        $tsToken = Get-Date -Format "yyyyMMdd_HHmmss"
        $dumpPath = Join-Path $DumpDir ("{0}_{1}_{2}.dmp" -f $tsToken, $p.ProcessName, $p.Id)
        $dumpArg = if ($UseFullDump) { "-ma" } else { "-mp" }

        try {
            if ($script:DumpTool -eq "procdump") {
                $args = @("-accepteula", $dumpArg, $p.Id, $dumpPath)
                $dumpProc = Start-Process -FilePath $script:ProcdumpExe -ArgumentList $args -PassThru -WindowStyle Hidden
                $dumpProc.WaitForExit()
            } elseif ($script:DumpTool -eq "comsvcs") {
                # comsvcs MiniDump requires full path and keyword token (mini/full)
                $miniToken = if ($UseFullDump) { "full" } else { "mini" }
                $miniArgs = "`"$script:ProcdumpExe`", MiniDump $($p.Id) `"$dumpPath`" $miniToken"
                $dumpProc = Start-Process -FilePath "rundll32.exe" -ArgumentList $miniArgs -PassThru -WindowStyle Hidden
                $dumpProc.WaitForExit()
            } else {
                throw "No dump tool configured"
            }

            if ($dumpProc.ExitCode -eq 0 -and (Test-Path $dumpPath)) {
                $script:CapturedDumpKeys[$dumpKey] = $true
                $script:StepDumpCounts[$Phase] = [int]$script:StepDumpCounts[$Phase] + 1
                $now = (Get-Date).ToString("o")
                $wsMb = [math]::Round($p.WorkingSet64 / 1MB, 2)
                "$now,$Phase,$($p.Id),$($p.ProcessName),$wsMb,$FreeMb,$dumpPath" | Add-Content -Path $DumpManifestCsv
                "[$now] captured dump for $($p.ProcessName) pid=$($p.Id) free_mb=$FreeMb path=$dumpPath" | Add-Content -Path $LowRamEventsLog
                Write-Host "[low-mem] Dump captured for $($p.ProcessName) pid=$($p.Id): $dumpPath" -ForegroundColor Yellow
            }
        } catch {
            $now = (Get-Date).ToString("o")
            "[$now] dump capture failed for $($p.ProcessName) pid=$($p.Id): $($_.Exception.Message)" | Add-Content -Path $LowRamEventsLog
        }
    }
}

function Write-StepSummary {
    param([string]$Phase)

    if (-not $script:StepStats.ContainsKey($Phase)) {
        Write-Host "[low-mem] Step '$Phase' summary: no samples" -ForegroundColor DarkYellow
        return
    }

    $s = $script:StepStats[$Phase]
    $peakWs = [math]::Round([double]$s.peak_ws_mb, 2)
    $minFree = if ([double]::IsPositiveInfinity([double]$s.min_sys_free_mb)) {
        -1
    } else {
        [math]::Round([double]$s.min_sys_free_mb, 2)
    }
    $dumpCount = if ($script:StepDumpCounts.ContainsKey($Phase)) { [int]$script:StepDumpCounts[$Phase] } else { 0 }

    Write-Host "[low-mem] Step '$Phase' summary: peak_proc=$($s.peak_proc) pid=$($s.peak_pid) peak_ws_mb=$peakWs min_sys_free_mb=$minFree samples=$($s.samples) dumps=$dumpCount" -ForegroundColor Cyan
}

if ($EnableLowRamDumpTrigger) {
    $script:ProcdumpExe = Resolve-DumpTool
    Write-Host "[low-mem] Low RAM dump trigger enabled: threshold=${LowRamThresholdMb}MB max_dumps_per_step=$MaxDumpsPerStep tool=$script:DumpTool path=$script:ProcdumpExe" -ForegroundColor Yellow
}

function Write-MemorySample {
    param([string]$Phase)

    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $freeMb = [math]::Round($os.FreePhysicalMemory / 1024.0, 2)
    $totalMb = [math]::Round($os.TotalVisibleMemorySize / 1024.0, 2)
    $ts = (Get-Date).ToString("o")

    $interesting = @("Code", "cmake", "ninja", "clang-cl", "clang", "lld-link", "link", "powershell", "pwsh")
    $procs = Get-Process | Where-Object { $interesting -contains $_.ProcessName }

    foreach ($p in $procs) {
        $wsMb = [math]::Round($p.WorkingSet64 / 1MB, 2)
        $pmMb = [math]::Round($p.PagedMemorySize64 / 1MB, 2)
        Update-StepStats -Phase $Phase -Proc $p.ProcessName -ProcId $p.Id -WsMb $wsMb -FreeMb $freeMb
        $line = "$ts,$Phase,$($p.ProcessName),$($p.Id),$wsMb,$pmMb,$($p.CPU),$freeMb,$totalMb"
        Add-Content -Path $MemoryCsv -Value $line
    }

    Try-CaptureLowRamDump -Phase $Phase -FreeMb $freeMb -Procs $procs
}

function Invoke-LoggedStep {
    param(
        [string]$Name,
        [string]$Command
    )

    $safe = ($Name -replace "[^A-Za-z0-9_-]", "_")
    $stdoutPath = Join-Path $LogDir "$safe.stdout.log"
    $stderrPath = Join-Path $LogDir "$safe.stderr.log"
    $cmdPath = Join-Path $LogDir "$safe.command.txt"

    $Command | Out-File -FilePath $cmdPath -Encoding ascii -Force
    Write-Host "[low-mem] Running step '$Name'" -ForegroundColor Cyan

    $proc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/d", "/c", $Command `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    while (-not $proc.HasExited) {
        Write-MemorySample -Phase $Name
        Start-Sleep -Seconds $SampleSeconds
    }
    Write-MemorySample -Phase "$Name-final"
    Write-StepSummary -Phase $Name
    Write-StepSummary -Phase "$Name-final"

    if ($proc.ExitCode -ne 0) {
        throw "Step '$Name' failed with exit code $($proc.ExitCode). See $stdoutPath and $stderrPath"
    }
}

Write-Host "[low-mem] Configuring with conservative parallelism..." -ForegroundColor Cyan
$env:CMAKE_BUILD_PARALLEL_LEVEL = "1"

Invoke-LoggedStep -Name "configure" -Command "cmake -S . -B $BuildDir -G Ninja -DFB_BACKEND_BUILD_JOBS=4 -DFB_LOW_MEMORY_THUNK_COMPILE=ON"

Write-Host "[low-mem] Building core + focused tests serially..." -ForegroundColor Cyan
Invoke-LoggedStep -Name "build-focused" -Command "cmake --build $BuildDir --target faster-blaster test_device_api test_openblas_backend -- -j1"

if ($RunFocusedTests) {
    Write-Host "[low-mem] Running focused tests..." -ForegroundColor Cyan
    Invoke-LoggedStep -Name "ctest-focused" -Command "ctest --test-dir $BuildDir -R \"OpenBLAS_Backend_Tests|^device_api$\" --output-on-failure"
}

Write-Host "[low-mem] Completed." -ForegroundColor Green
Write-Host "[low-mem] Logs: $LogDir" -ForegroundColor Green
