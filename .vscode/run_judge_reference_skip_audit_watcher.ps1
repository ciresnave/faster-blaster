param(
    [switch]$RunWorker,
    [string]$ArtifactDir = 'c:\Users\cires\OneDrive\Documents\projects\faster-blaster\build-clang\tests',
    [string]$Stamp
)

$status = Join-Path $ArtifactDir 'judge_reference_skip_audit_latest_status.txt'
$tail = Join-Path $ArtifactDir 'judge_reference_skip_audit_latest_tail.txt'
$latestLog = Join-Path $ArtifactDir 'judge_reference_skip_audit_latest.txt'
$latestErr = Join-Path $ArtifactDir 'judge_reference_skip_audit_latest.err.txt'
$completion = Join-Path $ArtifactDir 'judge_reference_skip_audit_latest_completion.txt'
$workerPidFile = Join-Path $ArtifactDir 'judge_reference_skip_audit_latest_worker_pid.txt'
$runnerPidFile = Join-Path $ArtifactDir 'judge_reference_skip_audit_latest_runner_pid.txt'
$buildRoot = Split-Path -Parent $ArtifactDir
$repoRoot = Split-Path -Parent $buildRoot
$referenceRoot = Join-Path (Split-Path -Parent $repoRoot) 'faster-blaster-reference'
$referenceBuildRoot = Join-Path $referenceRoot 'build-extended'
$testExe = Join-Path $ArtifactDir 'test_judge_runner.exe'
$quietSeconds = 3

$script:CurrentStamp = ''
$script:CurrentLog = ''
$script:CurrentErr = ''
$script:CurrentExitCode = -1
$script:CurrentRunnerPid = 0
$script:Watchers = @()
$script:Subscriptions = @()
$script:WatchState = [hashtable]::Synchronized(@{
    Pending = $true
    LastChangeUtc = [DateTime]::UtcNow
    LastChange = (Get-Date -Format o)
    TriggerPath = 'startup'
})

function Write-AtomicFile {
    param(
        [string]$Path,
        [string[]]$Lines,
        [string]$Encoding = 'ascii'
    )

    $tmp = $Path + '.tmp'
    $content = ''
    if ($null -ne $Lines -and $Lines.Count -gt 0) {
        $content = ($Lines -join "`r`n")
    }

    $textEncoding = switch ($Encoding.ToLowerInvariant()) {
        'ascii' { [System.Text.Encoding]::ASCII }
        'utf8' { [System.Text.UTF8Encoding]::new($false) }
        default { [System.Text.UTF8Encoding]::new($false) }
    }

    [System.IO.File]::WriteAllText($tmp, $content, $textEncoding)
    Move-Item $tmp $Path -Force
}

function Get-FileSize {
    param([string]$Path)

    if (Test-Path $Path) {
        return (Get-Item $Path).Length
    }

    return 0
}

function Read-KeyValueFile {
    param([string]$Path)

    $map = [ordered]@{}
    if (-not (Test-Path $Path)) {
        return $map
    }

    $raw = Get-Content $Path -Raw
    $matches = [regex]::Matches($raw, '(?ms)([A-Z_]+)=(.*?)(?=(?:\r?\n|\s+[A-Z_]+=)|$)')
    foreach ($match in $matches) {
        $map[$match.Groups[1].Value] = $match.Groups[2].Value.Trim()
    }

    return $map
}

function Initialize-CurrentAudit {
    $completionData = Read-KeyValueFile $completion

    if ($completionData.Contains('STAMP')) {
        $script:CurrentStamp = $completionData['STAMP']
    }
    if ($completionData.Contains('LOG')) {
        $script:CurrentLog = $completionData['LOG']
    }
    if ($completionData.Contains('ERR')) {
        $script:CurrentErr = $completionData['ERR']
    }
    if ($completionData.Contains('EXIT')) {
        $script:CurrentExitCode = [int]$completionData['EXIT']
    }
}

function Get-ReferenceSkipCount {
    param(
        [string]$LogPath,
        [string]$ErrPath
    )

    $countPath = $null

    if ((Get-FileSize $ErrPath) -gt 0) {
        $countPath = $ErrPath
    } elseif ((Get-FileSize $LogPath) -gt 0) {
        $countPath = $LogPath
    }

    if (-not $countPath) {
        return 0
    }

    try {
        return @(Select-String -Path $countPath -Pattern '^\[REF-SKIP\]').Count
    } catch {
        return 0
    }
}

function Get-TailLines {
    param(
        [string]$State,
        [string]$LogPath,
        [string]$ErrPath,
        [string]$TriggerPath,
        [string]$LastChange,
        [string]$ErrorText
    )

    $lines = @()
    switch ($State) {
        'STARTING' { $lines += 'Watcher launching detached audit controller.' }
        'WATCHING' { $lines += 'Watcher idle. Awaiting changes.' }
        'QUEUED' { $lines += 'Change detected. Audit queued after quiet period.' }
        'RUNNING' { $lines += 'Audit running.' }
        'COMPLETE' { $lines += 'Latest audit completed.' }
        'ERROR' { $lines += 'Watcher encountered an error.' }
    }

    if (-not [string]::IsNullOrWhiteSpace($LastChange)) {
        $lines += 'LAST_CHANGE=' + $LastChange
    }
    if (-not [string]::IsNullOrWhiteSpace($TriggerPath)) {
        $lines += 'TRIGGER_PATH=' + $TriggerPath
    }
    if (-not [string]::IsNullOrWhiteSpace($ErrorText)) {
        $lines += 'ERROR=' + $ErrorText
    }

    if (Test-Path $LogPath) {
        $lines += '--- LOG ---'
        try {
            $lines += @(Get-Content $LogPath -Tail 80)
        } catch {
            $lines += 'Unable to read stdout log tail.'
        }
    }

    if ((Test-Path $ErrPath) -and ((Get-FileSize $ErrPath) -gt 0)) {
        $lines += '--- STDERR ---'
        try {
            $lines += @(Get-Content $ErrPath -Tail 80)
        } catch {
            $lines += 'Unable to read stderr log tail.'
        }
    }

    if ($lines.Count -eq 0) {
        $lines = @('No tail available.')
    }

    return $lines
}

function Write-Status {
    param(
        [string]$State,
        [int]$RunnerPid = 0,
        [int]$ExitCode = -1,
        [bool]$Pending = $false,
        [string]$TriggerPath = '',
        [string]$LastChange = '',
        [string]$ErrorText = ''
    )

    $tailLines = Get-TailLines -State $State -LogPath $script:CurrentLog -ErrPath $script:CurrentErr -TriggerPath $TriggerPath -LastChange $LastChange -ErrorText $ErrorText
    $referenceSkipCount = Get-ReferenceSkipCount -LogPath $script:CurrentLog -ErrPath $script:CurrentErr

    Write-AtomicFile -Path $tail -Encoding 'utf8' -Lines $tailLines
    Write-AtomicFile -Path $status -Encoding 'ascii' -Lines @(
        'STATE=' + $State,
        'STAMP=' + $script:CurrentStamp,
        'LOG=' + $script:CurrentLog,
        'ERR=' + $script:CurrentErr,
        'LATEST_LOG=' + $latestLog,
        'LATEST_ERR=' + $latestErr,
        'LATEST_COMPLETION=' + $completion,
        'TAIL=' + $tail,
        'EXIT=' + $ExitCode,
        'SIZE=' + (Get-FileSize $script:CurrentLog),
        'ERR_SIZE=' + (Get-FileSize $script:CurrentErr),
        'REFERENCE_SKIP_COUNT=' + $referenceSkipCount,
        'RUNNER_PID=' + $RunnerPid,
        'WORKER_PID=' + $PID,
        'PENDING=' + ([int]$Pending),
        'LAST_CHANGE=' + $LastChange,
        'TRIGGER_PATH=' + $TriggerPath,
        'ERROR=' + $ErrorText,
        'UPDATED=' + (Get-Date -Format o)
    )
}

function Write-Completion {
    param(
        [string]$State,
        [int]$RunnerPid = 0,
        [int]$ExitCode = -1,
        [string]$TriggerPath = '',
        [string]$LastChange = '',
        [string]$ErrorText = ''
    )

    Write-AtomicFile -Path $completion -Encoding 'ascii' -Lines @(
        'STATE=' + $State,
        'STAMP=' + $script:CurrentStamp,
        'LOG=' + $script:CurrentLog,
        'ERR=' + $script:CurrentErr,
        'LATEST_LOG=' + $latestLog,
        'LATEST_ERR=' + $latestErr,
        'TAIL=' + $tail,
        'EXIT=' + $ExitCode,
        'SIZE=' + (Get-FileSize $script:CurrentLog),
        'ERR_SIZE=' + (Get-FileSize $script:CurrentErr),
        'REFERENCE_SKIP_COUNT=' + (Get-ReferenceSkipCount -LogPath $script:CurrentLog -ErrPath $script:CurrentErr),
        'RUNNER_PID=' + $RunnerPid,
        'WORKER_PID=' + $PID,
        'LAST_CHANGE=' + $LastChange,
        'TRIGGER_PATH=' + $TriggerPath,
        'ERROR=' + $ErrorText,
        'UPDATED=' + (Get-Date -Format o)
    )
}

function Add-Watcher {
    param(
        [string]$Path,
        [string]$Filter,
        [bool]$IncludeSubdirectories = $false
    )

    if (-not (Test-Path $Path)) {
        return
    }

    $watcher = New-Object System.IO.FileSystemWatcher $Path, $Filter
    $watcher.IncludeSubdirectories = $IncludeSubdirectories
    $watcher.NotifyFilter = [System.IO.NotifyFilters]'FileName, LastWrite, Size, CreationTime'
    $watcher.EnableRaisingEvents = $true
    $script:Watchers += $watcher

    foreach ($eventName in 'Changed', 'Created', 'Deleted', 'Renamed') {
        $subscription = Register-ObjectEvent -InputObject $watcher -EventName $eventName -MessageData $script:WatchState -Action {
                $state = $event.MessageData
                $fullPath = $event.SourceEventArgs.FullPath
                if ([string]::IsNullOrWhiteSpace($fullPath)) {
                    return
                }
                if ($fullPath -like '*judge_reference_skip_audit_*') {
                    return
                }
                $state['Pending'] = $true
                $state['LastChangeUtc'] = [DateTime]::UtcNow
                $state['LastChange'] = (Get-Date -Format o)
                $state['TriggerPath'] = $fullPath
            }
        $script:Subscriptions += $subscription
    }
}

function Sync-RuntimeArtifact {
    param(
        [string]$SourcePath,
        [string]$DestinationPath
    )

    if (-not (Test-Path $SourcePath)) {
        return 'Missing runtime artifact: ' + $SourcePath
    }

    try {
        Copy-Item $SourcePath $DestinationPath -Force
    } catch {
        return $_.Exception.Message -replace "`r?`n", ' '
    }

    return $null
}

function Sync-AuditRuntimeArtifacts {
    $errors = New-Object System.Collections.Generic.List[string]
    $runtimePairs = @(
        @{
            Source = Join-Path $buildRoot 'faster-blaster.dll'
            Destination = Join-Path $ArtifactDir 'faster-blaster.dll'
        },
        @{
            Source = Join-Path $referenceBuildRoot 'libfaster_blaster_reference.dll'
            Destination = Join-Path $ArtifactDir 'libfaster_blaster_reference.dll'
        }
    )

    foreach ($pair in $runtimePairs) {
        $syncError = Sync-RuntimeArtifact -SourcePath $pair.Source -DestinationPath $pair.Destination
        if (-not [string]::IsNullOrWhiteSpace($syncError)) {
            $errors.Add($syncError)
        }
    }

    return $errors
}

function Register-Watchers {
    Add-Watcher -Path (Join-Path $repoRoot 'src') -Filter '*.c' -IncludeSubdirectories $true
    Add-Watcher -Path (Join-Path $repoRoot 'src') -Filter '*.h' -IncludeSubdirectories $true
    Add-Watcher -Path (Join-Path $repoRoot 'include') -Filter '*.h' -IncludeSubdirectories $true
    Add-Watcher -Path (Join-Path $repoRoot '.vscode') -Filter '*.ps1'
    Add-Watcher -Path (Join-Path $repoRoot '.vscode') -Filter '*.cmd'
    Add-Watcher -Path (Join-Path $repoRoot '.vscode') -Filter '*.json'
    Add-Watcher -Path $repoRoot -Filter 'CMakeLists.txt'
    Add-Watcher -Path $buildRoot -Filter 'faster-blaster.dll'
    Add-Watcher -Path $ArtifactDir -Filter 'faster-blaster.dll'
    Add-Watcher -Path $ArtifactDir -Filter 'test_judge_runner.exe'
    Add-Watcher -Path $ArtifactDir -Filter 'libfaster_blaster_reference.dll'

    Add-Watcher -Path (Join-Path $referenceRoot 'src') -Filter '*.c' -IncludeSubdirectories $true
    Add-Watcher -Path (Join-Path $referenceRoot 'include') -Filter '*.h' -IncludeSubdirectories $true
    Add-Watcher -Path $referenceRoot -Filter 'CMakeLists.txt'
    Add-Watcher -Path $referenceRoot -Filter 'libfaster_blaster_reference.dll'
    Add-Watcher -Path $referenceBuildRoot -Filter 'libfaster_blaster_reference.dll'
}

function Start-DetachedWatcher {
    New-Item -ItemType Directory -Path $ArtifactDir -Force | Out-Null

    if (Test-Path $workerPidFile) {
        $existingWorkerPid = Get-Content $workerPidFile -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($existingWorkerPid) {
            $existingWorker = Get-Process -Id $existingWorkerPid -ErrorAction SilentlyContinue
            if ($existingWorker) {
                Write-Output ('WATCHER_ALREADY_RUNNING:' + $existingWorker.Id)
                Write-Output ('STATUS:' + $status)
                Write-Output ('TAIL:' + $tail)
                exit 0
            }
        }
        Remove-Item $workerPidFile -ErrorAction SilentlyContinue
    }

    Initialize-CurrentAudit

    $scriptPath = $PSCommandPath
    if ([string]::IsNullOrWhiteSpace($scriptPath)) {
        $scriptPath = $MyInvocation.PSCommandPath
    }
    if ([string]::IsNullOrWhiteSpace($scriptPath)) {
        $scriptPath = $MyInvocation.MyCommand.Path
    }
    if ([string]::IsNullOrWhiteSpace($scriptPath)) {
        throw 'Unable to resolve watcher script path for detached launch.'
    }
    $workerArgs = @(
        '-NoProfile',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        $scriptPath,
        '-RunWorker',
        '-ArtifactDir',
        $ArtifactDir
    )

    Write-AtomicFile -Path $tail -Encoding 'utf8' -Lines @('Watcher launching detached audit controller.')
    Write-Status -State 'STARTING' -RunnerPid 0 -ExitCode $script:CurrentExitCode -Pending $true -TriggerPath 'startup' -LastChange (Get-Date -Format o)

    $workerProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList $workerArgs -WindowStyle Hidden -PassThru
    Write-AtomicFile -Path $workerPidFile -Encoding 'ascii' -Lines @($workerProcess.Id)

    Write-Output ('WATCHER_STARTED:' + $workerProcess.Id)
    Write-Output ('STATUS:' + $status)
    Write-Output ('TAIL:' + $tail)
    exit 0
}

function Invoke-Audit {
    param(
        [string]$TriggerPath,
        [string]$LastChange
    )

    Remove-Item Env:FB_JUDGE_START_OP_ID -ErrorAction SilentlyContinue
    Remove-Item Env:FB_JUDGE_END_OP_ID -ErrorAction SilentlyContinue
    Remove-Item Env:FB_JUDGE_OP_FILTER -ErrorAction SilentlyContinue
    Remove-Item Env:FB_JUDGE_TRACE_OP_START -ErrorAction SilentlyContinue
    Remove-Item Env:FB_JUDGE_TRACE_OP_RETURN -ErrorAction SilentlyContinue
    Remove-Item Env:FB_JUDGE_VALIDATE_HEAP -ErrorAction SilentlyContinue

    $env:FB_JUDGE_SUPPRESS_OP_RESULTS = '1'
    $env:FB_JUDGE_FAIL_ON_REFERENCE_SKIP = '1'

    $script:CurrentStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $script:CurrentLog = Join-Path $ArtifactDir ('judge_reference_skip_audit_' + $script:CurrentStamp + '.txt')
    $script:CurrentErr = Join-Path $ArtifactDir ('judge_reference_skip_audit_' + $script:CurrentStamp + '.err.txt')
    $script:CurrentExitCode = -1
    $script:CurrentRunnerPid = 0

    Remove-Item $script:CurrentLog -ErrorAction SilentlyContinue
    Remove-Item $script:CurrentErr -ErrorAction SilentlyContinue
    Remove-Item $latestLog -ErrorAction SilentlyContinue
    Remove-Item $latestErr -ErrorAction SilentlyContinue
    Remove-Item $tail -ErrorAction SilentlyContinue

    $runner = $null
    $errorText = ''

    $syncErrors = Sync-AuditRuntimeArtifacts
    if ($syncErrors.Count -gt 0) {
        $errorText = ($syncErrors -join ' | ')
        Write-Status -State 'ERROR' -RunnerPid $script:CurrentRunnerPid -ExitCode $script:CurrentExitCode -Pending ([bool]$script:WatchState['Pending']) -TriggerPath $TriggerPath -LastChange $LastChange -ErrorText $errorText
        Write-Completion -State 'ERROR' -RunnerPid $script:CurrentRunnerPid -ExitCode $script:CurrentExitCode -TriggerPath $TriggerPath -LastChange $LastChange -ErrorText $errorText
        return
    }

    try {
        $runner = Start-Process -FilePath $testExe -WorkingDirectory $ArtifactDir -RedirectStandardOutput $script:CurrentLog -RedirectStandardError $script:CurrentErr -WindowStyle Hidden -PassThru
        $script:CurrentRunnerPid = $runner.Id
        Write-AtomicFile -Path $workerPidFile -Encoding 'ascii' -Lines @($PID)
        Write-AtomicFile -Path $runnerPidFile -Encoding 'ascii' -Lines @($runner.Id)

        do {
            $runner.Refresh()
            Write-Status -State 'RUNNING' -RunnerPid $runner.Id -ExitCode $script:CurrentExitCode -Pending ([bool]$script:WatchState['Pending']) -TriggerPath $TriggerPath -LastChange $LastChange
            if (-not $runner.HasExited) {
                Start-Sleep -Seconds 2
            }
        } while (-not $runner.HasExited)

        $script:CurrentExitCode = $runner.ExitCode
    } catch {
        $errorText = $_.Exception.Message -replace "`r?`n", ' '
        Write-Status -State 'ERROR' -RunnerPid $script:CurrentRunnerPid -ExitCode $script:CurrentExitCode -Pending ([bool]$script:WatchState['Pending']) -TriggerPath $TriggerPath -LastChange $LastChange -ErrorText $errorText
        Write-Completion -State 'ERROR' -RunnerPid $script:CurrentRunnerPid -ExitCode $script:CurrentExitCode -TriggerPath $TriggerPath -LastChange $LastChange -ErrorText $errorText
        return
    } finally {
        if (Test-Path $script:CurrentLog) {
            Copy-Item $script:CurrentLog $latestLog -Force
        }

        if (Test-Path $script:CurrentErr) {
            Copy-Item $script:CurrentErr $latestErr -Force
        } else {
            Write-AtomicFile -Path $latestErr -Encoding 'utf8' -Lines @('')
        }

        if ([string]::IsNullOrWhiteSpace($errorText)) {
            Write-Status -State 'COMPLETE' -RunnerPid $script:CurrentRunnerPid -ExitCode $script:CurrentExitCode -Pending ([bool]$script:WatchState['Pending']) -TriggerPath $TriggerPath -LastChange $LastChange
            Write-Completion -State 'COMPLETE' -RunnerPid $script:CurrentRunnerPid -ExitCode $script:CurrentExitCode -TriggerPath $TriggerPath -LastChange $LastChange
        }

        Remove-Item $runnerPidFile -ErrorAction SilentlyContinue
        $script:CurrentRunnerPid = 0
        Write-AtomicFile -Path $workerPidFile -Encoding 'ascii' -Lines @($PID)
    }
}

if (-not $RunWorker) {
    Start-DetachedWatcher
}

New-Item -ItemType Directory -Path $ArtifactDir -Force | Out-Null
Initialize-CurrentAudit
Write-AtomicFile -Path $workerPidFile -Encoding 'ascii' -Lines @($PID)
Register-Watchers

try {
    while ($true) {
        $pending = [bool]$script:WatchState['Pending']
        $lastChange = [string]$script:WatchState['LastChange']
        $triggerPath = [string]$script:WatchState['TriggerPath']

        if ($pending) {
            $ready = (([DateTime]::UtcNow - [DateTime]$script:WatchState['LastChangeUtc']).TotalSeconds -ge $quietSeconds)
            if ($ready -and (Test-Path $testExe)) {
                $script:WatchState['Pending'] = $false
                Invoke-Audit -TriggerPath $triggerPath -LastChange $lastChange
                continue
            }

            $errorText = ''
            if (-not (Test-Path $testExe)) {
                $errorText = 'Waiting for test_judge_runner.exe to exist.'
            }
            Write-Status -State 'QUEUED' -RunnerPid $script:CurrentRunnerPid -ExitCode $script:CurrentExitCode -Pending $true -TriggerPath $triggerPath -LastChange $lastChange -ErrorText $errorText
        } else {
            Write-Status -State 'WATCHING' -RunnerPid 0 -ExitCode $script:CurrentExitCode -Pending $false -TriggerPath $triggerPath -LastChange $lastChange
        }

        Start-Sleep -Seconds 1
    }
} catch {
    $errorText = $_.Exception.Message -replace "`r?`n", ' '
    Write-Status -State 'ERROR' -RunnerPid $script:CurrentRunnerPid -ExitCode $script:CurrentExitCode -Pending ([bool]$script:WatchState['Pending']) -TriggerPath ([string]$script:WatchState['TriggerPath']) -LastChange ([string]$script:WatchState['LastChange']) -ErrorText $errorText
    Write-Completion -State 'ERROR' -RunnerPid $script:CurrentRunnerPid -ExitCode $script:CurrentExitCode -TriggerPath ([string]$script:WatchState['TriggerPath']) -LastChange ([string]$script:WatchState['LastChange']) -ErrorText $errorText
    throw
} finally {
    foreach ($subscription in $script:Subscriptions) {
        try { Unregister-Event -SubscriptionId $subscription.Id -ErrorAction SilentlyContinue } catch {}
        try { Remove-Job -Id $subscription.Action.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
    foreach ($watcher in $script:Watchers) {
        try { $watcher.EnableRaisingEvents = $false } catch {}
        try { $watcher.Dispose() } catch {}
    }
    Remove-Item $runnerPidFile -ErrorAction SilentlyContinue
    Remove-Item $workerPidFile -ErrorAction SilentlyContinue
}