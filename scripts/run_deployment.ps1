param([switch]$CollectOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$utf8 = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:HEART_DISEASE_PROJECT_ROOT = $projectRoot
$reportRoot = Join-Path $projectRoot 'reports\deployment'
New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
$runName = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 6)
$runDirectory = Join-Path $reportRoot $runName
New-Item -ItemType Directory -Path $runDirectory | Out-Null
$bootstrapLog = Join-Path $runDirectory 'bootstrap.log'
$transcribing = $false
$runnerStarted = $false
$exitCode = 1

function Invoke-CheckedNative {
    param([string]$Executable, [string[]]$NativeArguments)
    # Windows PowerShell treats native stderr as ErrorRecords; nonzero exit is
    # checked explicitly so ordinary pip progress on stderr is not a false failure.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Executable @NativeArguments 2>&1 | ForEach-Object { Write-Host "$_" }
    $nativeCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($nativeCode -ne 0) { throw "Command failed (exit $nativeCode): $Executable $NativeArguments" }
}

function Write-DiagnosticArchive {
    param([string]$SourceDirectory, [string]$DestinationPath)
    Add-Type -AssemblyName System.IO.Compression
    $sourcePath = [IO.Path]::GetFullPath($SourceDirectory).TrimEnd('\') + '\'
    $temporaryPath = $DestinationPath + '.' + [Guid]::NewGuid().ToString('N') + '.partial'
    $outputStream = $null
    $zip = $null
    try {
        $outputStream = [IO.File]::Open($temporaryPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $zip = New-Object IO.Compression.ZipArchive($outputStream, [IO.Compression.ZipArchiveMode]::Create)
        foreach ($file in (Get-ChildItem -LiteralPath $SourceDirectory -Recurse -File -Force -ErrorAction Stop)) {
            $entryName = $file.FullName.Substring($sourcePath.Length).Replace('\', '/')
            $inputStream = $null
            $entryStream = $null
            try {
                # The tunnel intentionally keeps writing its log during the demo.
                # Share its writer handle and copy only the bytes present now.
                $sharing = [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete
                $inputStream = [IO.File]::Open($file.FullName, [IO.FileMode]::Open, [IO.FileAccess]::Read, $sharing)
                $remaining = $inputStream.Length
                $entryStream = $zip.CreateEntry($entryName).Open()
                $buffer = New-Object byte[] 65536
                while ($remaining -gt 0) {
                    $count = $inputStream.Read($buffer, 0, [int][Math]::Min($buffer.Length, $remaining))
                    if ($count -eq 0) { break }
                    $entryStream.Write($buffer, 0, $count)
                    $remaining -= $count
                }
            } finally {
                if ($entryStream) { $entryStream.Dispose() }
                if ($inputStream) { $inputStream.Dispose() }
            }
        }
        $zip.Dispose()
        $zip = $null
        $outputStream.Dispose()
        $outputStream = $null
        # Publish only a completed ZIP; retain any previous ZIP on read failure.
        if ([IO.File]::Exists($DestinationPath)) {
            [IO.File]::Replace($temporaryPath, $DestinationPath, [NullString]::Value)
        } else {
            [IO.File]::Move($temporaryPath, $DestinationPath)
        }
    } finally {
        if ($zip) { $zip.Dispose() }
        if ($outputStream) { $outputStream.Dispose() }
        if ([IO.File]::Exists($temporaryPath)) { [IO.File]::Delete($temporaryPath) }
    }
}

try {
    Start-Transcript -Path $bootstrapLog | Out-Null
    $transcribing = $true
    if (-not $CollectOnly) {
        $latest = @{ directory = "reports/deployment/$runName" } | ConvertTo-Json
        [IO.File]::WriteAllText((Join-Path $reportRoot 'latest.json'), $latest, $utf8)
    }
    Write-Host "Reports: $runDirectory"
    Write-Host 'Initial dependency setup and the Docker build can take several minutes.'
    $dockerBin = Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin'
    if (Test-Path -LiteralPath $dockerBin) { $env:Path = "$env:Path;$dockerBin" }
    $deploymentPython = Join-Path $projectRoot '.deployment-venv\Scripts\python.exe'
    $interpreterArguments = @()
    if ($CollectOnly) {
        if (-not (Test-Path -LiteralPath $deploymentPython)) {
            # The collector uses only Python's standard library, including after
            # a prerequisite failure that prevented environment creation.
            $launcher = Get-Command py -ErrorAction SilentlyContinue
            if ($launcher) {
                $deploymentPython = $launcher.Source
                $interpreterArguments = @('-3.13')
            } else {
                $launcher = Get-Command python -ErrorAction SilentlyContinue
                if (-not $launcher) { throw 'Install Python 3.13 to collect live diagnostics. The previous run ZIP is already available.' }
                $deploymentPython = $launcher.Source
            }
        }
    } else {
        foreach ($tool in @('docker', 'minikube', 'kubectl')) {
            if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
                throw "Missing $tool. Follow deployment\windows\START_HERE.md, then rerun."
            }
        }
        if (-not (Test-Path -LiteralPath $deploymentPython)) {
            $launcher = Get-Command py -ErrorAction SilentlyContinue
            if ($launcher) {
                Invoke-CheckedNative -Executable $launcher.Source -NativeArguments @('-3.13', '-m', 'venv', '.deployment-venv')
            } else {
                $launcher = Get-Command python -ErrorAction SilentlyContinue
                if (-not $launcher) { throw 'Install Python 3.13 with the Python launcher or add it to PATH.' }
                Invoke-CheckedNative -Executable $launcher.Source -NativeArguments @('-c', 'import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 1)')
                Invoke-CheckedNative -Executable $launcher.Source -NativeArguments @('-m', 'venv', '.deployment-venv')
            }
        }
        Invoke-CheckedNative -Executable $deploymentPython -NativeArguments @('-c', 'import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 1)')
        # Use serving/build dependencies, not the much larger training/MLflow environment.
        Invoke-CheckedNative -Executable $deploymentPython -NativeArguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements-runtime.txt', '-r', 'requirements-package.txt')
        Invoke-CheckedNative -Executable $deploymentPython -NativeArguments @('-m', 'pip', 'check')
    }
    $runnerArguments = $interpreterArguments + @('scripts/deployment_runner.py', '--output', $runDirectory)
    if ($CollectOnly) { $runnerArguments += '--collect-only' }
    $runnerStarted = $true
    Invoke-CheckedNative -Executable $deploymentPython -NativeArguments $runnerArguments
    $exitCode = 0
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    $failureStage = 'bootstrap'
    if ($runnerStarted) { $failureStage = 'deployment-runner' }
    $failureStatus = 'blocked'
    if ($runnerStarted) { $failureStatus = 'failed' }
    $failure = @{ status = $failureStatus; stage = $failureStage; error = $_.Exception.Message; cluster_success_claimed = $false }
    [IO.File]::WriteAllText((Join-Path $runDirectory 'launcher-failure.json'), ($failure | ConvertTo-Json -Depth 6), $utf8)
    if (-not $runnerStarted) {
        [IO.File]::WriteAllText((Join-Path $runDirectory 'summary.json'), ($failure | ConvertTo-Json -Depth 6), $utf8)
    }
    $resultFile = Join-Path $runDirectory 'RESULTS.txt'
    if (-not (Test-Path -LiteralPath $resultFile)) {
        $readable = "RESULT: $failureStatus`r`nStage: $failureStage`r`nReason: $($_.Exception.Message)`r`nRead deployment\windows\START_HERE.md and upload this run's diagnostics ZIP if unresolved.`r`n"
        [IO.File]::WriteAllText($resultFile, $readable, $utf8)
    }
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    # Repack after the transcript is closed, including failures before Python ran.
    try {
        $archive = "$runDirectory-diagnostics.zip"
        Write-DiagnosticArchive -SourceDirectory $runDirectory -DestinationPath $archive
        Write-Host "UPLOAD THIS FILE FOR DIAGNOSIS: $archive" -ForegroundColor Cyan
    } catch {
        Write-Host "ZIP creation failed: $($_.Exception.Message). Upload files from $runDirectory instead."
        $exitCode = 1
    }
    Write-Host "Read: $runDirectory\RESULTS.txt (or launcher-failure.json if setup failed)."
}
exit $exitCode
