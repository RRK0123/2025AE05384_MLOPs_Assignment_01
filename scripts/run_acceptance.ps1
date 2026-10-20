param([string]$Config = 'configs\acceptance.json')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:MLFLOW_DISABLE_AGENT_HINT = '1'
$runName = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$bootstrap = Join-Path $projectRoot "reports\acceptance\bootstrap-$runName"
New-Item -ItemType Directory -Path $bootstrap -Force | Out-Null
$exitCode = 1
$transcribing = $false
$output = Join-Path $projectRoot "reports\acceptance\$runName"
@{ directory = $output; bootstrap = $bootstrap } | ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $projectRoot 'reports\acceptance\latest-launch.json') -Encoding UTF8

function Invoke-AcceptanceNative {
    param([string]$Executable, [string[]]$NativeArguments)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @NativeArguments 2>&1 | ForEach-Object { Write-Host "$_" }
        $nativeCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($nativeCode -ne 0) { throw "Command exited ${nativeCode}: $Executable $NativeArguments" }
}

try {
    Start-Transcript -Path (Join-Path $bootstrap 'bootstrap.log') | Out-Null
    $transcribing = $true
    $dockerBin = Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin'
    if (Test-Path -LiteralPath $dockerBin) { $env:Path = "$env:Path;$dockerBin" }
    foreach ($tool in @('docker', 'kubectl', 'minikube')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "Missing $tool. Read docs\acceptance-suite.md prerequisites."
        }
    }
    $python = Join-Path $projectRoot '.acceptance-venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            Invoke-AcceptanceNative -Executable $launcher.Source -NativeArguments @('-3.13', '-m', 'venv', '.acceptance-venv')
        } else {
            $launcher = Get-Command python -ErrorAction SilentlyContinue
            if (-not $launcher) { throw 'Python 3.13 is required; install it before launching.' }
            Invoke-AcceptanceNative -Executable $launcher.Source -NativeArguments @('-c', 'import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 1)')
            Invoke-AcceptanceNative -Executable $launcher.Source -NativeArguments @('-m', 'venv', '.acceptance-venv')
        }
    }
    Write-Host 'Preparing the full training, testing and recording environment. First setup needs internet.'
    Invoke-AcceptanceNative -Executable $python -NativeArguments @('-c', 'import sys; sys.exit(0 if sys.version_info[:2] == (3,13) else 1)')
    Invoke-AcceptanceNative -Executable $python -NativeArguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements-acceptance.txt')
    Invoke-AcceptanceNative -Executable $python -NativeArguments @('-m', 'pip', 'check')
    $browserFound = $false
    foreach ($candidate in @('C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', 'C:\Program Files\Microsoft\Edge\Application\msedge.exe', 'C:\Program Files\Google\Chrome\Application\chrome.exe', $env:ACCEPTANCE_BROWSER)) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { $browserFound = $true }
    }
    if (-not $browserFound) {
        Invoke-AcceptanceNative -Executable $python -NativeArguments @('-m', 'playwright', 'install', 'chromium')
    }
    $output = Join-Path $projectRoot "reports\acceptance\$runName"
    Write-Host 'Recording begins after dependency setup, before the automated tests.'
    Invoke-AcceptanceNative -Executable $python -NativeArguments @('-m', 'scripts.acceptance.run', '--track', 'windows', '--config', $Config, '--output', $output)
    $exitCode = 0
} catch {
    $message = $_.Exception.Message
    Write-Host "FAILED: $message" -ForegroundColor Red
    @{ status = 'failed'; error = $message; complete_acceptance = $false; bootstrap_log = 'bootstrap.log' } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $bootstrap 'failure.json') -Encoding UTF8
    $escaped = [Net.WebUtility]::HtmlEncode($message)
    "<!doctype html><meta charset='utf-8'><title>Bootstrap failed</title><h1>Bootstrap failed</h1><p>$escaped</p><p>The automated tests have not completed. Coverage is unavailable and downstream stages are blocked.</p><p>Read bootstrap.log and failure.json. Correct the prerequisite and rerun RUN_ALL.bat.</p>" |
        Set-Content -LiteralPath (Join-Path $bootstrap 'START_HERE.html') -Encoding UTF8
    Write-Host "Bootstrap evidence: $bootstrap"
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    $report = Join-Path $output 'START_HERE.html'
    if (-not (Test-Path -LiteralPath $report)) { $report = Join-Path $bootstrap 'START_HERE.html' }
    if (Test-Path -LiteralPath $report) { Start-Process -FilePath $report }
}
exit $exitCode
