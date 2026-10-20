param(
    [Parameter(Mandatory = $true)][string]$RunDirectory,
    [Parameter(Mandatory = $true)][string]$Minikube
)
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = 'Heart MLOps LoadBalancer - keep open during demo'
Write-Host 'Keep this window open while using the Kubernetes API.'
Write-Host 'If a route permission error appears, rerun the main batch as administrator.'
Write-Host 'After saving screenshots and collecting logs, press Ctrl+C here to stop the tunnel.'
$exitCode = 1
try {
    & $Minikube -p heart-mlops tunnel --bind-address=127.0.0.1 2>&1 |
        Tee-Object -FilePath (Join-Path $RunDirectory 'tunnel.log')
    $exitCode = $LASTEXITCODE
} finally {
    @{ exit_code = $exitCode; ended_at = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $RunDirectory 'tunnel-exit.json')
}
exit $exitCode
