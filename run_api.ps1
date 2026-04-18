$env:PYTHONUNBUFFERED = "1"
$proc = Start-Process python -ArgumentList "api_server.py" -WorkingDirectory "D:\Heat-wave-backend" -PassThru -NoNewWindow
Start-Sleep 25
if (-not $proc.HasExited) {
    Write-Host "Server running on PID: $($proc.Id)"
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/api/health" -UseBasicParsing -TimeoutSec 5
        Write-Host "Health check: $($response.Content)"
    } catch {
        Write-Host "Health check failed: $_"
    }
} else {
    Write-Host "Server exited with code: $($proc.ExitCode)"
}